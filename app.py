# app.py
import os, sys, io, json, re
from typing import Optional, Dict, Any, List

# Asegura que la raíz del proyecto está en el path de importación
ROOT = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
from openai import OpenAI, BadRequestError

from config import (
    APP_TITLE,
    OPENAI_MODEL,
    OPENAI_API_KEY,
    OPENAI_TEMPERATURE,   # no se usará en UI; temperatura fija 0.2
    MODELS_CATALOG,       # no se usará; dejamos import por compatibilidad
    ADMIN_USER,
    ADMIN_PASSWORD,
    MAX_TOKENS_PER_REQUEST,
)

# Servicios locales ya existentes
from services.pdf_loader import extract_pdf_text
from services.schema import OfertaAnalizada
from services.openai_client import analyze_text_chunk, merge_offers
from utils.text import clean_text, chunk_text
from components.ui import render_header

# ---------------------------------------------------------------------
# Parámetros de esta versión
# ---------------------------------------------------------------------
AVAILABLE_MODELS = ["gpt-4o", "gpt-4o-mini"]
FIXED_TEMPERATURE = 0.2
CHUNK_MAX_CHARS_DEFAULT = 12_000  # usado en fallback map-reduce
SECOND_TAB_KEY = "full_view_ready"


# ---------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")


# ---------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------
def login_gate():
    """Bloquea la UI hasta hacer login; mantiene botón de logout en la sidebar."""
    if st.session_state.get("is_auth", False):
        with st.sidebar:
            if st.button("Cerrar sesión"):
                st.session_state.clear()
                st.rerun()
        return True

    st.title("Acceso")
    with st.form("login_form"):
        u = st.text_input("Usuario")
        p = st.text_input("Contraseña", type="password")
        submitted = st.form_submit_button("Entrar")
    if submitted:
        if u == ADMIN_USER and p == ADMIN_PASSWORD:
            st.session_state["is_auth"] = True
            st.success("Acceso concedido.")
            st.rerun()
        else:
            st.error("Credenciales inválidas.")
    st.stop()


# ---------------------------------------------------------------------
# OpenAI (Responses) – utilidades robustas
# ---------------------------------------------------------------------
if not OPENAI_API_KEY:
    st.error("No se encontró OPENAI_API_KEY. Configura `.env`, variables de entorno o `st.secrets`.")
    st.stop()

_oai_client = OpenAI(api_key=OPENAI_API_KEY)

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r'^\s*```(?:json)?\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s*```\s*$', '', s)
    return s

def _extract_json_block(text: str) -> str:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise RuntimeError("La respuesta del modelo no contiene JSON parseable.")
    return m.group(0)

def _json_loads_robust(raw: Any) -> Any:
    if raw is None:
        raise RuntimeError("Respuesta vacía del modelo.")
    if not isinstance(raw, str):
        return raw
    s = _strip_code_fences(raw)
    if not s:
        raise RuntimeError("El modelo devolvió cadena vacía.")
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        brace = _extract_json_block(s)
        return json.loads(brace)

def _coalesce_text_from_responses(rsp) -> Optional[str]:
    txt = getattr(rsp, "output_text", None)
    if txt:
        return txt
    out = getattr(rsp, "output", None)
    if out:
        parts = []
        for item in out:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for block in content:
                    val = getattr(block, "text", None)
                    if val:
                        parts.append(val)
                    elif isinstance(block, dict):
                        v = block.get("text") or block.get("value")
                        if v:
                            parts.append(v)
        if parts:
            return "\n".join(parts)
    msg = getattr(rsp, "message", None)
    if msg and isinstance(getattr(msg, "content", None), list):
        parts = []
        for block in msg.content:
            val = getattr(block, "text", None)
            if val:
                parts.append(val)
            elif isinstance(block, dict):
                v = block.get("text") or block.get("value")
                if v:
                    parts.append(v)
        if parts:
            return "\n".join(parts)
    return None

def _is_temperature_error(e: Exception) -> bool:
    s = str(e)
    return ("temperature" in s) and ("Unsupported value" in s or "unsupported_value" in s or "does not support" in s)

def _is_unsupported_param(e: Exception, param: str) -> bool:
    s = str(e)
    return ("unsupported_parameter" in s or "Unexpected" in s or "unexpected" in s or "Unknown parameter" in s) and (param in s)

def _responses_create_robust(args: dict):
    """
    Llama a Responses API y se auto-adapta a variaciones del SDK/endpoint:
    - quita temperature si el modelo lo rechaza,
    - quita response_format si el SDK/endpoint no lo soporta,
    - cambia max_output_tokens -> max_completion_tokens,
    - mueve tools/attachments a extra_body si el SDK no los acepta como kwargs.
    """
    a = dict(args)
    extra = a.pop("extra_body", {}) or {}

    for _ in range(6):
        try:
            if extra:
                return _oai_client.responses.create(**a, extra_body=extra)
            else:
                return _oai_client.responses.create(**a)
        except (BadRequestError, TypeError) as e:
            s = str(e)

            if _is_temperature_error(e) or ("unexpected keyword" in s and "temperature" in s):
                a.pop("temperature", None); continue

            if _is_unsupported_param(e, "response_format") or ("unexpected keyword" in s and "response_format" in s):
                a.pop("response_format", None); continue

            if _is_unsupported_param(e, "max_output_tokens") or ("unexpected keyword" in s and "max_output_tokens" in s):
                val = a.pop("max_output_tokens", None)
                if val is not None: a["max_completion_tokens"] = val
                continue

            if _is_unsupported_param(e, "attachments") or ("unexpected keyword" in s and "attachments" in s):
                att = a.pop("attachments", None)
                if att is not None: extra["attachments"] = att
                continue

            if _is_unsupported_param(e, "tools") or ("unexpected keyword" in s and "tools" in s):
                tools = a.pop("tools", None)
                if tools is not None: extra["tools"] = tools
                continue

            raise


def create_vector_store_from_streamlit_files(files, name: str = "RFP Vector Store"):
    """
    Sube los PDFs a OpenAI, los indexa en un Vector Store y
    devuelve (vector_store_id, file_ids) para fallback con 'attachments'.
    """
    store = _oai_client.vector_stores.create(
        name=name,
        expires_after={"anchor": "last_active_at", "days": 2},
    )
    file_ids = []
    for f in files:
        data = f.read()
        up = _oai_client.files.create(
            file=(f.name, data, "application/pdf"),
            purpose="assistants"
        )
        file_ids.append(up.id)
        _oai_client.vector_stores.files.create_and_poll(
            vector_store_id=store.id,
            file_id=up.id
        )
    return store.id, file_ids


# ---------------------------------------------------------------------
# Secciones (prompts para File Search / Map-Reduce)
# ---------------------------------------------------------------------
SECTION_SPECS: Dict[str, Dict[str, str]] = {
    "objetivos_contexto": {
        "titulo": "Objetivos y contexto",
        "user_prompt": (
            "Sobre los PDFs aportados, extrae OBJETIVOS y CONTEXTO del pliego. "
            "Devuelve SOLO JSON con campos:\n"
            "{\n"
            '  "resumen_servicios": str|null,\n'
            '  "objetivos": [str],\n'
            '  "alcance": str|null,\n'
            '  "referencias_paginas": [int]\n'
            "}"
        ),
    },
    "servicios": {
        "titulo": "Servicios solicitados (detalle)",
        "user_prompt": (
            "Sobre los PDFs, lista los SERVICIOS SOLICITADOS con detalle. "
            "Devuelve SOLO JSON con campos:\n"
            "{\n"
            '  "resumen_servicios": str|null,\n'
            '  "servicios_detalle": [{"nombre": str, "descripcion": str|null}],\n'
            '  "alcance": str|null,\n'
            '  "referencias_paginas": [int]\n'
            "}"
        ),
    },
    "importe": {
        "titulo": "Importe de licitación",
        "user_prompt": (
            "Extrae el IMPORTE DE LICITACIÓN. Devuelve SOLO JSON:\n"
            "{\n"
            '  "importe_total": float|null,\n'
            '  "moneda": str|null,\n'
            '  "importes_detalle": [{"concepto": str|null, "importe": float|null, "moneda": str|null, "observaciones": str|null}],\n'
            '  "referencias_paginas": [int]\n'
            "}\n"
            "Usa punto decimal. Si hay base/IVA/prórrogas/anualidades, inclúyelos."
        ),
    },
    "criterios_valoracion": {
        "titulo": "Criterios de valoración",
        "user_prompt": (
            "Extrae CRITERIOS DE VALORACIÓN y subcriterios con máximos/pesos. Devuelve SOLO JSON:\n"
            "{\n"
            '  "criterios_valoracion": [\n'
            '    {"nombre": str, "peso_max": float|null, "tipo": str|null,\n'
            '     "subcriterios": [{"nombre": str, "peso_max": float|null, "tipo": str|null, "observaciones": str|null}]}\n'
            '  ],\n'
            '  "referencias_paginas": [int]\n'
            "}\n"
        ),
    },
    "indice_tecnico": {
        "titulo": "Índice de la respuesta técnica",
        "user_prompt": (
            "Extrae el ÍNDICE SOLICITADO de la respuesta técnica (literal del pliego) "
            "y propone un ÍNDICE ALINEADO. Devuelve SOLO JSON:\n"
            "{\n"
            '  "indice_respuesta_tecnica": [{"titulo": str, "descripcion": str|null, "subapartados": [str]}],\n'
            '  "indice_propuesto": [{"titulo": str, "descripcion": str|null, "subapartados": [str]}],\n'
            '  "referencias_paginas": [int]\n'
            "}\n"
        ),
    },
    "riesgos_exclusiones": {
        "titulo": "Riesgos y exclusiones",
        "user_prompt": (
            "Identifica RIESGOS, DUDAS, EXCLUSIONES o incompatibilidades del pliego. Devuelve SOLO JSON:\n"
            "{\n"
            '  "riesgos_y_dudas": str|null,\n'
            '  "exclusiones": [str],\n'
            '  "referencias_paginas": [int]\n'
            "}\n"
        ),
    },
    "solvencia": {
        "titulo": "Criterios de solvencia",
        "user_prompt": (
            "Extrae criterios de SOLVENCIA (técnica, económica, administrativa). Devuelve SOLO JSON:\n"
            "{\n"
            '  "solvencia": {\n'
            '    "tecnica": [str],\n'
            '    "economica": [str],\n'
            '    "administrativa": [str]\n'
            "  },\n"
            '  "referencias_paginas": [int]\n'
            "}\n"
        ),
    },
}

SYSTEM_PREFIX = (
    "Eres un analista experto en licitaciones públicas en España. "
    "Debes responder SIEMPRE con JSON válido (sin texto adicional). "
    "Si algo no aparece en los PDFs, devuelve null o listas vacías."
)

# ---------------- Conversión a JSON forzada (segunda llamada) ----------------
def _force_jsonify_from_text(section_key: Optional[str], raw_text: str, model: str, temperature: Optional[float]):
    schema_hint = ""
    if section_key and section_key in SECTION_SPECS:
        schema_hint = SECTION_SPECS[section_key]["user_prompt"]

    sys_msg = {
        "role": "system",
        "content": [{"type": "input_text", "text": "Devuelve SOLO un JSON válido (UTF-8), sin texto adicional."}],
    }
    usr_msg = {
        "role": "user",
        "content": [{
            "type": "input_text",
            "text": (
                "Convierte estrictamente la siguiente respuesta a JSON válido que cumpla el esquema indicado. "
                "Si falta información, usa null o listas vacías. "
                f"\n\n[ESQUEMA]\n{schema_hint}\n\n[RESPUESTA]\n<<<\n{raw_text}\n>>>\n"
            ),
        }],
    }

    args = dict(
        model=(model or OPENAI_MODEL).strip(),
        input=[sys_msg, usr_msg],
        response_format={"type": "json_object"},
        max_output_tokens=MAX_TOKENS_PER_REQUEST,
    )
    if temperature is not None:
        args["temperature"] = float(temperature)

    rsp = _responses_create_robust(args)
    text = _coalesce_text_from_responses(rsp)
    if not text:
        dump = json.dumps(rsp, default=str)
        text = _extract_json_block(dump)
    return _json_loads_robust(text)


def _file_search_section_call(
    vector_store_id: str,                 # no usado aquí, se mantiene por compatibilidad
    user_prompt: str,
    model: str,
    temperature: Optional[float] = None,
    file_ids: Optional[List[str]] = None,
    section_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Llama Responses+FileSearch; si el texto no es JSON, fuerza una 2ª llamada de normalización.
    """
    if not file_ids:
        raise RuntimeError("No hay file_ids para adjuntar a la llamada. Reindexa los PDFs.")

    model = (model or OPENAI_MODEL).strip()
    sys_msg = {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PREFIX}]}
    usr_msg = {"role": "user",   "content": [{"type": "input_text", "text": user_prompt}]}

    args = dict(
        model=model,
        input=[sys_msg, usr_msg],
        tools=[{"type": "file_search"}],
        attachments=[{"file_id": fid, "tools": [{"type": "file_search"}]} for fid in file_ids],
        response_format={"type": "json_object"},
        max_output_tokens=MAX_TOKENS_PER_REQUEST,
    )
    if temperature is not None:
        args["temperature"] = float(temperature)

    rsp = _responses_create_robust(args)
    text = _coalesce_text_from_responses(rsp)

    if not text:
        dump = json.dumps(rsp, default=str)
        try:
            text = _extract_json_block(dump)
            return _json_loads_robust(text)
        except Exception:
            return _force_jsonify_from_text(section_key, dump, model, temperature)

    try:
        return _json_loads_robust(text)
    except Exception:
        return _force_jsonify_from_text(section_key, text, model, temperature)


# ---------------------------------------------------------------------
# Fallback local: Map-Reduce por sección
# ---------------------------------------------------------------------
def analyze_chunk_safe(current_result, ch_text, model, temperature):
    try:
        return analyze_text_chunk(current_result, ch_text, model=model, temperature=temperature)
    except TypeError:
        return analyze_text_chunk(current_result, ch_text, model=model)

def _merge_section_payload(section_key: str, acc: Dict[str, Any], part: Dict[str, Any]) -> Dict[str, Any]:
    if section_key == "objetivos_contexto":
        acc["resumen_servicios"] = max([acc.get("resumen_servicios"), part.get("resumen_servicios")], key=lambda x: len(x or ""), default=None)
        acc["alcance"] = max([acc.get("alcance"), part.get("alcance")], key=lambda x: len(x or ""), default=None)
        acc["objetivos"] = list({*(acc.get("objetivos") or []), * (part.get("objetivos") or [])})
        acc["referencias_paginas"] = sorted(set([*(acc.get("referencias_paginas") or []), * (part.get("referencias_paginas") or [])]))
        return acc
    if section_key == "servicios":
        acc["resumen_servicios"] = max([acc.get("resumen_servicios"), part.get("resumen_servicios")], key=lambda x: len(x or ""), default=None)
        acc["alcance"] = max([acc.get("alcance"), part.get("alcance")], key=lambda x: len(x or ""), default=None)
        acc["servicios_detalle"] = (acc.get("servicios_detalle") or []) + (part.get("servicios_detalle") or [])
        acc["referencias_paginas"] = sorted(set([*(acc.get("referencias_paginas") or []), * (part.get("referencias_paginas") or [])]))
        return acc
    if section_key == "importe":
        acc["importe_total"] = acc.get("importe_total") or part.get("importe_total")
        acc["moneda"] = acc.get("moneda") or part.get("moneda")
        acc["importes_detalle"] = (acc.get("importes_detalle") or []) + (part.get("importes_detalle") or [])
        acc["referencias_paginas"] = sorted(set([*(acc.get("referencias_paginas") or []), * (part.get("referencias_paginas") or [])]))
        return acc
    if section_key == "criterios_valoracion":
        acc["criterios_valoracion"] = (acc.get("criterios_valoracion") or []) + (part.get("criterios_valoracion") or [])
        acc["referencias_paginas"] = sorted(set([*(acc.get("referencias_paginas") or []), * (part.get("referencias_paginas") or [])]))
        return acc
    if section_key == "indice_tecnico":
        acc["indice_respuesta_tecnica"] = (acc.get("indice_respuesta_tecnica") or []) + (part.get("indice_respuesta_tecnica") or [])
        acc["indice_propuesto"] = (acc.get("indice_propuesto") or []) + (part.get("indice_propuesto") or [])
        acc["referencias_paginas"] = sorted(set([*(acc.get("referencias_paginas") or []), * (part.get("referencias_paginas") or [])]))
        return acc
    if section_key == "riesgos_exclusiones":
        acc["riesgos_y_dudas"] = max([acc.get("riesgos_y_dudas"), part.get("riesgos_y_dudas")], key=lambda x: len(x or ""), default=None)
        acc["exclusiones"] = list({*(acc.get("exclusiones") or []), * (part.get("exclusiones") or [])})
        acc["referencias_paginas"] = sorted(set([*(acc.get("referencias_paginas") or []), * (part.get("referencias_paginas") or [])]))
        return acc
    if section_key == "solvencia":
        sv = acc.get("solvencia") or {"tecnica": [], "economica": [], "administrativa": []}
        part_sv = part.get("solvencia") or {"tecnica": [], "economica": [], "administrativa": []}
        sv["tecnica"] = list({*sv.get("tecnica", []), *part_sv.get("tecnica", [])})
        sv["economica"] = list({*sv.get("economica", []), *part_sv.get("economica", [])})
        sv["administrativa"] = list({*sv.get("administrativa", []), *part_sv.get("administrativa", [])})
        acc["solvencia"] = sv
        acc["referencias_paginas"] = sorted(set([*(acc.get("referencias_paginas") or []), * (part.get("referencias_paginas") or [])]))
        return acc
    return acc

def _empty_payload_for_section(section_key: str) -> Dict[str, Any]:
    if section_key == "objetivos_contexto":
        return {"resumen_servicios": None, "objetivos": [], "alcance": None, "referencias_paginas": []}
    if section_key == "servicios":
        return {"resumen_servicios": None, "servicios_detalle": [], "alcance": None, "referencias_paginas": []}
    if section_key == "importe":
        return {"importe_total": None, "moneda": None, "importes_detalle": [], "referencias_paginas": []}
    if section_key == "criterios_valoracion":
        return {"criterios_valoracion": [], "referencias_paginas": []}
    if section_key == "indice_tecnico":
        return {"indice_respuesta_tecnica": [], "indice_propuesto": [], "referencias_paginas": []}
    if section_key == "riesgos_exclusiones":
        return {"riesgos_y_dudas": None, "exclusiones": [], "referencias_paginas": []}
    if section_key == "solvencia":
        return {"solvencia": {"tecnica": [], "economica": [], "administrativa": []}, "referencias_paginas": []}
    return {}

def _local_map_reduce_section(section_key: str, model: str, temperature: float, max_chars: int):
    docs = st.session_state.get("fs_local_docs", [])
    if not docs:
        raise RuntimeError("No hay texto local para fallback. Reindexa los PDFs.")
    all_chunks: List[str] = []
    for d in docs:
        all_chunks.extend(chunk_text(d["pages"], max_chars=max_chars))

    sys_msg = {"role": "system", "content": [{"type": "input_text", "text": (
        "Eres un analista experto en licitaciones públicas en España. "
        "Devuelve SIEMPRE JSON válido. Si el fragmento no contiene la información, devuelve campos vacíos/null."
    )}]}
    schema_hint = SECTION_SPECS[section_key]["user_prompt"]
    acc = _empty_payload_for_section(section_key)

    for ch in all_chunks:
        usr = {"role": "user", "content": [{
            "type": "input_text",
            "text": (
                f"Extrae SOLO la sección solicitada según este esquema (JSON):\n{schema_hint}\n\n"
                "Texto del pliego (fragmento):\n<<<\n" + ch[:15000] + "\n>>>\n"
                "Responde únicamente con JSON válido UTF-8."
            )
        }]}
        args = dict(
            model=(model or OPENAI_MODEL).strip(),
            input=[sys_msg, usr],
            response_format={"type": "json_object"},
            max_output_tokens=MAX_TOKENS_PER_REQUEST,
            temperature=float(temperature),
        )
        try:
            rsp = _responses_create_robust(args)
            text = _coalesce_text_from_responses(rsp)
            if not text:
                text = _extract_json_block(json.dumps(rsp, default=str))
            part = _json_loads_robust(text)
        except Exception:
            part = _empty_payload_for_section(section_key)
        acc = _merge_section_payload(section_key, acc, part)
    return acc

def _run_section_with_fallback(section_key: str, vs_id: str, file_ids: List[str], model: str, temperature: float, max_chars: int):
    try:
        data = _file_search_section_call(
            vector_store_id=vs_id,
            user_prompt=SECTION_SPECS[section_key]["user_prompt"],
            model=model,
            temperature=temperature,
            file_ids=file_ids,
            section_key=section_key,
        )
        # vacío → forzamos fallback
        if not data or all((v in (None, [], {}) for v in data.values())):
            raise RuntimeError("File Search devolvió JSON vacío.")
        return data, "file_search"
    except Exception:
        data = _local_map_reduce_section(section_key, model=model, temperature=temperature, max_chars=max_chars)
        return data, "local_map_reduce"


# ---------------------------------------------------------------------
# Sidebar (config) – limitado a 4o/4o-mini, temperatura fija
# ---------------------------------------------------------------------
def sidebar_config():
    with st.sidebar:
        st.header("Configuración")
        # Modelo (solo 4o y 4o-mini)
        if OPENAI_MODEL in AVAILABLE_MODELS:
            default_idx = AVAILABLE_MODELS.index(OPENAI_MODEL)
        else:
            default_idx = 0
        model = st.selectbox("Modelo OpenAI", options=AVAILABLE_MODELS, index=default_idx)
        st.caption("Temperatura fija: 0.2")
    # Devolvemos temperatura fija y sin modo (solo File Search)
    return model, FIXED_TEMPERATURE


# ---------------------------------------------------------------------
# Render de “Vista completa”
# ---------------------------------------------------------------------
def render_full_view(fs_sections: Dict[str, Any]):
    st.markdown("### Vista completa del análisis")
    # Métricas principales
    c1, c2, c3 = st.columns(3)
    # Objetivos/Resumen
    oc = fs_sections.get("objetivos_contexto", {})
    resumen = oc.get("resumen_servicios") or "—"
    objetivos = oc.get("objetivos") or []
    alcance = oc.get("alcance") or "—"

    # Importe
    im = fs_sections.get("importe", {})
    imp_total = im.get("importe_total")
    moneda = im.get("moneda") or "€"
    imp_str = f"{imp_total:,.2f} {moneda}" if isinstance(imp_total, (int, float)) else "—"

    # Solvencia
    sv = fs_sections.get("solvencia", {}).get("solvencia", {})
    solv_tec = len(sv.get("tecnica", []))
    solv_eco = len(sv.get("economica", []))
    solv_adm = len(sv.get("administrativa", []))

    c1.metric("Importe total", imp_str)
    c2.metric("Objetivos detectados", len(objetivos))
    c3.metric("Requisitos de solvencia (tot.)", solv_tec + solv_eco + solv_adm)

    st.divider()

    # Sección Objetivos/Contexto
    with st.expander("🎯 Objetivos y contexto", expanded=True):
        st.markdown(f"**Resumen de servicios**: {resumen}")
        if objetivos:
            st.markdown("**Objetivos**:")
            st.write("\n".join([f"- {o}" for o in objetivos]))
        st.markdown(f"**Alcance**: {alcance}")

    # Servicios solicitados
    svs = fs_sections.get("servicios", {})
    with st.expander("🧩 Servicios solicitados (detalle)", expanded=False):
        st.markdown(f"**Resumen**: {svs.get('resumen_servicios') or '—'}")
        detalle = svs.get("servicios_detalle") or []
        if detalle:
            # tabla simple
            import pandas as pd
            df = pd.DataFrame([{"Servicio": d.get("nombre"), "Descripción": d.get("descripcion")} for d in detalle])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No se encontraron servicios detallados explícitos en el texto analizado.")

    # Importe y desglose
    with st.expander("💶 Importe de licitación", expanded=False):
        st.markdown(f"**Importe total**: {imp_str}")
        det = im.get("importes_detalle") or []
        if det:
            import pandas as pd
            df = pd.DataFrame(det)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Sin desglose adicional de importes.")

    # Criterios de valoración
    cv = fs_sections.get("criterios_valoracion", {}).get("criterios_valoracion", [])
    with st.expander("📊 Criterios de valoración", expanded=False):
        if cv:
            rows = []
            for c in cv:
                rows.append({
                    "Criterio": c.get("nombre"),
                    "Peso máx": c.get("peso_max"),
                    "Tipo": c.get("tipo"),
                    "Subcriterios": "; ".join([sc.get("nombre") for sc in (c.get("subcriterios") or []) if sc.get("nombre")]),
                })
            import pandas as pd
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No se encontraron criterios explícitos.")

    # Índice técnico: solicitado y propuesto
    it = fs_sections.get("indice_tecnico", {})
    with st.expander("🗂️ Índice de la respuesta técnica", expanded=False):
        col1, col2 = st.columns(2)
        req = it.get("indice_respuesta_tecnica") or []
        prop = it.get("indice_propuesto") or []
        with col1:
            st.markdown("**Índice solicitado**")
            if req:
                st.write("\n".join([f"- {s.get('titulo')}" for s in req]))
            else:
                st.info("Sin índice solicitado detectado.")
        with col2:
            st.markdown("**Índice propuesto**")
            if prop:
                st.write("\n".join([f"- {s.get('titulo')}" for s in prop]))
            else:
                st.info("Sin índice propuesto.")

    # Riesgos / Exclusiones
    rx = fs_sections.get("riesgos_exclusiones", {})
    with st.expander("⚠️ Riesgos y exclusiones", expanded=False):
        ry = rx.get("riesgos_y_dudas")
        ex = rx.get("exclusiones") or []
        st.markdown(f"**Riesgos y dudas**: {ry or '—'}")
        if ex:
            st.markdown("**Exclusiones**:")
            st.write("\n".join([f"- {e}" for e in ex]))

    # Solvencia
    with st.expander("📜 Solvencia", expanded=False):
        col1, col2, col3 = st.columns(3)
        tec = sv.get("tecnica", [])
        eco = sv.get("economica", [])
        adm = sv.get("administrativa", [])
        with col1:
            st.markdown("**Técnica**")
            st.write("\n".join([f"- {x}" for x in tec]) or "—")
        with col2:
            st.markdown("**Económica**")
            st.write("\n".join([f"- {x}" for x in eco]) or "—")
        with col3:
            st.markdown("**Administrativa**")
            st.write("\n".join([f"- {x}" for x in adm]) or "—")


# ---------------------------------------------------------------------
# App principal
# ---------------------------------------------------------------------
def main():
    # 1) Login
    login_gate()

    # 2) Config: solo modelo y temperatura fija
    model, temperature = sidebar_config()

    # 3) Cabecera
    render_header(APP_TITLE)

    # 4) Subida de archivos
    files = st.file_uploader(
        "Sube uno o varios PDFs del pliego (pliego general, anexos, etc.)",
        type=["pdf"], accept_multiple_files=True
    )
    if not files:
        st.info("Sube al menos un PDF para comenzar el análisis.")
        st.stop()

    # 5) Indexación en OpenAI + captura de texto local (para fallback/map-reduce)
    if "fs_vs_id" not in st.session_state or st.button("Reindexar PDFs en OpenAI"):
        st.session_state.pop("fs_sections", None)
        st.session_state.pop(SECOND_TAB_KEY, None)

        uploaded = [{"name": f.name, "bytes": f.read()} for f in files]

        class _MemFile:
            def __init__(self, name, data): self.name=name; self._data=data
            def read(self): return self._data

        mem_files = [_MemFile(u["name"], u["bytes"]) for u in uploaded]

        with st.spinner("Indexando PDFs en OpenAI (Vector Store)..."):
            vs_id, file_ids = create_vector_store_from_streamlit_files(mem_files, name="RFP Vector Store")

        # Texto local y diagnóstico
        local_docs = []
        char_stats = []
        for u in uploaded:
            pages, _ = extract_pdf_text(io.BytesIO(u["bytes"]))
            pages = [clean_text(p) for p in pages]
            local_docs.append({"name": u["name"], "pages": pages})
            total_chars = sum(len(p or "") for p in pages)
            char_stats.append((u["name"], len(pages), total_chars))

        st.session_state["fs_vs_id"] = vs_id
        st.session_state["fs_file_ids"] = file_ids
        st.session_state["fs_local_docs"] = local_docs
        st.session_state["char_stats"] = char_stats

        st.success("PDF(s) indexados en OpenAI y texto local preparado.")

    vs_id = st.session_state.get("fs_vs_id")
    file_ids = st.session_state.get("fs_file_ids", [])
    local_docs = st.session_state.get("fs_local_docs", [])
    if not vs_id:
        st.stop()

    # Diagnóstico de texto extraído
    st.info("Texto extraído por PDF (diagnóstico):")
    for name, npages, nchar in st.session_state.get("char_stats", []):
        st.write(f"- **{name}**: {npages} páginas, {nchar} caracteres")
        if nchar < 1000:
            st.warning(f"{name}: muy poco texto extraído (posible PDF escaneado sin OCR).")

    st.info(f"Vector Store listo: `{vs_id}` – {len(file_ids)} archivo(s) adjuntables")

    # --- Estado de ejecución para evitar duplicados visuales ---
    st.session_state.setdefault("busy", False)
    st.session_state.setdefault("job", None)        # sección a ejecutar
    st.session_state.setdefault("job_all", False)   # ejecutar todas
    
    def _start_job(section: str | None = None, do_all: bool = False):
        st.session_state["job"] = section
        st.session_state["job_all"] = do_all
        st.session_state["busy"] = True
    
    # 6) Pestañas de trabajo
    tab1, tab2 = st.tabs(["Análisis por secciones", "Vista completa"])

    with tab1:
        st.subheader("Análisis por secciones")
    
        # Contenedor único para los controles (no se duplican)
        controls = st.container()
        with controls:
            c1, c2, c3 = st.columns(3)
            dis = st.session_state["busy"]
    
            with c1:
                st.button("Objetivos y contexto",   key="btn_obj",  use_container_width=True,
                          disabled=dis, on_click=_start_job, kwargs={"section": "objetivos_contexto"})
                st.button("Servicios solicitados",  key="btn_srv",  use_container_width=True,
                          disabled=dis, on_click=_start_job, kwargs={"section": "servicios"})
                st.button("Importe de licitación",  key="btn_imp",  use_container_width=True,
                          disabled=dis, on_click=_start_job, kwargs={"section": "importe"})
    
            with c2:
                st.button("Criterios de valoración", key="btn_crit", use_container_width=True,
                          disabled=dis, on_click=_start_job, kwargs={"section": "criterios_valoracion"})
                st.button("Índice de la respuesta técnica", key="btn_idx", use_container_width=True,
                          disabled=dis, on_click=_start_job, kwargs={"section": "indice_tecnico"})
                st.button("Riesgos y exclusiones",   key="btn_risk", use_container_width=True,
                          disabled=dis, on_click=_start_job, kwargs={"section": "riesgos_exclusiones"})
    
            with c3:
                st.button("Criterios de solvencia",  key="btn_solv", use_container_width=True,
                          disabled=dis, on_click=_start_job, kwargs={"section": "solvencia"})
                st.write("")
                st.button("🔎 Análisis Completo", type="primary", key="btn_all", use_container_width=True,
                          disabled=dis, on_click=_start_job, kwargs={"do_all": True})
    
        # Resultado/estado de ejecución: un único bloque controlado
        if st.session_state["busy"]:
            with st.status("Procesando análisis…", expanded=True) as status:
                try:
                    if st.session_state["job_all"]:
                        order = list(SECTION_SPECS.keys())
                        for k in order:
                            data, mode_used = _run_section_with_fallback(
                                section_key=k,
                                vs_id=st.session_state["fs_vs_id"],
                                file_ids=st.session_state["fs_file_ids"],
                                model=model,
                                temperature=temperature,
                                max_chars=CHUNK_MAX_CHARS_DEFAULT,
                            )
                            st.session_state.setdefault("fs_sections", {})
                            st.session_state["fs_sections"][k] = data
                            status.write(f"✓ {SECTION_SPECS[k]['titulo']} ({'File Search' if mode_used=='file_search' else 'Local'})")
                        st.session_state[SECOND_TAB_KEY] = True
                        status.update(label="Análisis completo finalizado", state="complete")
                    else:
                        k = st.session_state["job"]
                        spec = SECTION_SPECS[k]
                        status.update(label=f"Analizando sección: {spec['titulo']}…")
                        data, mode_used = _run_section_with_fallback(
                            section_key=k,
                            vs_id=st.session_state["fs_vs_id"],
                            file_ids=st.session_state["fs_file_ids"],
                            model=model,
                            temperature=temperature,
                            max_chars=CHUNK_MAX_CHARS_DEFAULT,
                        )
                        st.session_state.setdefault("fs_sections", {})
                        st.session_state["fs_sections"][k] = data
                        status.update(label=f"Sección '{spec['titulo']}' completada", state="complete")
                finally:
                    # Limpieza de estado y re-render limpio
                    st.session_state["busy"] = False
                    st.session_state["job"] = None
                    st.session_state["job_all"] = False
                    st.rerun()
    
        # Visualización de resultados por sección (solo si hay datos y no estamos ocupados)
        st.subheader("Resultados por sección")
        if not st.session_state["busy"]:
            for key, spec in SECTION_SPECS.items():
                if "fs_sections" in st.session_state and key in st.session_state["fs_sections"]:
                    with st.expander(spec["titulo"], expanded=False):
                        st.json(st.session_state["fs_sections"][key])
                        st.download_button(
                            f"Descargar JSON – {spec['titulo']}",
                            json.dumps(st.session_state["fs_sections"][key], indent=2, ensure_ascii=False),
                            file_name=f"{key}.json",
                            mime="application/json",
                        )


    with tab2:
        fs_sections = st.session_state.get("fs_sections", {})
        if not fs_sections:
            st.info("Aún no hay resultados. Pulsa **Análisis Completo** o ejecuta alguna sección en la pestaña anterior.")
        else:
            render_full_view(fs_sections)


if __name__ == "__main__":
    main()
