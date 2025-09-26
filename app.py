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
    OPENAI_TEMPERATURE,
    MODELS_CATALOG,
    ADMIN_USER,
    ADMIN_PASSWORD,
    MAX_TOKENS_PER_REQUEST,
)

# Servicios locales ya existentes
from services.pdf_loader import extract_pdf_text
from services.schema import OfertaAnalizada
from services.openai_client import analyze_text_chunk, merge_offers
from utils.text import clean_text, chunk_text
from components.ui import render_header, render_result

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
    # 1) Propiedad estándar
    txt = getattr(rsp, "output_text", None)
    if txt:
        return txt
    # 2) Recorrer bloques (output / content / text)
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
    # 3) A veces el SDK expone message único
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

            # 1) temperature
            if _is_temperature_error(e) or ("unexpected keyword" in s and "temperature" in s):
                a.pop("temperature", None)
                continue

            # 2) response_format
            if _is_unsupported_param(e, "response_format") or ("unexpected keyword" in s and "response_format" in s):
                a.pop("response_format", None)
                continue

            # 3) max_output_tokens -> max_completion_tokens
            if _is_unsupported_param(e, "max_output_tokens") or ("unexpected keyword" in s and "max_output_tokens" in s):
                val = a.pop("max_output_tokens", None)
                if val is not None:
                    a["max_completion_tokens"] = val
                continue

            # 4) attachments: mover a extra_body si el SDK no lo acepta como kw
            if _is_unsupported_param(e, "attachments") or ("unexpected keyword" in s and "attachments" in s):
                att = a.pop("attachments", None)
                if att is not None:
                    extra["attachments"] = att
                continue

            # 5) tools: mover a extra_body si el SDK no lo acepta como kw
            if _is_unsupported_param(e, "tools") or ("unexpected keyword" in s and "tools" in s):
                tools = a.pop("tools", None)
                if tools is not None:
                    extra["tools"] = tools
                continue

            # errores de servidor por parámetros desconocidos: propaga para que el caller active fallback
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
# Secciones (prompts específicos para File Search)
# ---------------------------------------------------------------------
SECTION_SPECS: Dict[str, Dict[str, str]] = {
    "objetivos_contexto": {
        "titulo": "Objetivos y contexto",
        "user_prompt": (
            "Sobre los PDFs aportados, extrae OBJETIVOS y CONTEXTO del pliego. "
            "Devuelve SOLO JSON con campos:\n"
            "{\n"
            '  "resumen_servicios": str,\n'
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
            '  "resumen_servicios": str,\n'
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
    """
    Si el modelo devolvió texto no JSON, hacemos una 2ª llamada sin herramientas para
    convertir a JSON válido según el esquema descrito en el prompt de la sección.
    """
    schema_hint = ""
    if section_key and section_key in SECTION_SPECS:
        # Reutilizamos el bloque de esquema embebido en el user_prompt
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
    # Modelos gpt-5: suelen rechazar response_format/temperature ≠ 1
    if args["model"].lower().startswith("gpt-5"):
        args.pop("temperature", None)
        args.pop("response_format", None)

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
    Lanza una llamada a Responses+FileSearch para una sección concreta y devuelve dict.
    IMPORTANTE: sin 'tool_resources' (algunos endpoints lo rechazan).
    Usamos 'attachments' + 'tools' (y _responses_create_robust lo moverá a extra_body si hace falta).
    Si el texto devuelto no es JSON, se fuerza a JSON con una 2ª llamada sin herramientas.
    """
    if not file_ids:
        raise RuntimeError("No hay file_ids para adjuntar a la llamada. Reindexa los PDFs.")

    model = (model or OPENAI_MODEL).strip()
    is_gpt5 = model.lower().startswith("gpt-5")

    sys_msg = {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PREFIX}]}
    usr_msg = {"role": "user",   "content": [{"type": "input_text", "text": user_prompt}]}

    args = dict(
        model=model,
        input=[sys_msg, usr_msg],
        tools=[{"type": "file_search"}],  # habilita la herramienta
        attachments=[{"file_id": fid, "tools": [{"type": "file_search"}]} for fid in file_ids],
        response_format={"type": "json_object"},
        max_output_tokens=MAX_TOKENS_PER_REQUEST,
    )
    if temperature is not None:
        args["temperature"] = float(temperature)
    if is_gpt5:
        # Muchos despliegues gpt-5* rechazan temp != 1 y response_format
        args.pop("temperature", None)
        args.pop("response_format", None)

    rsp = _responses_create_robust(args)
    text = _coalesce_text_from_responses(rsp)

    # Si no hay texto, intentamos rascar JSON del dump crudo
    if not text:
        dump = json.dumps(rsp, default=str)
        try:
            text = _extract_json_block(dump)
            return _json_loads_robust(text)
        except Exception:
            # Forzamos conversión a JSON con 2ª llamada
            return _force_jsonify_from_text(section_key, dump, model, temperature)

    # Intentamos parsear; si falla, forzamos conversión
    try:
        return _json_loads_robust(text)
    except Exception:
        return _force_jsonify_from_text(section_key, text, model, temperature)

def _merge_section_payload(section_key: str, acc: Dict[str, Any], part: Dict[str, Any]) -> Dict[str, Any]:
    """Fusión simple y determinista para cada sección."""
    if section_key == "objetivos_contexto":
        acc["resumen_servicios"] = max([acc.get("resumen_servicios"), part.get("resumen_servicios")], key=lambda x: len(x or ""), default=None)
        acc["alcance"] = max([acc.get("alcance"), part.get("alcance")], key=lambda x: len(x or ""), default=None)
        acc["objetivos"] = list({*(acc.get("objetivos") or []), * (part.get("objetivos") or [])})
        acc["referencias_paginas"] = list({*(acc.get("referencias_paginas") or []), * (part.get("referencias_paginas") or [])})
        return acc

    if section_key == "servicios":
        acc["resumen_servicios"] = max([acc.get("resumen_servicios"), part.get("resumen_servicios")], key=lambda x: len(x or ""), default=None)
        acc["alcance"] = max([acc.get("alcance"), part.get("alcance")], key=lambda x: len(x or ""), default=None)
        acc["servicios_detalle"] = (acc.get("servicios_detalle") or []) + (part.get("servicios_detalle") or [])
        acc["referencias_paginas"] = list({*(acc.get("referencias_paginas") or []), * (part.get("referencias_paginas") or [])})
        return acc

    if section_key == "importe":
        acc["importe_total"] = acc.get("importe_total") or part.get("importe_total")
        acc["moneda"] = acc.get("moneda") or part.get("moneda")
        acc["importes_detalle"] = (acc.get("importes_detalle") or []) + (part.get("importes_detalle") or [])
        acc["referencias_paginas"] = list({*(acc.get("referencias_paginas") or []), * (part.get("referencias_paginas") or [])})
        return acc

    if section_key == "criterios_valoracion":
        acc["criterios_valoracion"] = (acc.get("criterios_valoracion") or []) + (part.get("criterios_valoracion") or [])
        acc["referencias_paginas"] = list({*(acc.get("referencias_paginas") or []), * (part.get("referencias_paginas") or [])})
        return acc

    if section_key == "indice_tecnico":
        acc["indice_respuesta_tecnica"] = (acc.get("indice_respuesta_tecnica") or []) + (part.get("indice_respuesta_tecnica") or [])
        acc["indice_propuesto"] = (acc.get("indice_propuesto") or []) + (part.get("indice_propuesto") or [])
        acc["referencias_paginas"] = list({*(acc.get("referencias_paginas") or []), * (part.get("referencias_paginas") or [])})
        return acc

    if section_key == "riesgos_exclusiones":
        # mantiene texto más largo y concatena exclusiones
        acc["riesgos_y_dudas"] = max([acc.get("riesgos_y_dudas"), part.get("riesgos_y_dudas")], key=lambda x: len(x or ""), default=None)
        acc["exclusiones"] = list({*(acc.get("exclusiones") or []), * (part.get("exclusiones") or [])})
        acc["referencias_paginas"] = list({*(acc.get("referencias_paginas") or []), * (part.get("referencias_paginas") or [])})
        return acc

    if section_key == "solvencia":
        sv = acc.get("solvencia") or {"tecnica": [], "economica": [], "administrativa": []}
        part_sv = part.get("solvencia") or {"tecnica": [], "economica": [], "administrativa": []}
        sv["tecnica"] = list({*sv.get("tecnica", []), *part_sv.get("tecnica", [])})
        sv["economica"] = list({*sv.get("economica", []), *part_sv.get("economica", [])})
        sv["administrativa"] = list({*sv.get("administrativa", []), *part_sv.get("administrativa", [])})
        acc["solvencia"] = sv
        acc["referencias_paginas"] = list({*(acc.get("referencias_paginas") or []), * (part.get("referencias_paginas") or [])})
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
    """
    Ejecuta prompts por sección sobre chunks locales y fusiona resultados.
    """
    docs = st.session_state.get("fs_local_docs", [])
    if not docs:
        raise RuntimeError("No hay texto local para fallback. Reindexa los PDFs.")

    # Construye chunks (unidos entre documentos)
    all_chunks: List[str] = []
    for d in docs:
        all_chunks.extend(chunk_text(d["pages"], max_chars=max_chars))

    # System + esquema a usar (del SECTION_SPECS)
    sys_msg = {"role": "system", "content": [{"type": "input_text", "text": (
        "Eres un analista experto en licitaciones públicas en España. "
        "Devuelve SIEMPRE JSON válido. Si el fragmento no contiene la información, devuelve campos vacíos/null."
    )}]}
    schema_hint = SECTION_SPECS[section_key]["user_prompt"]
    acc = _empty_payload_for_section(section_key)

    # Itera chunks → llama a Responses → merge
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
        )
        if temperature is not None:
            args["temperature"] = float(temperature)
        if args["model"].lower().startswith("gpt-5"):
            args.pop("temperature", None)
            args.pop("response_format", None)

        try:
            rsp = _responses_create_robust(args)
            text = _coalesce_text_from_responses(rsp)
            if not text:
                # intentamos rascar JSON del dump
                text = _extract_json_block(json.dumps(rsp, default=str))
            part = _json_loads_robust(text)
        except Exception:
            # chunk sin señal → parte vacía
            part = _empty_payload_for_section(section_key)

        acc = _merge_section_payload(section_key, acc, part)

    return acc


# ---------------------------------------------------------------------
# Fallback local por secciones (si el endpoint no admite File Search)
# ---------------------------------------------------------------------
def analyze_chunk_safe(current_result, ch_text, model, temperature):
    """Compat con firmas antiguas de analyze_text_chunk."""
    try:
        return analyze_text_chunk(current_result, ch_text, model=model, temperature=temperature)
    except TypeError:
        return analyze_text_chunk(current_result, ch_text, model=model)

def _fallback_local_section(section_key: str, model: str, temperature: float, max_chars: int):
    """
    Fallback robusto: analiza localmente por chunks con analyze_text_chunk (pipeline existente)
    y devuelve un dict con la forma esperada por la sección.
    """
    docs = st.session_state.get("fs_local_docs", [])
    if not docs:
        raise RuntimeError("No hay texto local para fallback. Reindexa los PDFs.")

    # Construimos chunks según el slider actual
    all_chunks: List[str] = []
    for d in docs:
        chunks = chunk_text(d["pages"], max_chars=max_chars)
        all_chunks.extend(chunks)

    # Pasada de análisis global con tu pipeline existente
    result = None
    for ch in all_chunks:
        result = analyze_chunk_safe(result, ch, model=model, temperature=temperature)

    # Map al JSON esperado por cada sección
    if section_key == "objetivos_contexto":
        return {
            "resumen_servicios": result.resumen_servicios,
            "objetivos": result.objetivos,
            "alcance": result.alcance,
            "referencias_paginas": result.referencias_paginas,
        }
    elif section_key == "servicios":
        return {
            "resumen_servicios": result.resumen_servicios,
            "servicios_detalle": [],
            "alcance": result.alcance,
            "referencias_paginas": result.referencias_paginas,
        }
    elif section_key == "importe":
        return {
            "importe_total": result.importe_total,
            "moneda": result.moneda,
            "importes_detalle": [d.model_dump() for d in result.importes_detalle],
            "referencias_paginas": result.referencias_paginas,
        }
    elif section_key == "criterios_valoracion":
        return {
            "criterios_valoracion": [c.model_dump() for c in result.criterios_valoracion],
            "referencias_paginas": result.referencias_paginas,
        }
    elif section_key == "indice_tecnico":
        return {
            "indice_respuesta_tecnica": [s.model_dump() for s in result.indice_respuesta_tecnica],
            "indice_propuesto": [s.model_dump() for s in result.indice_propuesto],
            "referencias_paginas": result.referencias_paginas,
        }
    elif section_key == "riesgos_exclusiones":
        return {
            "riesgos_y_dudas": result.riesgos_y_dudas,
            "exclusiones": [],
            "referencias_paginas": result.referencias_paginas,
        }
    elif section_key == "solvencia":
        return {
            "solvencia": {"tecnica": [], "economica": [], "administrativa": []},
            "referencias_paginas": result.referencias_paginas,
        }
    else:
        return {"mensaje": "Sección no reconocida", "section_key": section_key}

def _run_section_with_fallback(section_key: str, vs_id: str, file_ids: List[str], model: str, temperature: float, max_chars: int):
    # 1º intento: File Search (si el servidor lo soporta y devuelve algo)
    try:
        data = _file_search_section_call(
            vector_store_id=vs_id,
            user_prompt=SECTION_SPECS[section_key]["user_prompt"],
            model=model,
            temperature=temperature,
            file_ids=file_ids,
            section_key=section_key,
        )
        # ¿vino vacío?
        if not data or all((v in (None, [], {}) for v in data.values())):
            raise RuntimeError("File Search devolvió JSON vacío.")
        return data, "file_search"
    except Exception as e:
        msg = str(e)
        # Cualquier rechazo/ausencia → Map-Reduce local
        data = _local_map_reduce_section(section_key, model=model, temperature=temperature, max_chars=max_chars)
        return data, "local_map_reduce"



# ---------------------------------------------------------------------
# Sidebar (config)
# ---------------------------------------------------------------------
def sidebar_config():
    """Dibuja la configuración en sidebar y devuelve (model, temperature, max_chars, mode)."""
    with st.sidebar:
        st.header("Configuración")

        # Selector de modelo (catálogo + opción personalizada)
        options = MODELS_CATALOG + ["Otro…"]
        try:
            default_index = options.index(OPENAI_MODEL) if OPENAI_MODEL in options else 0
        except Exception:
            default_index = 0

        model_choice = st.selectbox(
            "Modelo OpenAI",
            options=options,
            index=default_index,
            help="Selecciona un modelo del catálogo o usa 'Otro…' para escribir uno manualmente."
        )
        if model_choice == "Otro…":
            model = st.text_input("Modelo personalizado", value=OPENAI_MODEL)
        else:
            model = model_choice

        # Temperatura
        temperature = st.slider(
            "Temperatura",
            min_value=0.0, max_value=2.0,
            value=float(OPENAI_TEMPERATURE), step=0.1
        )

        # Modo
        mode = st.radio(
            "Modo de análisis",
            ["Chunking local", "PDF completo (File Search)"],
            index=0,
            help="File Search sube los PDF a OpenAI y los reutiliza en varias consultas; si no está disponible, la app hará fallback local."
        )

        # Chunking local: tamaño de chunk
        max_chars = st.slider(
            "Tamaño aprox. de chunk (caracteres)",
            6_000, 40_000, 12_000, step=1_000
        )
        st.caption(
            "A mayor chunk, menos llamadas pero más tokens. "
            "Nota: algunos modelos (p.ej., gpt-5*) ignoran temperaturas ≠ 1.0; el cliente se autoajusta."
        )

    return model, temperature, max_chars, mode


# ---------------------------------------------------------------------
# App principal
# ---------------------------------------------------------------------
def main():
    # 1) Login
    login_gate()

    # 2) Config
    model, temperature, max_chars, mode = sidebar_config()

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

    # =========================================================
    # A) Modo Chunking local (pipeline existente)
    # =========================================================
    if mode == "Chunking local":
        docs = []
        total_chunks = 0
        with st.spinner("Extrayendo texto de los PDFs..."):
            for file in files:
                pdf_bytes = file.read()
                buf = io.BytesIO(pdf_bytes)
                pages, _full = extract_pdf_text(buf)
                pages = [clean_text(p) for p in pages]
                chunks = chunk_text(pages, max_chars=max_chars)
                docs.append({
                    "name": file.name,
                    "pages": pages,
                    "chunks": chunks,
                    "num_pages": len(pages)
                })
                total_chunks += len(chunks)

        st.success(
            f"Se han leído {len(docs)} fichero(s). "
            f"Total páginas: {sum(d['num_pages'] for d in docs)}; "
            f"total chunks: {total_chunks}."
        )

        if st.button("Analizar con OpenAI (chunking local)", type="primary"):
            aggregate_result = None
            per_file_results = {}
            prog = st.progress(0.0)
            status = st.empty()
            processed = 0

            for d in docs:
                n_chunks = len(d["chunks"])
                status.info(f"Analizando: {d['name']} ({n_chunks} chunks)...")
                result = None
                for ch_idx, ch in enumerate(d["chunks"], start=1):
                    try:
                        result = analyze_chunk_safe(result, ch, model=model, temperature=temperature)
                    except Exception as e:
                        st.error(f"Error analizando **{d['name']}**, chunk {ch_idx}/{n_chunks}: {e}")
                        raise
                    processed += 1
                    prog.progress(processed / max(total_chunks, 1))
                per_file_results[d["name"]] = result
                aggregate_result = result if aggregate_result is None else merge_offers(aggregate_result, result)

            status.success("Análisis completado.")
            st.session_state["per_file_results"] = per_file_results
            st.session_state["aggregate_result"] = aggregate_result

    # =========================================================
    # B) Modo PDF completo (File Search) + Fallback local
    # =========================================================
    else:
        # (1) Preparar/crear Vector Store (solo 1 vez por subida) y capturar texto local para fallback
        if "fs_vs_id" not in st.session_state or st.button("Reindexar PDFs en OpenAI"):
            st.session_state.pop("fs_sections", None)

            # Leemos los PDFs a memoria UNA vez
            uploaded = [{"name": f.name, "bytes": f.read()} for f in files]

            # Wrapper para usar create_vector_store_from_streamlit_files
            class _MemFile:
                def __init__(self, name, data):
                    self.name = name
                    self._data = data
                def read(self):
                    return self._data

            mem_files = [_MemFile(u["name"], u["bytes"]) for u in uploaded]

            with st.spinner("Indexando PDFs en OpenAI (Vector Store)..."):
                vs_id, file_ids = create_vector_store_from_streamlit_files(mem_files, name="RFP Vector Store")

            # Guardamos ids y también el TEXTO local (para fallback)
            local_docs = []
            for u in uploaded:
                pages, _ = extract_pdf_text(io.BytesIO(u["bytes"]))
                pages = [clean_text(p) for p in pages]
                local_docs.append({
                    "name": u["name"],
                    "pages": pages,
                })
            char_stats = []
            for d in local_docs:
                total_chars = sum(len(p or "") for p in d["pages"])
                char_stats.append((d["name"], len(d["pages"]), total_chars))
            
            st.info("Texto extraído por PDF (diagnóstico):")
            for name, npages, nchar in char_stats:
                st.write(f"- **{name}**: {npages} páginas, {nchar} caracteres extraídos")
                if nchar < 1000:
                    st.warning(f"{name}: muy poco texto extraído (posible PDF escaneado sin OCR).")
        
            st.session_state["fs_vs_id"] = vs_id
            st.session_state["fs_file_ids"] = file_ids
            st.session_state["fs_local_docs"] = local_docs
            st.success("PDF(s) indexados en OpenAI y texto local preparado.")

        vs_id = st.session_state.get("fs_vs_id")
        file_ids = st.session_state.get("fs_file_ids", [])
        local_docs = st.session_state.get("fs_local_docs", [])
        if not vs_id:
            st.stop()

        st.info(f"Vector Store listo: `{vs_id}` – {len(file_ids)} archivo(s) adjuntables")

        # (2) Botonera de secciones
        st.subheader("Análisis por secciones")
        c1, c2, c3 = st.columns(3)
        with c1:
            b_obj = st.button("Objetivos y contexto")
            b_serv = st.button("Servicios solicitados")
            b_imp = st.button("Importe de licitación")
        with c2:
            b_crit = st.button("Criterios de valoración")
            b_idx = st.button("Índice de la respuesta técnica")
            b_risk = st.button("Riesgos y exclusiones")
        with c3:
            b_solv = st.button("Criterios de solvencia")
            st.write("")

        if "fs_sections" not in st.session_state:
            st.session_state["fs_sections"] = {}

        # (3) Ejecutar sección solicitada (con fallback automático)
        def run_section(section_key: str):
            spec = SECTION_SPECS[section_key]
            with st.spinner(f"Analizando sección: {spec['titulo']}..."):
                data, mode_used = _run_section_with_fallback(
                    section_key=section_key,
                    vs_id=vs_id,
                    file_ids=file_ids,
                    model=model,
                    temperature=temperature,
                    max_chars=max_chars,
                )
            st.session_state["fs_sections"][section_key] = data
            if mode_used == "local_fallback":
                st.warning(f"Sección '{spec['titulo']}' analizada con **fallback local** (endpoint sin File Search en Responses o sin JSON).")
            else:
                st.info(f"Sección '{spec['titulo']}' analizada con **File Search**.")

        if b_obj:  run_section("objetivos_contexto")
        if b_serv: run_section("servicios")
        if b_imp:  run_section("importe")
        if b_crit: run_section("criterios_valoracion")
        if b_idx:  run_section("indice_tecnico")
        if b_risk: run_section("riesgos_exclusiones")
        if b_solv: run_section("solvencia")

        # (4) Mostrar resultados por sección (si existen)
        st.subheader("Resultados")
        for key, spec in SECTION_SPECS.items():
            if key in st.session_state["fs_sections"]:
                with st.expander(spec["titulo"], expanded=False):
                    st.json(st.session_state["fs_sections"][key])
                    st.download_button(
                        f"Descargar JSON – {spec['titulo']}",
                        json.dumps(st.session_state["fs_sections"][key], indent=2, ensure_ascii=False),
                        file_name=f"{key}.json",
                        mime="application/json",
                    )

        # (5) (Opcional) Construir resultado agregado con el esquema principal
        if st.session_state["fs_sections"]:
            if st.button("Construir resultado agregado (map a esquema principal)"):
                payload: Dict[str, Any] = {
                    "resumen_servicios": "",
                    "objetivos": [],
                    "alcance": None,
                    "importe_total": None,
                    "moneda": None,
                    "importes_detalle": [],
                    "criterios_valoracion": [],
                    "indice_respuesta_tecnica": [],
                    "indice_propuesto": [],
                    "riesgos_y_dudas": None,
                    "referencias_paginas": [],
                }

                sec = st.session_state["fs_sections"]
                # objetivos/contexto
                oc = sec.get("objetivos_contexto", {})
                payload["resumen_servicios"] = oc.get("resumen_servicios") or payload["resumen_servicios"]
                payload["objetivos"] = oc.get("objetivos") or payload["objetivos"]
                payload["alcance"] = oc.get("alcance") or payload["alcance"]
                payload["referencias_paginas"] += oc.get("referencias_paginas", [])

                # servicios
                sv = sec.get("servicios", {})
                if isinstance(sv.get("resumen_servicios"), str) and len(sv["resumen_servicios"]) > len(payload["resumen_servicios"]):
                    payload["resumen_servicios"] = sv["resumen_servicios"]
                payload["alcance"] = payload["alcance"] or sv.get("alcance")
                payload["referencias_paginas"] += sv.get("referencias_paginas", [])

                # importe
                im = sec.get("importe", {})
                payload["importe_total"] = im.get("importe_total") or payload["importe_total"]
                payload["moneda"] = im.get("moneda") or payload["moneda"]
                payload["importes_detalle"] += im.get("importes_detalle", [])
                payload["referencias_paginas"] += im.get("referencias_paginas", [])

                # criterios
                cv = sec.get("criterios_valoracion", {})
                payload["criterios_valoracion"] += cv.get("criterios_valoracion", [])
                payload["referencias_paginas"] += cv.get("referencias_paginas", [])

                # índice
                ix = sec.get("indice_tecnico", {})
                payload["indice_respuesta_tecnica"] += ix.get("indice_respuesta_tecnica", [])
                payload["indice_propuesto"] += ix.get("indice_propuesto", [])
                payload["referencias_paginas"] += ix.get("referencias_paginas", [])

                # riesgos/exclusiones
                rx = sec.get("riesgos_exclusiones", {})
                ry = rx.get("riesgos_y_dudas")
                ex = rx.get("exclusiones", [])
                if ry or ex:
                    txt = (ry or "") + ("\nExclusiones: " + "; ".join(ex) if ex else "")
                    payload["riesgos_y_dudas"] = (payload["riesgos_y_dudas"] + "\n" + txt).strip() if payload["riesgos_y_dudas"] else txt
                payload["referencias_paginas"] += rx.get("referencias_paginas", [])

                # ordena y dedup páginas
                try:
                    payload["referencias_paginas"] = sorted(set(int(p) for p in payload["referencias_paginas"]))
                except Exception:
                    payload["referencias_paginas"] = []

                try:
                    result = OfertaAnalizada.model_validate(payload)
                    st.session_state["aggregate_result"] = result
                    st.success("Resultado agregado construido.")
                except Exception as e:
                    st.error(f"No fue posible validar contra el esquema principal: {e}")

    # =========================================================
    # Visualización común (agregado / por fichero local)
    # =========================================================
    per_file_results = st.session_state.get("per_file_results", {})
    aggregate_result = st.session_state.get("aggregate_result", None)

    if aggregate_result is not None or per_file_results:
        options = ["Agregado (todos)"] + list(per_file_results.keys())
        choice = st.selectbox("Vista de resultados (modo chunking local / agregado File Search)", options)

        if choice == "Agregado (todos)":
            if isinstance(aggregate_result, OfertaAnalizada):
                render_result(aggregate_result)
        else:
            r = per_file_results.get(choice)
            if isinstance(r, OfertaAnalizada):
                render_result(r)

        if isinstance(aggregate_result, OfertaAnalizada):
            with st.expander("Descargas del resultado agregado"):
                st.download_button(
                    "Descargar JSON (Agregado)",
                    aggregate_result.model_dump_json(indent=2),
                    file_name="analisis_agregado.json",
                    mime="application/json"
                )
                st.download_button(
                    "Descargar Markdown (Agregado)",
                    aggregate_result.to_markdown,
                    file_name="analisis_agregado.md",
                    mime="text/markdown"
                )


if __name__ == "__main__":
    main()
