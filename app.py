# app.py
# ---------------------------------------------------------------------
# RFP Analyzer – Streamlit (vista única mejorada)
#  - Modelos visibles: gpt-4o y gpt-4o-mini
#  - Temperatura fija = 0.2 (no editable)
#  - Solo 1 página (sin pestañas): controles arriba + “Vista completa” como referencia (sin JSON)
#  - “Análisis Completo” ejecuta TODAS las secciones SECUENCIALMENTE (una a una)
#  - PDF completo con File Search; si falla/queda vacío → fallback local rápido (1 llamada/ sección)
#  - Cacheo de parseo PDF y selección de páginas relevantes por sección (keyword scoring) para acelerar
# ---------------------------------------------------------------------

import os
import sys
import io
import json
import re
import hashlib
from typing import Optional, Dict, Any, List

import streamlit as st
from openai import OpenAI, BadRequestError

# ---------------------------------------------------------------------
# Imports de la app (se asume que existen en tu repo)
# ---------------------------------------------------------------------
try:
    from config import (
        APP_TITLE,
        OPENAI_MODEL,
        OPENAI_API_KEY,
        ADMIN_USER,
        ADMIN_PASSWORD,
        MAX_TOKENS_PER_REQUEST,
    )
except Exception as e:
    st.error(f"Falta o no es importable `config.py`: {e}")
    st.stop()

try:
    from components.ui import render_header
except Exception:
    # Fallback mínimo si no existiera el componente
    def render_header(title: str):
        st.title(title)

try:
    from services.pdf_loader import extract_pdf_text
except Exception as e:
    st.error(f"No se pudo importar services.pdf_loader.extract_pdf_text: {e}")
    st.stop()

try:
    from utils.text import clean_text
except Exception as e:
    st.error(f"No se pudo importar utils.text.clean_text: {e}")
    st.stop()


# ---------------------------------------------------------------------
# Parámetros de esta versión
# ---------------------------------------------------------------------
AVAILABLE_MODELS = ["gpt-4o", "gpt-4o-mini"]
FIXED_TEMPERATURE = 0.2
LOCAL_CONTEXT_MAX_CHARS = 40_000       # por doc en selección de páginas relevantes
SECOND_TAB_KEY = "full_view_ready"     # compat (no hay pestañas, lo mantenemos por si se usa fuera)


# ---------------------------------------------------------------------
# Página / Tema
# ---------------------------------------------------------------------
st.set_page_config(page_title="RFP Analyzer", layout="wide")


# ---------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------
def login_gate():
    """Bloquea la UI hasta hacer login; botón de logout en la sidebar."""
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
    # SDK moderno
    txt = getattr(rsp, "output_text", None)
    if txt:
        return txt
    # Compat: explorar estructura
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
    return (
        ("unsupported_parameter" in s or "Unexpected" in s or "unexpected" in s or "Unknown parameter" in s)
        and (param in s)
    )


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

    for _ in range(4):
        try:
            if extra:
                return _oai_client.responses.create(**a, extra_body=extra)
            else:
                return _oai_client.responses.create(**a)
        except (BadRequestError, TypeError) as e:
            s = str(e)

            if _is_temperature_error(e) or ("unexpected keyword" in s and "temperature" in s):
                a.pop("temperature", None)
                continue

            if _is_unsupported_param(e, "response_format") or ("unexpected keyword" in s and "response_format" in s):
                a.pop("response_format", None)
                continue

            if _is_unsupported_param(e, "max_output_tokens") or ("unexpected keyword" in s and "max_output_tokens" in s):
                val = a.pop("max_output_tokens", None)
                if val is not None:
                    a["max_completion_tokens"] = val
                continue

            if _is_unsupported_param(e, "attachments") or ("unexpected keyword" in s and "attachments" in s):
                att = a.pop("attachments", None)
                if att is not None:
                    extra["attachments"] = att
                continue

            if _is_unsupported_param(e, "tools") or ("unexpected keyword" in s and "tools" in s):
                tools = a.pop("tools", None)
                if tools is not None:
                    extra["tools"] = tools
                continue

            raise


def create_vector_store_from_streamlit_files(files, name: str = "RFP Vector Store"):
    """
    Sube los PDFs a OpenAI, los indexa en un Vector Store y
    devuelve (vector_store_id, file_ids) para uso con 'attachments'.
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
# Secciones (prompts para File Search / Local)
# ---------------------------------------------------------------------
SYSTEM_PREFIX = (
    "Eres un analista sénior de licitaciones públicas en España. "
    "Trabajas SOLO con la información contenida en los PDFs adjuntos. "
    "Obligatorio: responde SIEMPRE con JSON VÁLIDO (UTF-8) y NADA MÁS. "
    "Nunca inventes; si falta información, usa null o listas vacías. "
    "Normaliza unidades y moneda; usa punto decimal (e.g., 12345.67). "
    "Incluye referencias de página de donde extraes cada dato (si existen). "
    "Si detectas inconsistencias, indícalas en un campo 'discrepancias' con breve explicación. "
    "Optimiza por: precisión factual > concisión > completitud. "
    "Evita repetir texto; sintetiza. "
)

SECTION_SPECS = {
    "objetivos_contexto": {
        "titulo": "Objetivos y contexto",
        "user_prompt": (
            "Analiza los PDFs y extrae OBJETIVOS y CONTEXTO. "
            "Prioriza los apartados tipo 'Objeto del contrato', 'Alcance', 'Descripción del servicio', 'Contexto'. "
            "Devuelve SOLO JSON con esta estructura (mantén claves, añade solo los opcionales si hay datos):\n"
            "{\n"
            '  "resumen_servicios": str|null,             // síntesis ejecutiva (máx. 120-150 palabras)\n'
            '  "objetivos": [str],                        // 5-12 bullets, accionables\n'
            '  "alcance": str|null,                       // qué incluye; si hay límites, resume en 1-2 frases\n'
            '  "referencias_paginas": [int],              // páginas únicas, orden ascendente\n'
            '  "evidencias": [{"pagina": int, "cita": str}],  // 1-3 citas cortas literales\n'
            '  "discrepancias": [str]                     // conflictos, ambigüedades o lagunas\n'
            "}\n"
            "Reglas: no especules; si los objetivos/alcance no aparecen, deja null/[]; "
            "las citas deben ser breves (≤240 caracteres)."
        ),
    },
    "servicios": {
        "titulo": "Servicios solicitados (detalle)",
        "user_prompt": (
            "Identifica y lista los SERVICIOS SOLICITADOS con el máximo detalle. "
            "Busca secciones de 'Servicios', 'Actividades', 'Tareas', 'Entregables', 'SLA/KPI', 'Periodicidad', 'Volumen'. "
            "Devuelve SOLO JSON:\n"
            "{\n"
            '  "resumen_servicios": str|null, \n'
            '  "servicios_detalle": [\n'
            '    {\n'
            '      "nombre": str,                         // etiqueta sintética del servicio\n'
            '      "descripcion": str|null,               // 1-3 frases precisas\n'
            '      "entregables": [str],                  // opcional\n'
            '      "requisitos": [str],                   // SLAs/KPIs/NFRs si aparecen\n'
            '      "periodicidad": str|null,              // p.ej., mensual, semanal, bajo demanda\n'
            '      "volumen": str|null,                   // p.ej., #usuarios, #horas, #tickets/mes\n'
            '      "ubicacion_modalidad": str|null        // remoto/presencial, on-site/off-site\n'
            "    }\n"
            "  ],\n"
            '  "alcance": str|null,\n'
            '  "referencias_paginas": [int],\n'
            '  "evidencias": [{"pagina": int, "cita": str}],\n'
            '  "discrepancias": [str]\n'
            "}\n"
            "Reglas: deduplica servicios equivalentes; sintetiza; si algún campo no aparece, déjalo null/[]."
        ),
    },
    "importe": {
        "titulo": "Importe de licitación",
        "user_prompt": (
            "Extrae importes: presupuesto base de licitación, posibles anualidades/prórrogas, e IVA si se explicita. "
            "Normaliza a número decimal con punto y devuelve moneda detectada. "
            "Devuelve SOLO JSON:\n"
            "{\n"
            '  "importe_total": float|null,               // total principal indicado (siempre que sea inequívoco)\n'
            '  "moneda": str|null,                        // p.ej., EUR\n'
            '  "importes_detalle": [\n'
            '    {"concepto": str|null, "importe": float|null, "moneda": str|null, "observaciones": str|null}\n'
            '  ],\n'
            '  "referencias_paginas": [int],\n'
            '  "evidencias": [{"pagina": int, "cita": str}],\n'
            '  "discrepancias": [str]\n'
            "}\n"
            "Reglas: si hay varios importes (p.ej., por anualidad/prórroga), incluye cada uno en importes_detalle con observaciones. "
            "Si no hay cifra clara, usa null y explica en 'discrepancias'."
        ),
    },
    "criterios_valoracion": {
        "titulo": "Criterios de valoración",
        "user_prompt": (
            "Extrae criterios de valoración y subcriterios, con pesos máximos y tipo (puntos o porcentaje). "
            "Si el pliego mezcla unidades, indica el valor numérico y el tipo. "
            "Devuelve SOLO JSON:\n"
            "{\n"
            '  "criterios_valoracion": [\n'
            '    {\n'
            '      "nombre": str,\n'
            '      "peso_max": float|null,                // valor numérico del peso\n'
            '      "tipo": str|null,                      // \"puntos\" | \"porcentaje\" | null\n'
            '      "subcriterios": [\n'
            '        {"nombre": str, "peso_max": float|null, "tipo": str|null, "observaciones": str|null}\n'
            '      ]\n'
            '    }\n'
            '  ],\n'
            '  "referencias_paginas": [int],\n'
            '  "evidencias": [{"pagina": int, "cita": str}],\n'
            '  "discrepancias": [str]\n'
            "}\n"
            "Reglas: conserva la jerarquía; deduplica; si el peso se expresa en texto, extrae el número si es inequívoco."
        ),
    },
    "indice_tecnico": {
        "titulo": "Índice de la respuesta técnica",
        "user_prompt": (
            "Tarea:\n"
            "1) Extrae el ÍNDICE SOLICITADO literal del pliego (si existe).\n"
            "2) Si no existe literal, propón un ÍNDICE ALINEADO con los objetivos, alcance, servicios y criterios de valoración que se desprenden del pliego.\n"
            "3) La propuesta debe ser concreta y utilizable: títulos claros, 1-2 líneas de descripción por apartado y subapartados accionables.\n\n"
            "Devuelve SOLO JSON:\n"
            "{\n"
            '  "indice_respuesta_tecnica": [\n'
            '    {"titulo": str, "descripcion": str|null, "subapartados": [str]}\n'
            '  ],\n'
            '  "indice_propuesto": [\n'
            '    {"titulo": str, "descripcion": str|null, "subapartados": [str]}\n'
            '  ],\n'
            '  "trazabilidad": [                          // mapea cada propuesto→solicitado o null si no aplica\n'
            '    {"propuesto": str, "solicitado_match": str|null}\n'
            '  ],\n'
            '  "referencias_paginas": [int],\n'
            '  "evidencias": [{"pagina": int, "cita": str}],\n'
            '  "discrepancias": [str]\n'
            "}\n"
            "Reglas:\n"
            "- Si no hay índice literal, pon [] en 'indice_respuesta_tecnica' pero SIEMPRE rellena 'indice_propuesto'.\n"
            "- Sé conciso y específico; evita títulos vacíos o genéricos.\n"
            "- Incluye referencias de página y 1-3 citas cortas si existen fuentes explícitas.\n"
        ),
    },
    "riesgos_exclusiones": {
        "titulo": "Riesgos y exclusiones",
        "user_prompt": (
            "Identifica RIESGOS (contractuales, técnicos/operativos, plazos, dependencias) y EXCLUSIONES.\n"
            "Si el pliego no los enumera explícitamente, infiere riesgos plausibles basados en objetivos, alcance, servicios, criterios y condiciones contractuales del propio pliego.\n"
            "Devuelve SOLO JSON:\n"
            "{\n"
            '  "riesgos_y_dudas": str|null,               // síntesis de 3-6 frases\n'
            '  "exclusiones": [str],                      // literal si existen; si no, []\n'
            '  "referencias_paginas": [int],\n'
            '  "evidencias": [{"pagina": int, "cita": str}],\n'
            '  "discrepancias": [str]\n'
            "}\n"
            "Reglas:\n"
            "- No inventes datos fuera del pliego; si infieres, debe ser compatible con lo que el pliego sí establece (menciona la base).\n"
            "- Prioriza precisión factual > concisión > completitud.\n"
        ),
    },
    "solvencia": {
        "titulo": "Criterios de solvencia",
        "user_prompt": (
            "Extrae criterios de SOLVENCIA (técnica, económica, administrativa/otros). "
            "Reconoce sinónimos (p.ej., 'capacidad', 'experiencia mínima', 'clasificación empresarial'). "
            "Devuelve SOLO JSON:\n"
            "{\n"
            '  "solvencia": {\n'
            '    "tecnica": [str],\n'
            '    "economica": [str],\n'
            '    "administrativa": [str]\n'
            "  },\n"
            '  "referencias_paginas": [int],\n'
            '  "evidencias": [{"pagina": int, "cita": str}],\n'
            '  "discrepancias": [str]\n'
            "}\n"
            "Reglas: devuelve bullets atómicos (una condición por elemento); si no hay datos, [] en la categoría correspondiente."
        ),
    },
}



# ---------------------------------------------------------------------
# Selección de páginas relevantes por sección (acelera local)
# ---------------------------------------------------------------------
SECTION_KEYWORDS.update({
    "indice_tecnico": {
        "índice": 6, "indice": 6, "estructura": 5, "estructura mínima": 6,
        "contenido de la oferta": 6, "contenido mínimo": 6, "memoria técnica": 5,
        "documentación técnica": 5, "apartados": 4, "secciones": 4,
        "instrucciones de preparación": 5, "formato de la propuesta": 5,
        "orden de contenidos": 5, "capítulos": 4, "anexos": 3,
        "presentación de ofertas": 4, "sobre técnico": 5
    },
    "riesgos_exclusiones": {
        "exclusiones": 7, "no incluye": 7, "quedan excluidos": 7,
        "no serán objeto": 6, "limitaciones": 5, "incompatibilidades": 5,
        "responsabilidad": 4, "exenciones": 4, "penalizaciones": 4,
        "causas de exclusión": 6, "supuestos de exclusión": 6,
        "condiciones especiales": 4, "garantías": 4, "plazos": 4,
        "régimen sancionador": 5, "riesgos": 4, "restricciones": 4
    },
})

SECTION_CONTEXT_TUNING = {
    # max chars por doc y ventana de páginas para _select_relevant_spans
    "indice_tecnico": {"max_chars": 80000, "window": 2},
    "riesgos_exclusiones": {"max_chars": 60000, "window": 2},
}

def _select_relevant_spans(pages: List[str], section_key: str,
                           max_chars: int = LOCAL_CONTEXT_MAX_CHARS, window: int = 1) -> str:
    # override si hay tuning específico
    tune = SECTION_CONTEXT_TUNING.get(section_key, {})
    max_chars = tune.get("max_chars", max_chars)
    window    = tune.get("window", window)
    # ... (resto de la función sin cambios)

def _score_page(text: str, weights: dict) -> int:
    if not text:
        return 0
    t = text.lower()
    s = 0
    for kw, w in weights.items():
        s += t.count(kw) * w
    return s


def _select_relevant_spans(pages: List[str], section_key: str,
                           max_chars: int = LOCAL_CONTEXT_MAX_CHARS, window: int = 1) -> str:
    weights = SECTION_KEYWORDS.get(section_key, {})
    scored = [(_score_page(p, weights), i) for i, p in enumerate(pages)]
    scored.sort(reverse=True)  # mayor puntuación primero

    selected, total, used = [], 0, set()
    for sc, i in scored:
        if sc <= 0:
            break
        for j in range(max(0, i - window), min(len(pages), i + window + 1)):
            if j in used:
                continue
            txt = pages[j]
            if not txt:
                continue
            if total + len(txt) > max_chars:
                break
            selected.append(f"[Pág {j+1}]\n{txt}")
            used.add(j)
            total += len(txt)
        if total >= max_chars:
            break

    if not selected:
        for j, txt in enumerate(pages[:3]):
            if not txt:
                continue
            selected.append(f"[Pág {j+1}]\n{txt}")
            total += len(txt)
            if total >= max_chars:
                break

    return "\n\n".join(selected)


# ---------------------------------------------------------------------
# File Search (Responses) – llamada con adjuntos
# ---------------------------------------------------------------------
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
        temperature=float(temperature) if temperature is not None else None,
    )
    rsp = _responses_create_robust(args)
    text = _coalesce_text_from_responses(rsp)
    if not text:
        dump = json.dumps(rsp, default=str)
        text = _extract_json_block(dump)
    return _json_loads_robust(text)


def _file_search_section_call(
    vector_store_id: str,                 # no usado directamente (compat)
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
        temperature=float(temperature) if temperature is not None else None,
    )
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
# Fallback local rápido: UNA llamada por sección
# ---------------------------------------------------------------------
def _local_singlecall_section(section_key: str, model: str, temperature: float, max_chars: int):
    docs = st.session_state.get("fs_local_docs", [])
    if not docs:
        raise RuntimeError("No hay texto local para fallback. Reindexa los PDFs.")

    # Construye contexto con solo páginas relevantes de cada documento
    contexts = []
    for d in docs:
        sel = _select_relevant_spans(d["pages"], section_key, max_chars=max_chars)
        if sel:
            contexts.append(sel)
    context = "\n\n".join(contexts)[:120_000]  # tapón de seguridad

    sys_msg = {"role": "system", "content": [{"type": "input_text", "text": (
        "Eres un analista experto en licitaciones públicas en España. "
        "Devuelve SIEMPRE JSON válido (sin texto adicional). "
        "Si el contexto no contiene la información, usa null/listas vacías."
    )}]}
    schema_hint = SECTION_SPECS[section_key]["user_prompt"]
    usr = {"role": "user", "content": [{
        "type": "input_text",
        "text": (
            f"Extrae SOLO la sección solicitada según este esquema (JSON):\n{schema_hint}\n\n"
            "Usa EXCLUSIVAMENTE el siguiente contexto (fragmentos del pliego):\n<<<\n"
            + context + "\n>>>\n"
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

    rsp = _responses_create_robust(args)
    text = _coalesce_text_from_responses(rsp)
    if not text:
        text = _extract_json_block(json.dumps(rsp, default=str))
    return _json_loads_robust(text)


def _run_section_with_fallback(section_key: str, vs_id: str, file_ids: List[str],
                               model: str, temperature: float, max_chars: int):
    # 1º intento: File Search
    try:
        data = _file_search_section_call(
            vector_store_id=vs_id,
            user_prompt=SECTION_SPECS[section_key]["user_prompt"],
            model=model,
            temperature=temperature,
            file_ids=file_ids,
            section_key=section_key,
        )
        if not data or all((v in (None, [], {}) for v in data.values())):
            raise RuntimeError("File Search devolvió JSON vacío.")
        return data, "file_search"
    except Exception:
        # 2º intento: local rápido con UNA llamada
        data = _local_singlecall_section(section_key, model=model,
                                         temperature=temperature, max_chars=max_chars)
        return data, "local_single"


# ---------------------------------------------------------------------
# Cacheo de parseo PDF
# ---------------------------------------------------------------------
def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@st.cache_data(show_spinner=False)
def parse_pdf_cached(name: str, content_bytes: bytes):
    # Cachea por hash de bytes para no reprocesar un PDF idéntico
    h = _sha256(content_bytes)
    pages, _ = extract_pdf_text(io.BytesIO(content_bytes))
    pages = [clean_text(p) for p in pages]
    total_chars = sum(len(p or "") for p in pages)
    return {"name": name, "pages": pages, "total_chars": total_chars, "hash": h}


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
    return model, FIXED_TEMPERATURE


# ---------------------------------------------------------------------
# Render de “Vista completa” (sin JSON)
# ---------------------------------------------------------------------
def render_full_view(fs_sections: Dict[str, Any]):
    st.markdown("### Vista completa del análisis")
    # Métricas principales
    c1, c2, c3 = st.columns(3)
    oc = fs_sections.get("objetivos_contexto", {})
    objetivos = oc.get("objetivos") or []
    im = fs_sections.get("importe", {})
    imp_total = im.get("importe_total")
    moneda = im.get("moneda") or "€"
    imp_str = f"{imp_total:,.2f} {moneda}" if isinstance(imp_total, (int, float)) else "—"
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
        resumen = oc.get("resumen_servicios") or "—"
        alcance = oc.get("alcance") or "—"
        st.markdown(f"**Resumen de servicios:** {resumen}")
        if objetivos:
            st.markdown("**Objetivos**:")
            st.write("\n".join([f"- {o}" for o in objetivos]))
        st.markdown(f"**Alcance:** {alcance}")

    # Servicios solicitados
    svs = fs_sections.get("servicios", {})
    with st.expander("🧩 Servicios solicitados (detalle)", expanded=False):
        st.markdown(f"**Resumen:** {svs.get('resumen_servicios') or '—'}")
        detalle = svs.get("servicios_detalle") or []
        if detalle:
            import pandas as pd
            df = pd.DataFrame([{"Servicio": d.get("nombre"), "Descripción": d.get("descripcion")} for d in detalle])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No se encontraron servicios detallados explícitos en el texto analizado.")

    # Importe y desglose
    with st.expander("💶 Importe de licitación", expanded=False):
        st.markdown(f"**Importe total:** {imp_str}")
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
                st.write("\n".join([f"- {s.get('titulo')}" for s in req if s.get("titulo")]))
            else:
                st.info("Sin índice solicitado detectado.")
        with col2:
            st.markdown("**Índice propuesto**")
            if prop:
                st.write("\n".join([f"- {s.get('titulo')}" for s in prop if s.get("titulo")]))
            else:
                st.info("Sin índice propuesto.")

    # Riesgos / Exclusiones
    rx = fs_sections.get("riesgos_exclusiones", {})
    with st.expander("⚠️ Riesgos y exclusiones", expanded=False):
        ry = rx.get("riesgos_y_dudas")
        ex = rx.get("exclusiones") or []
        st.markdown(f"**Riesgos y dudas:** {ry or '—'}")
        if ex:
            st.markdown("**Exclusiones:**")
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


def _markdown_full(fs_sections: Dict[str, Any]) -> str:
    """Genera un Markdown consolidado para descarga."""
    parts = ["# Análisis de Pliego – Resultado Completo\n"]
    oc = fs_sections.get("objetivos_contexto", {})
    svs = fs_sections.get("servicios", {})
    im = fs_sections.get("importe", {})
    cv = fs_sections.get("criterios_valoracion", {}).get("criterios_valoracion", [])
    it = fs_sections.get("indice_tecnico", {})
    rx = fs_sections.get("riesgos_exclusiones", {})
    sv = fs_sections.get("solvencia", {}).get("solvencia", {})

    parts.append("## Objetivos y contexto\n")
    parts.append(f"- **Resumen**: {oc.get('resumen_servicios') or '—'}")
    if oc.get("objetivos"):
        parts.append("**Objetivos**:\n" + "\n".join([f"- {o}" for o in oc["objetivos"]]))
    parts.append(f"- **Alcance**: {oc.get('alcance') or '—'}\n")

    parts.append("## Servicios solicitados\n")
    parts.append(f"- **Resumen**: {svs.get('resumen_servicios') or '—'}")
    det = svs.get("servicios_detalle") or []
    if det:
        parts.append("**Detalle:**\n" + "\n".join([f"- {d.get('nombre')}: {d.get('descripcion') or ''}" for d in det]))

    parts.append("\n## Importe de licitación\n")
    imp_total = im.get("importe_total")
    moneda = im.get("moneda") or "€"
    parts.append(f"- **Importe total**: {f'{imp_total:,.2f} {moneda}' if isinstance(imp_total,(int,float)) else '—'}")
    deti = im.get("importes_detalle") or []
    if deti:
        parts.append("**Desglose:**\n" + "\n".join([
            f"- {d.get('concepto') or '—'}: {d.get('importe')} {d.get('moneda') or moneda} ({d.get('observaciones') or ''})"
            for d in deti
        ]))

    parts.append("\n## Criterios de valoración\n")
    if cv:
        for c in cv:
            parts.append(f"- {c.get('nombre')} (peso: {c.get('peso_max')}, tipo: {c.get('tipo')})")
            sc = c.get("subcriterios") or []
            for s in sc:
                parts.append(f"  - {s.get('nombre')} (peso: {s.get('peso_max')}, tipo: {s.get('tipo')})")
    else:
        parts.append("- —")

    parts.append("\n## Índice de la respuesta técnica\n")
    req = it.get("indice_respuesta_tecnica") or []
    prop = it.get("indice_propuesto") or []
    parts.append("**Solicitado**:\n" + ("\n".join([f"- {s.get('titulo')}" for s in req if s.get("titulo")]) or "- —"))
    parts.append("\n**Propuesto**:\n" + ("\n".join([f"- {s.get('titulo')}" for s in prop if s.get("titulo")]) or "- —"))

    parts.append("\n## Riesgos y exclusiones\n")
    parts.append(f"- **Riesgos y dudas**: {rx.get('riesgos_y_dudas') or '—'}")
    ex = rx.get("exclusiones") or []
    if ex:
        parts.append("**Exclusiones**:\n" + "\n".join([f"- {e}" for e in ex]))

    parts.append("\n## Solvencia\n")
    parts.append("**Técnica**:\n" + ("\n".join([f"- {x}" for x in sv.get("tecnica", [])]) or "- —"))
    parts.append("\n**Económica**:\n" + ("\n".join([f"- {x}" for x in sv.get("economica", [])]) or "- —"))
    parts.append("\n**Administrativa**:\n" + ("\n".join([f"- {x}" for x in sv.get("administrativa", [])]) or "- —"))

    return "\n".join(parts)


# ---------------------------------------------------------------------
# App principal (UNA SOLA VISTA)
# ---------------------------------------------------------------------
def main():
    # 1) Login
    login_gate()

    # 2) Config (sidebar)
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

    # 5) Indexación en OpenAI + captura de texto local (cacheado)
    if "fs_vs_id" not in st.session_state or st.button("Reindexar PDFs en OpenAI"):
        # Limpiar resultados previos
        st.session_state.pop("fs_sections", None)
        st.session_state.pop(SECOND_TAB_KEY, None)

        uploaded = [{"name": f.name, "bytes": f.read()} for f in files]

        # crear objetos memfile para la API de OpenAI
        class _MemFile:
            def __init__(self, name, data): self.name = name; self._data = data
            def read(self): return self._data

        mem_files = [_MemFile(u["name"], u["bytes"]) for u in uploaded]

        with st.spinner("Indexando PDFs en OpenAI (Vector Store)..."):
            vs_id, file_ids = create_vector_store_from_streamlit_files(mem_files, name="RFP Vector Store")

        # Texto local cacheado y diagnóstico
        local_docs = []
        char_stats = []
        for u in uploaded:
            parsed = parse_pdf_cached(u["name"], u["bytes"])
            local_docs.append({"name": u["name"], "pages": parsed["pages"]})
            char_stats.append((u["name"], len(parsed["pages"]), parsed["total_chars"]))

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
    diag_box = st.expander("Diagnóstico de extracción de texto", expanded=False)
    with diag_box:
        st.info(f"Vector Store listo: `{vs_id}` – {len(file_ids)} archivo(s) adjuntables")
        for name, npages, nchar in st.session_state.get("char_stats", []):
            st.write(f"- **{name}**: {npages} páginas, {nchar} caracteres")
            if nchar < 1000:
                st.warning(f"{name}: muy poco texto extraído (posible PDF escaneado sin OCR).")

    # 6) Estado de ejecución (para evitar duplicados visuales)
    st.session_state.setdefault("busy", False)
    st.session_state.setdefault("job", None)        # sección a ejecutar
    st.session_state.setdefault("job_all", False)   # ejecutar todas

    def _start_job(section: str | None = None, do_all: bool = False):
        st.session_state["job"] = section
        st.session_state["job_all"] = do_all
        st.session_state["busy"] = True

    # 7) Controles (arriba) + Vista completa debajo
    st.subheader("Controles de análisis")

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

    # 8) Ejecución controlada (SECUENCIAL para Análisis Completo)
    if st.session_state["busy"]:
        with st.status("Procesando análisis…", expanded=True) as status:
            try:
                if st.session_state["job_all"]:
                    order = list(SECTION_SPECS.keys())
                    st.session_state.setdefault("fs_sections", {})
                    prog = st.progress(0.0)
                    for i, k in enumerate(order, start=1):
                        spec = SECTION_SPECS[k]
                        status.update(label=f"Analizando sección: {spec['titulo']}…")
                        data, mode_used = _run_section_with_fallback(
                            section_key=k,
                            vs_id=st.session_state["fs_vs_id"],
                            file_ids=st.session_state["fs_file_ids"],
                            model=model,
                            temperature=FIXED_TEMPERATURE,
                            max_chars=LOCAL_CONTEXT_MAX_CHARS,
                        )
                        st.session_state["fs_sections"][k] = data
                        status.write(f"✓ {spec['titulo']} ({'File Search' if mode_used=='file_search' else 'Local'})")
                        prog.progress(i/len(order))
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
                        temperature=FIXED_TEMPERATURE,
                        max_chars=LOCAL_CONTEXT_MAX_CHARS,
                    )
                    st.session_state.setdefault("fs_sections", {})
                    st.session_state["fs_sections"][k] = data
                    status.update(label=f"Sección '{spec['titulo']}' completada", state="complete")
            finally:
                st.session_state["busy"] = False
                st.session_state["job"] = None
                st.session_state["job_all"] = False
                st.rerun()

    # 9) Vista completa (referencia) + descargas (sin JSON)
    st.subheader("Resultados")
    fs_sections = st.session_state.get("fs_sections", {})
    if not fs_sections:
        st.info("Aún no hay resultados. Pulsa **Análisis Completo** o ejecuta alguna sección arriba.")
    else:
        render_full_view(fs_sections)
        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                "Descargar JSON – Análisis completo",
                json.dumps(fs_sections, indent=2, ensure_ascii=False),
                file_name="analisis_completo.json",
                mime="application/json",
                use_container_width=True
            )
        with col_b:
            st.download_button(
                "Descargar Markdown – Análisis completo",
                _markdown_full(fs_sections),
                file_name="analisis_completo.md",
                mime="text/markdown",
                use_container_width=True
            )


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
