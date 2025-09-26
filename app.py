# app.py
# -----------------------------------------------------------------------------------
# RFP Analyzer – Streamlit (consultoría TI) | Prompts “excelentes”
# - Modelos visibles: gpt-4o y gpt-4o-mini (temperatura fija = 0.2)
# - UX de una sola página (sin JSON visible); botones por secciones + “Análisis Completo”
# - File Search con Vector Store (OpenAI) + fallback local rápido y síntesis garantizada
# - Prompts reforzados: precisión factual, trazabilidad, evidencias, discrepancias
# -----------------------------------------------------------------------------------

import os
import io
import re
import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# -----------------------------------------------------------------------------------
# Config (robusto: intenta importar config.py y, si no existe, usa defaults/env)
# -----------------------------------------------------------------------------------
APP_TITLE = "RFP Analyzer (Consultoría TI)"
OPENAI_MODEL_DEFAULT = "gpt-4o"
OPENAI_API_KEY = None
ADMIN_USER = "admin"
ADMIN_PASSWORD = "rfpanalyzer"
MAX_TOKENS_PER_REQUEST = 1600

try:
    from config import (
        APP_TITLE as _APP_TITLE,
        OPENAI_MODEL as _OPENAI_MODEL,
        OPENAI_API_KEY as _OPENAI_API_KEY,
        ADMIN_USER as _ADMIN_USER,
        ADMIN_PASSWORD as _ADMIN_PASSWORD,
        MAX_TOKENS_PER_REQUEST as _MAX_TOKENS,
    )
    APP_TITLE = _APP_TITLE or APP_TITLE
    OPENAI_MODEL_DEFAULT = _OPENAI_MODEL or OPENAI_MODEL_DEFAULT
    OPENAI_API_KEY = _OPENAI_API_KEY or OPENAI_API_KEY
    ADMIN_USER = _ADMIN_USER or ADMIN_USER
    ADMIN_PASSWORD = _ADMIN_PASSWORD or ADMIN_PASSWORD
    MAX_TOKENS_PER_REQUEST = int(_MAX_TOKENS or MAX_TOKENS_PER_REQUEST)
except Exception:
    pass  # config.py opcional

# secrets/env fallback
OPENAI_API_KEY = OPENAI_API_KEY or st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# -----------------------------------------------------------------------------------
# OpenAI SDK (v1.x)
# -----------------------------------------------------------------------------------
if not OPENAI_API_KEY:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.error("No se encontró OPENAI_API_KEY. Configura `st.secrets` o variable de entorno.")
    st.stop()

try:
    from openai import OpenAI, BadRequestError
except Exception as e:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.error(f"No se pudo importar `openai`: {e}")
    st.stop()

_oai = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------------------------------------------------------
# PDF parsing: usa services.pdf_loader si existe; si no, fallback PyPDF2
# -----------------------------------------------------------------------------------
def _fallback_extract_pdf_text(file_like: io.BytesIO) -> Tuple[List[str], str]:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_like)
        pages = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            pages.append(t)
        return pages, "\n".join(pages)
    except Exception as e:
        raise RuntimeError(f"Fallo al parsear PDF (fallback): {e}")

try:
    from services.pdf_loader import extract_pdf_text as _svc_extract_pdf_text  # type: ignore
    def extract_pdf_text(file_like: io.BytesIO) -> Tuple[List[str], str]:
        return _svc_extract_pdf_text(file_like)
except Exception:
    def extract_pdf_text(file_like: io.BytesIO) -> Tuple[List[str], str]:
        return _fallback_extract_pdf_text(file_like)

# clean_text
try:
    from utils.text import clean_text as _svc_clean_text  # type: ignore
    def clean_text(s: str) -> str:
        return _svc_clean_text(s)
except Exception:
    def clean_text(s: str) -> str:
        s = s.replace("\x00", " ")
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\r?\n\s*\r?\n", "\n\n", s)
        return s.strip()

# -----------------------------------------------------------------------------------
# Parámetros de la app
# -----------------------------------------------------------------------------------
AVAILABLE_MODELS = ["gpt-4o", "gpt-4o-mini"]
FIXED_TEMPERATURE = 0.2
LOCAL_CONTEXT_MAX_CHARS = 40_000  # por doc en selección local
st.set_page_config(page_title=APP_TITLE, layout="wide")

# -----------------------------------------------------------------------------------
# Login
# -----------------------------------------------------------------------------------
def login_gate():
    if st.session_state.get("is_auth", False):
        with st.sidebar:
            if st.button("Cerrar sesión"):
                st.session_state.clear()
                st.rerun()
        return
    st.title("Acceso")
    with st.form("login_form"):
        u = st.text_input("Usuario")
        p = st.text_input("Contraseña", type="password")
        ok = st.form_submit_button("Entrar")
        if ok:
            if u == ADMIN_USER and p == ADMIN_PASSWORD:
                st.session_state["is_auth"] = True
                st.success("Acceso concedido.")
                st.rerun()
            else:
                st.error("Credenciales inválidas.")
    st.stop()

# -----------------------------------------------------------------------------------
# Prompts “excelentes”: SYSTEM + SECCIONES
# -----------------------------------------------------------------------------------
SYSTEM_PREFIX = (
    "Eres un analista sénior de licitaciones públicas en España y consultor de TI. "
    "Trabajas EXCLUSIVAMENTE con la información contenida en los PDFs adjuntos. "
    "Obligatorio: responde SIEMPRE con JSON VÁLIDO (UTF-8) y NADA MÁS. "
    "Nunca inventes; si falta información, usa null o listas vacías. "
    "Normaliza unidades y moneda; usa punto decimal (e.g., 12345.67). "
    "Usa SIEMPRE números JSON puros (sin separadores de miles). "
    "Incluye referencias de página únicas y orden ascendente. "
    "Las citas literales no deben superar 180 caracteres. "
    "Si hay varias cifras/interpretaciones para un mismo dato, indícalo en 'discrepancias' y en 'evidencias' aporta la más relevante. "
    "Incluye, cuando proceda, un campo opcional 'calidad_extraccion' con {\"texto_total\": int, \"texto_utilizado\": int}. "
    "Optimiza por: precisión factual > concisión > completitud. "
    "Evita repetir texto; sintetiza en cada campo y mantén coherencia terminológica. "
)

SECTION_SPECS: Dict[str, Dict[str, str]] = {
    "objetivos_contexto": {
        "titulo": "Objetivos y contexto",
        "user_prompt": (
            "Analiza los PDFs y extrae OBJETIVOS y CONTEXTO. "
            "Prioriza 'Objeto del contrato', 'Alcance', 'Descripción del servicio', 'Contexto'. "
            "Devuelve SOLO JSON con esta estructura (mantén claves, añade opcionales solo si hay datos):\n"
            "{\n"
            '  "resumen_servicios": str|null,\n'
            '  "objetivos": [str],\n'
            '  "alcance": str|null,\n'
            '  "referencias_paginas": [int],\n'
            '  "evidencias": [{"pagina": int, "cita": str}],\n'
            '  "discrepancias": [str],\n'
            '  "calidad_extraccion": {"texto_total": int, "texto_utilizado": int}|null\n'
            "}\n"
            "Reglas: no especules; si los objetivos/alcance no aparecen, usa null/[]; citas ≤180 chars."
        ),
    },
    "servicios": {
        "titulo": "Servicios solicitados (detalle)",
        "user_prompt": (
            "Identifica y lista los SERVICIOS SOLICITADOS con máximo detalle. "
            "Busca 'Servicios', 'Actividades', 'Tareas', 'Entregables', 'SLA/KPI', 'Periodicidad', 'Volumen'. "
            "Devuelve SOLO JSON:\n"
            "{\n"
            '  "resumen_servicios": str|null,\n'
            '  "servicios_detalle": [\n'
            '    {\n'
            '      "nombre": str,\n'
            '      "descripcion": str|null,\n'
            '      "entregables": [str],\n'
            '      "requisitos": [str],\n'
            '      "periodicidad": str|null,\n'
            '      "volumen": str|null,\n'
            '      "ubicacion_modalidad": str|null,\n'
            '      "sla_kpi": [{"nombre": str, "objetivo": str|null, "unidad": str|null, "metodo_medicion": str|null}],\n'
            '      "criterios_aceptacion": [str]\n'
            "    }\n"
            "  ],\n"
            '  "alcance": str|null,\n'
            '  "referencias_paginas": [int],\n'
            '  "evidencias": [{"pagina": int, "cita": str}],\n'
            '  "discrepancias": [str]\n'
            "}\n"
            "Reglas: deduplica servicios equivalentes; sintetiza; si un campo no aparece, deja null/[]; citas ≤180 chars."
        ),
    },
    "importe": {
        "titulo": "Importe de licitación",
        "user_prompt": (
            "Extrae importes: presupuesto base de licitación, anualidades/prórrogas e IVA si se explicita. "
            "Devuelve SOLO JSON (números decimales con punto):\n"
            "{\n"
            '  "importe_total": float|null,\n'
            '  "moneda": str|null,\n'
            '  "iva_incluido": bool|null,\n'
            '  "tipo_iva": float|null,\n'
            '  "importes_detalle": [\n'
            '    {"concepto": str|null, "importe": float|null, "moneda": str|null, "observaciones": str|null,\n'
            '     "periodo": {"tipo": "anualidad"|"prorroga"|null, "anyo": int|null, "duracion_meses": int|null}}\n'
            '  ],\n'
            '  "referencias_paginas": [int],\n'
            '  "evidencias": [{"pagina": int, "cita": str}],\n'
            '  "discrepancias": [str]\n'
            "}\n"
            "Reglas: si hay varias cifras para el mismo concepto, recoge todas en 'importes_detalle' y explica en 'discrepancias'."
        ),
    },
    "criterios_valoracion": {
        "titulo": "Criterios de valoración",
        "user_prompt": (
            "Extrae criterios y subcriterios con pesos y tipo (puntos/porcentaje), umbrales y método de evaluación si existe. "
            "Devuelve SOLO JSON:\n"
            "{\n"
            '  "criterios_valoracion": [\n'
            '    {\n'
            '      "nombre": str,\n'
            '      "peso_max": float|null,\n'
            '      "tipo": "puntos"|"porcentaje"|null,\n'
            '      "umbral_minimo": float|null,\n'
            '      "metodo_evaluacion": str|null,\n'
            '      "subcriterios": [\n'
            '        {"nombre": str, "peso_max": float|null, "tipo": "puntos"|"porcentaje"|null, "observaciones": str|null}\n'
            '      ]\n'
            '    }\n'
            '  ],\n'
            '  "criterios_desempate": [str],\n'
            '  "referencias_paginas": [int],\n'
            '  "evidencias": [{"pagina": int, "cita": str}],\n'
            '  "discrepancias": [str]\n'
            "}\n"
            "Reglas: conserva jerarquía; deduplica; si el peso se expresa en texto, extrae el número si es inequívoco."
        ),
    },
    "indice_tecnico": {
        "titulo": "Índice de la respuesta técnica",
        "user_prompt": (
            "1) Extrae el ÍNDICE SOLICITADO literal del pliego (si existe). "
            "2) Si no existe, propón un ÍNDICE ALINEADO con objetivos, alcance, servicios y criterios; implementable: "
            "   títulos claros, 1-2 líneas de descripción, subapartados accionables (entregables/evidencias). "
            "Devuelve SOLO JSON:\n"
            "{\n"
            '  "indice_respuesta_tecnica": [\n'
            '    {"titulo": str, "descripcion": str|null, "subapartados": [str]}\n'
            '  ],\n'
            '  "indice_propuesto": [\n'
            '    {"titulo": str, "descripcion": str|null, "subapartados": [str]}\n'
            '  ],\n'
            '  "trazabilidad": [{"propuesto": str, "solicitado_match": str|null}],\n'
            '  "referencias_paginas": [int],\n'
            '  "evidencias": [{"pagina": int, "cita": str}],\n'
            '  "discrepancias": [str]\n'
            "}\n"
            "Reglas: si no hay índice literal, 'indice_respuesta_tecnica' puede ir [], pero 'indice_propuesto' NO debe ir vacío."
        ),
    },
    "riesgos_exclusiones": {
        "titulo": "Riesgos y exclusiones",
        "user_prompt": (
            "Identifica RIESGOS (contractuales, técnicos/operativos, plazos, dependencias) y EXCLUSIONES. "
            "Si el pliego no los enumera, sintetiza riesgos plausibles basados EXCLUSIVAMENTE en lo que el pliego sí define "
            "(objetivos/alcance/servicios/criterios/condiciones). "
            "Devuelve SOLO JSON:\n"
            "{\n"
            '  "riesgos_y_dudas": str|null,\n'
            '  "exclusiones": [str],\n'
            '  "matriz_riesgos": [\n'
            '    {"riesgo": str, "probabilidad_1_5": int|null, "impacto_1_5": int|null,\n'
            '     "criticidad_1_25": int|null, "mitigacion": str|null, "responsable": str|null}\n'
            '  ],\n'
            '  "referencias_paginas": [int],\n'
            '  "evidencias": [{"pagina": int, "cita": str}],\n'
            '  "discrepancias": [str]\n'
            "}\n"
            "Reglas: no inventes fuera del pliego; si infieres, debe ser compatible con el contenido del pliego y anota la base."
        ),
    },
    "solvencia": {
        "titulo": "Criterios de solvencia",
        "user_prompt": (
            "Extrae SOLVENCIA (técnica, económica, administrativa/otros) y cómo se acredita (documentos/normas/umbrales). "
            "Devuelve SOLO JSON:\n"
            "{\n"
            '  "solvencia": {\n'
            '    "tecnica": [str],\n'
            '    "economica": [str],\n'
            '    "administrativa": [str],\n'
            '    "acreditacion": [{"requisito": str, "documento_necesario": str|null, "norma_referencia": str|null, "umbral": str|null}]\n'
            "  },\n"
            '  "referencias_paginas": [int],\n'
            '  "evidencias": [{"pagina": int, "cita": str}],\n'
            '  "discrepancias": [str]\n'
            "}\n"
            "Reglas: devuelve bullets atómicos (una condición por elemento); si no hay datos, []."
        ),
    },
}

# -----------------------------------------------------------------------------------
# Keywords (recall) + Tuning por sección
# -----------------------------------------------------------------------------------
SECTION_KEYWORDS = {
    "objetivos_contexto": {
        "objeto del contrato": 5, "objeto": 3, "alcance": 4, "objetivo": 3,
        "contexto": 3, "descripción del servicio": 4, "alcances": 3,
    },
    "servicios": {
        "servicios": 5, "actividades": 4, "tareas": 4, "entregables": 4,
        "nivel de servicio": 4, "sla": 3, "kpi": 3, "periodicidad": 3, "volumen": 3,
    },
    "importe": {
        "presupuesto base": 6, "importe": 5, "precio": 4, "iva": 4,
        "base imponible": 4, "prórroga": 4, "anualidad": 4, "licitación": 4,
    },
    "criterios_valoracion": {
        "criterios de valoración": 6, "criterios de adjudicación": 6,
        "baremo": 5, "puntuación": 5, "puntos": 4, "porcentaje": 4,
        "peso": 4, "umbral": 4, "desempate": 4, "fórmula": 4,
    },
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
        "responsabilidad": 4, "exenciones": 4, "penalizaciones": 5,
        "causas de exclusión": 6, "supuestos de exclusión": 6,
        "condiciones especiales": 4, "garantías": 4, "plazos": 4,
        "régimen sancionador": 5, "riesgos": 4, "restricciones": 4
    },
    "solvencia": {
        "solvencia técnica": 6, "solvencia económica": 6, "solvencia financiera": 5,
        "requisitos de solvencia": 6, "clasificación": 4, "experiencia": 4,
        "medios personales": 4, "medios materiales": 4, "acreditación": 5,
    },
}

SECTION_CONTEXT_TUNING = {
    "indice_tecnico": {"max_chars": 80_000, "window": 2},
    "riesgos_exclusiones": {"max_chars": 60_000, "window": 2},
}

# -----------------------------------------------------------------------------------
# Utilidades OpenAI (Responses) + robustez de parámetros
# -----------------------------------------------------------------------------------
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r'^\s*```(?:json)?\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s*```\s*$', '', s)
    return s

def _extract_json_block(s: str) -> str:
    m = re.search(r"\{[\s\S]*\}", s)
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
        raise RuntimeError("Cadena vacía tras limpiar cercas de código.")
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return json.loads(_extract_json_block(s))

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
    return (("unsupported_parameter" in s or "Unknown parameter" in s or "unexpected keyword" in s) and (param in s))

def _responses_create_robust(args: dict):
    """
    Envuelve OpenAI Responses y se adapta a variaciones del SDK:
    - si 'temperature' no es soportado, lo elimina
    - si 'response_format' no es soportado, lo elimina
    - renombra max_output_tokens -> max_completion_tokens si hace falta
    - mueve 'tools'/'attachments' a extra_body si el SDK no los acepta como kwargs
    """
    a = dict(args)
    extra = dict(a.pop("extra_body", {}) or {})

    # Evita “Unknown parameter: attachments” en ciertos SDK: pásalo por extra_body desde el inicio
    if "attachments" in a:
        extra["attachments"] = a.pop("attachments")

    for _ in range(5):
        try:
            if extra:
                return _oai.responses.create(**a, extra_body=extra)
            else:
                return _oai.responses.create(**a)
        except (BadRequestError, TypeError) as e:
            s = str(e)
            # temperatura no soportada
            if _is_temperature_error(e) or ("temperature" in s and "unexpected" in s.lower()):
                a.pop("temperature", None)
                continue
            # response_format no soportado
            if _is_unsupported_param(e, "response_format"):
                a.pop("response_format", None)
                continue
            # renombrar max_output_tokens
            if _is_unsupported_param(e, "max_output_tokens"):
                val = a.pop("max_output_tokens", None)
                if val is not None:
                    a["max_completion_tokens"] = val
                continue
            # tools no soportado → mover a extra_body
            if _is_unsupported_param(e, "tools"):
                tools = a.pop("tools", None)
                if tools is not None:
                    extra["tools"] = tools
                continue
            # attachments en kwargs → mover a extra_body
            if _is_unsupported_param(e, "attachments"):
                att = a.pop("attachments", None)
                if att is not None:
                    extra["attachments"] = att
                continue
            raise

# -----------------------------------------------------------------------------------
# Vector Store (File Search)
# -----------------------------------------------------------------------------------
def create_vector_store_from_streamlit_files(files, name: str = "RFP Vector Store") -> Tuple[str, List[str]]:
    store = _oai.vector_stores.create(
        name=name,
        expires_after={"anchor": "last_active_at", "days": 2},
    )
    file_ids = []
    for f in files:
        data = f.read()
        up = _oai.files.create(
            file=(f.name, data, "application/pdf"),
            purpose="assistants",
        )
        file_ids.append(up.id)
        _oai.vector_stores.files.create_and_poll(vector_store_id=store.id, file_id=up.id)
    return store.id, file_ids

# -----------------------------------------------------------------------------------
# Selección local de páginas relevantes (acelera el fallback local)
# -----------------------------------------------------------------------------------
def _score_page(text: str, weights: dict) -> int:
    if not text:
        return 0
    t = text.lower()
    score = 0
    for kw, w in weights.items():
        score += t.count(kw) * w
    return score

def _select_relevant_spans(pages: List[str], section_key: str,
                           max_chars: int = LOCAL_CONTEXT_MAX_CHARS, window: int = 1) -> str:
    # Tuning por sección
    tune = SECTION_CONTEXT_TUNING.get(section_key, {})
    max_chars = tune.get("max_chars", max_chars)
    window = tune.get("window", window)

    weights = SECTION_KEYWORDS.get(section_key, {})
    scored = [(_score_page(p, weights), i) for i, p in enumerate(pages)]
    scored.sort(reverse=True)

    used, total, selected = set(), 0, []
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
        # fallback: primeras 2 páginas
        for j, txt in enumerate(pages[:2]):
            if txt:
                selected.append(f"[Pág {j+1}]\n{txt}")
                total += len(txt)
                if total >= max_chars:
                    break

    return "\n\n".join(selected)

# -----------------------------------------------------------------------------------
# Llamadas a OpenAI (File Search + local + normalizador)
# -----------------------------------------------------------------------------------
def _force_jsonify_from_text(section_key: Optional[str], raw_text: str, model: str, temperature: Optional[float]):
    schema_hint = ""
    if section_key and section_key in SECTION_SPECS:
        schema_hint = SECTION_SPECS[section_key]["user_prompt"]

    sys_msg = {"role": "system", "content": [{"type": "input_text", "text": "Devuelve SOLO un JSON válido (UTF-8)."}]}
    usr_msg = {"role": "user", "content": [{
        "type": "input_text",
        "text": (
            "Convierte estrictamente a JSON VÁLIDO que cumpla el esquema. "
            "Usa null/[] si falta información.\n\n[ESQUEMA]\n"
            f"{schema_hint}\n\n[RESPUESTA]\n<<<\n{raw_text}\n>>>\n"
        ),
    }]}
    args = dict(
        model=model,
        input=[sys_msg, usr_msg],
        response_format={"type": "json_object"},
        max_output_tokens=MAX_TOKENS_PER_REQUEST,
        temperature=temperature,
    )
    rsp = _responses_create_robust(args)
    text = _coalesce_text_from_responses(rsp) or _extract_json_block(json.dumps(rsp, default=str))
    return _json_loads_robust(text)

def _file_search_section_call(
    user_prompt: str,
    model: str,
    temperature: Optional[float],
    file_ids: List[str],
    section_key: Optional[str] = None,
) -> Dict[str, Any]:
    sys_msg = {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PREFIX}]}
    usr_msg = {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}

    # Pasamos attachments por extra_body para máxima compatibilidad
    args = dict(
        model=model,
        input=[sys_msg, usr_msg],
        tools=[{"type": "file_search"}],
        response_format={"type": "json_object"},
        max_output_tokens=MAX_TOKENS_PER_REQUEST,
        temperature=temperature,
        extra_body={
            "attachments": [{"file_id": fid, "tools": [{"type": "file_search"}]} for fid in file_ids]
        },
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

def _local_singlecall_section(section_key: str, model: str, temperature: float, max_chars: int):
    docs = st.session_state.get("fs_local_docs", [])
    if not docs:
        raise RuntimeError("No hay texto local para fallback.")

    contexts = []
    for d in docs:
        sel = _select_relevant_spans(d["pages"], section_key, max_chars=max_chars)
        if sel:
            contexts.append(sel)
    context = "\n\n".join(contexts)[:120_000]

    sys_msg = {"role": "system", "content": [{"type": "input_text", "text": (
        "Eres un analista sénior de licitaciones y consultor de TI. "
        "Responde SOLO con JSON válido. Nunca inventes; null/[] si falta información."
    )}]}
    schema_hint = SECTION_SPECS[section_key]["user_prompt"]
    usr_msg = {"role": "user", "content": [{
        "type": "input_text",
        "text": (
            "Extrae la sección solicitada según el siguiente esquema (JSON). "
            "Usa EXCLUSIVAMENTE el contexto que sigue.\n\n[ESQUEMA]\n"
            f"{schema_hint}\n\n[CONTEXTO]\n<<<\n{context}\n>>>\n"
            "Responde solo con JSON válido."
        )
    }]}

    args = dict(
        model=model,
        input=[sys_msg, usr_msg],
        response_format={"type": "json_object"},
        max_output_tokens=MAX_TOKENS_PER_REQUEST,
        temperature=temperature,
    )
    rsp = _responses_create_robust(args)
    text = _coalesce_text_from_responses(rsp) or _extract_json_block(json.dumps(rsp, default=str))
    return _json_loads_robust(text)

def _is_section_empty(section_key: str, data: Dict[str, Any]) -> bool:
    if not data or not isinstance(data, dict):
        return True
    if section_key == "indice_tecnico":
        a = data.get("indice_respuesta_tecnica") or []
        b = data.get("indice_propuesto") or []
        return len(a) == 0 and len(b) == 0
    if section_key == "riesgos_exclusiones":
        ry = data.get("riesgos_y_dudas")
        ex = data.get("exclusiones") or []
        return (ry is None or str(ry).strip() == "") and len(ex) == 0 and not data.get("matriz_riesgos")
    return False

def _dedupe_sorted_pages(pages: List[int]) -> List[int]:
    try:
        return sorted(set(int(p) for p in pages if p is not None))
    except Exception:
        return pages or []

def _synthesis_fill_section(section_key: str, model: str, temperature: float):
    """Síntesis garantizada para índice técnico / riesgos si no hay literal claro."""
    docs = st.session_state.get("fs_local_docs", [])
    if not docs:
        return None

    snippets = []
    for d in docs:
        pages = d["pages"]
        # primeras páginas + relevantes
        head = "\n\n".join([f"[Pág {i+1}]\n{p}" for i, p in enumerate(pages[:2]) if p])
        kw = "indice_tecnico" if section_key == "indice_tecnico" else "riesgos_exclusiones"
        tail = _select_relevant_spans(pages, kw, max_chars=60_000, window=2)
        combined = "\n\n".join([head, tail]).strip()
        if combined:
            snippets.append(combined)
    context = "\n\n---\n\n".join(snippets)[:140_000]

    if section_key == "indice_tecnico":
        instruction = (
            "No hay índice literal claro. Constrúyelo a partir de objetivos, alcance, servicios y criterios del pliego. "
            "Devuelve SOLO JSON con claves obligatorias: 'indice_respuesta_tecnica' (puede ir []), "
            "'indice_propuesto' (NO vacío), 'trazabilidad' (opcional), 'referencias_paginas', 'evidencias', 'discrepancias'."
        )
        schema_hint = SECTION_SPECS["indice_tecnico"]["user_prompt"]
    else:
        instruction = (
            "Si el pliego no enumera riesgos/exclusiones, sintetiza riesgos plausibles basados EXCLUSIVAMENTE en el contenido del pliego "
            "(objetivos/alcance/servicios/criterios/condiciones). Devuelve SOLO JSON con "
            "'riesgos_y_dudas' (no vacío), 'exclusiones' si hay, 'matriz_riesgos' con PxI y mitigación si deducible, "
            "'referencias_paginas' y 'evidencias' si existen, 'discrepancias' si detectas incoherencias."
        )
        schema_hint = SECTION_SPECS["riesgos_exclusiones"]["user_prompt"]

    sys_msg = {"role": "system", "content": [{"type": "input_text", "text": "Responde SOLO con JSON válido."}]}
    usr_msg = {"role": "user", "content": [{
        "type": "input_text",
        "text": f"{instruction}\n\n[ESQUEMA]\n{schema_hint}\n\n[CONTEXTO]\n<<<\n{context}\n>>>\n"
    }]}

    args = dict(
        model=model,
        input=[sys_msg, usr_msg],
        response_format={"type": "json_object"},
        max_output_tokens=MAX_TOKENS_PER_REQUEST,
        temperature=temperature,
    )
    rsp = _responses_create_robust(args)
    text = _coalesce_text_from_responses(rsp) or _extract_json_block(json.dumps(rsp, default=str))
    return _json_loads_robust(text)

def run_section(section_key: str, model: str, temperature: float, max_chars: int, file_ids: List[str]) -> Tuple[Dict[str, Any], str]:
    """Ejecuta sección con File Search → fallback local → síntesis garantizada (si aplica)."""
    # 1) File Search
    try:
        data = _file_search_section_call(
            user_prompt=SECTION_SPECS[section_key]["user_prompt"],
            model=model,
            temperature=temperature,
            file_ids=file_ids,
            section_key=section_key,
        )
        if _is_section_empty(section_key, data):
            raise RuntimeError("File Search devolvió salida vacía para esta sección.")
        if "referencias_paginas" in data:
            data["referencias_paginas"] = _dedupe_sorted_pages(data.get("referencias_paginas") or [])
        return data, "file_search"
    except Exception:
        # 2) Local
        data = _local_singlecall_section(section_key, model=model, temperature=temperature, max_chars=max_chars)
        # 3) Síntesis garantizada
        if section_key in {"indice_tecnico", "riesgos_exclusiones"} and _is_section_empty(section_key, data):
            synth = _synthesis_fill_section(section_key, model=model, temperature=temperature)
            if isinstance(synth, dict):
                data = synth
        if "referencias_paginas" in data:
            data["referencias_paginas"] = _dedupe_sorted_pages(data.get("referencias_paginas") or [])
        return data, "local_single"

# -----------------------------------------------------------------------------------
# Cache de parseo PDF
# -----------------------------------------------------------------------------------
def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

@st.cache_data(show_spinner=False)
def parse_pdf_cached(name: str, content_bytes: bytes) -> Dict[str, Any]:
    pages, _ = extract_pdf_text(io.BytesIO(content_bytes))
    pages = [clean_text(p) for p in pages]
    total_chars = sum(len(p or "") for p in pages)
    return {"name": name, "pages": pages, "total_chars": total_chars, "hash": _sha256(content_bytes)}

# -----------------------------------------------------------------------------------
# Sidebar (solo modelo, temp fija)
# -----------------------------------------------------------------------------------
def sidebar_config() -> Tuple[str, float]:
    with st.sidebar:
        st.header("Configuración")
        # modelo restringido
        if OPENAI_MODEL_DEFAULT in AVAILABLE_MODELS:
            idx = AVAILABLE_MODELS.index(OPENAI_MODEL_DEFAULT)
        else:
            idx = 0
        model = st.selectbox("Modelo OpenAI", AVAILABLE_MODELS, index=idx)
        st.caption("Temperatura fija: 0.2")
    return model, FIXED_TEMPERATURE

# -----------------------------------------------------------------------------------
# Render de resultados (UX una sola vista, sin JSON visible)
# -----------------------------------------------------------------------------------
def _badge(text: str):
    st.markdown(
        f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;"
        f"background:#eef;border:1px solid #ccd;color:#334;font-size:12px'>{text}</span>",
        unsafe_allow_html=True
    )

def render_full_view(fs_sections: Dict[str, Any]):
    # KPIs
    oc = fs_sections.get("objetivos_contexto", {})
    objetivos = oc.get("objetivos") or []
    im = fs_sections.get("importe", {})
    imp_total = im.get("importe_total")
    moneda = im.get("moneda") or "€"
    imp_str = f"{imp_total:.2f} {moneda}" if isinstance(imp_total, (int, float)) else "—"
    sv = fs_sections.get("solvencia", {}).get("solvencia", {})
    solv_tec = len(sv.get("tecnica", []))
    solv_eco = len(sv.get("economica", []))
    solv_adm = len(sv.get("administrativa", []))

    st.markdown("### Resumen ejecutivo")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Importe total", imp_str)
    c2.metric("Objetivos", len(objetivos))
    c3.metric("Criterios solvencia (tot.)", solv_tec + solv_eco + solv_adm)
    c4.metric("Secciones completas", sum(1 for k, v in fs_sections.items() if v))

    st.divider()

    # Objetivos y contexto
    with st.expander("🎯 Objetivos y contexto", expanded=True):
        resumen = oc.get("resumen_servicios") or "—"
        alcance = oc.get("alcance") or "—"
        st.markdown(f"**Resumen de servicios:** {resumen}")
        if objetivos:
            st.markdown("**Objetivos**:")
            st.write("\n".join([f"- {o}" for o in objetivos]))
        st.markdown(f"**Alcance:** {alcance}")
        # evidencias / discrepancias
        evs = oc.get("evidencias") or []
        disc = oc.get("discrepancias") or []
        if evs or disc:
            colA, colB = st.columns(2)
            if evs:
                with colA:
                    st.caption("Evidencias")
                    for e in evs[:3]:
                        _badge(f"Pág {e.get('pagina')}")
                        st.write(f"“{e.get('cita','')[:180]}”")
            if disc:
                with colB:
                    st.caption("Discrepancias")
                    st.write("\n".join([f"- {d}" for d in disc]))

    # Servicios
    svs = fs_sections.get("servicios", {})
    with st.expander("🧩 Servicios solicitados (detalle)", expanded=False):
        st.markdown(f"**Resumen:** {svs.get('resumen_servicios') or '—'}")
        detalle = svs.get("servicios_detalle") or []
        if detalle:
            import pandas as pd
            rows = []
            for d in detalle:
                rows.append({
                    "Servicio": d.get("nombre"),
                    "Descripción": d.get("descripcion"),
                    "Entregables": ", ".join(d.get("entregables") or []),
                    "SLA/KPI": ", ".join([sk.get("nombre") for sk in (d.get("sla_kpi") or []) if sk.get("nombre")]),
                    "Periodicidad": d.get("periodicidad"),
                    "Volumen": d.get("volumen"),
                    "Ubicación": d.get("ubicacion_modalidad"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Sin servicios detallados explícitos en el texto analizado.")

    # Importe
    with st.expander("💶 Importe de licitación", expanded=False):
        st.markdown(f"**Importe total:** {imp_str}")
        iva_incl = im.get("iva_incluido")
        tip_iva = im.get("tipo_iva")
        st.write(f"- **IVA incluido:** {iva_incl if iva_incl is not None else '—'}")
        st.write(f"- **Tipo IVA:** {tip_iva if tip_iva is not None else '—'}")
        det = im.get("importes_detalle") or []
        if det:
            import pandas as pd
            st.dataframe(pd.DataFrame(det), use_container_width=True, hide_index=True)
        disc = im.get("discrepancias") or []
        if disc:
            st.caption("Discrepancias")
            st.write("\n".join([f"- {d}" for d in disc]))

    # Criterios de valoración
    cv_all = fs_sections.get("criterios_valoracion", {})
    cv = cv_all.get("criterios_valoracion") or []
    with st.expander("📊 Criterios de valoración", expanded=False):
        if cv:
            import pandas as pd
            rows = []
            for c in cv:
                rows.append({
                    "Criterio": c.get("nombre"),
                    "Peso máx": c.get("peso_max"),
                    "Tipo": c.get("tipo"),
                    "Umbral mín.": c.get("umbral_minimo"),
                    "Método eval.": c.get("metodo_evaluacion"),
                    "Subcriterios": "; ".join([sc.get("nombre") for sc in (c.get("subcriterios") or []) if sc.get("nombre")]),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            dmp = cv_all.get("criterios_desempate") or []
            if dmp:
                st.markdown("**Criterios de desempate:**")
                st.write("\n".join([f"- {x}" for x in dmp]))
        else:
            st.info("No se encontraron criterios explícitos.")

    # Índice técnico
    it = fs_sections.get("indice_tecnico", {})
    with st.expander("🗂️ Índice de la respuesta técnica", expanded=False):
        col1, col2 = st.columns(2)
        req = it.get("indice_respuesta_tecnica") or []
        prop = it.get("indice_propuesto") or []
        with col1:
            st.markdown("**Índice solicitado (literal)**")
            if req:
                st.write("\n".join([f"- {s.get('titulo')}" for s in req if s.get("titulo")]))
            else:
                st.info("Sin índice solicitado detectado.")
        with col2:
            st.markdown("**Índice propuesto (accionable)**")
            if prop:
                st.write("\n".join([f"- {s.get('titulo')}" for s in prop if s.get("titulo")]))
            else:
                st.info("Sin índice propuesto.")
        tr = it.get("trazabilidad") or []
        if tr:
            st.caption("Trazabilidad propuesto → solicitado")
            for t in tr[:8]:
                st.write(f"- **{t.get('propuesto')}** → {t.get('solicitado_match') or '—'}")
        disc = it.get("discrepancias") or []
        if disc:
            st.caption("Discrepancias")
            st.write("\n".join([f"- {d}" for d in disc]))

    # Riesgos / Exclusiones
    rx = fs_sections.get("riesgos_exclusiones", {})
    with st.expander("⚠️ Riesgos y exclusiones", expanded=False):
        ry = rx.get("riesgos_y_dudas")
        ex = rx.get("exclusiones") or []
        st.markdown(f"**Riesgos y dudas (síntesis):** {ry or '—'}")
        if ex:
            st.markdown("**Exclusiones:**")
            st.write("\n".join([f"- {e}" for e in ex]))
        mrx = rx.get("matriz_riesgos") or []
        if mrx:
            import pandas as pd
            st.caption("Matriz de riesgos (PxI)")
            st.dataframe(pd.DataFrame(mrx), use_container_width=True, hide_index=True)
        disc = rx.get("discrepancias") or []
        if disc:
            st.caption("Discrepancias")
            st.write("\n".join([f"- {d}" for d in disc]))

    # Solvencia
    solv = fs_sections.get("solvencia", {}).get("solvencia", {})
    with st.expander("📜 Solvencia y acreditación", expanded=False):
        col1, col2, col3 = st.columns(3)
        tec = solv.get("tecnica", [])
        eco = solv.get("economica", [])
        adm = solv.get("administrativa", [])
        with col1:
            st.markdown("**Técnica**")
            st.write("\n".join([f"- {x}" for x in tec]) or "—")
        with col2:
            st.markdown("**Económica**")
            st.write("\n".join([f"- {x}" for x in eco]) or "—")
        with col3:
            st.markdown("**Administrativa**")
            st.write("\n".join([f"- {x}" for x in adm]) or "—")
        acr = solv.get("acreditacion") or []
        if acr:
            import pandas as pd
            st.caption("Acreditación (cómo se demuestra)")
            st.dataframe(pd.DataFrame(acr), use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------------
# App principal
# -----------------------------------------------------------------------------------
def main():
    # Login
    login_gate()

    # Sidebar: modelo + temp fija
    model, temperature = sidebar_config()

    # Header
    st.title(APP_TITLE)
    st.caption("Analizador de pliegos con enfoque de consultoría TI. Modelos: GPT-4o / 4o-mini. Temperatura fija (0.2).")

    # Uploader
    files = st.file_uploader("Sube uno o varios PDFs del pliego", type=["pdf"], accept_multiple_files=True)
    if not files:
        st.info("Sube al menos un PDF para comenzar.")
        st.stop()

    # Indexación e ingest
    if "fs_vs_id" not in st.session_state or st.button("Reindexar PDFs en OpenAI"):
        st.session_state.pop("fs_sections", None)

        # Clonar contenido (los widgets consumen el stream)
        uploaded = [{"name": f.name, "bytes": f.read()} for f in files]

        # MemFile para API
        class _MemFile:
            def __init__(self, name, data): self.name, self._data = name, data
            def read(self): return self._data

        mem_files = [_MemFile(u["name"], u["bytes"]) for u in uploaded]

        with st.spinner("Indexando en Vector Store de OpenAI…"):
            vs_id, file_ids = create_vector_store_from_streamlit_files(mem_files, name="RFP Vector Store")

        # Texto local cacheado
        local_docs, char_stats = [], []
        for u in uploaded:
            parsed = parse_pdf_cached(u["name"], u["bytes"])
            local_docs.append({"name": u["name"], "pages": parsed["pages"]})
            char_stats.append((u["name"], len(parsed["pages"]), parsed["total_chars"]))

        st.session_state["fs_vs_id"] = vs_id
        st.session_state["fs_file_ids"] = file_ids
        st.session_state["fs_local_docs"] = local_docs
        st.session_state["char_stats"] = char_stats

        st.success("PDF(s) indexados y texto local preparado.")

    # Diagnóstico
    with st.expander("Diagnóstico de extracción", expanded=False):
        st.info(f"Vector Store listo: `{st.session_state['fs_vs_id']}` – {len(st.session_state['fs_file_ids'])} archivo(s)")
        for name, npages, nchar in st.session_state.get("char_stats", []):
            st.write(f"- **{name}**: {npages} páginas, {nchar} caracteres")
            if nchar < 1000:
                st.warning(f"{name}: muy poco texto (posible escaneado sin OCR).")

    # Estado
    st.session_state.setdefault("busy", False)
    st.session_state.setdefault("job", None)
    st.session_state.setdefault("job_all", False)

    def _start_job(section: Optional[str] = None, do_all: bool = False):
        st.session_state["job"] = section
        st.session_state["job_all"] = do_all
        st.session_state["busy"] = True

    # Controles
    st.subheader("Controles de análisis")
    dis = st.session_state["busy"]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.button("Objetivos y contexto",   use_container_width=True, disabled=dis,
                  on_click=_start_job, kwargs={"section": "objetivos_contexto"})
        st.button("Servicios solicitados",  use_container_width=True, disabled=dis,
                  on_click=_start_job, kwargs={"section": "servicios"})
        st.button("Importe de licitación",  use_container_width=True, disabled=dis,
                  on_click=_start_job, kwargs={"section": "importe"})
    with c2:
        st.button("Criterios de valoración", use_container_width=True, disabled=dis,
                  on_click=_start_job, kwargs={"section": "criterios_valoracion"})
        st.button("Índice de la respuesta técnica", use_container_width=True, disabled=dis,
                  on_click=_start_job, kwargs={"section": "indice_tecnico"})
        st.button("Riesgos y exclusiones",   use_container_width=True, disabled=dis,
                  on_click=_start_job, kwargs={"section": "riesgos_exclusiones"})
    with c3:
        st.button("Criterios de solvencia",  use_container_width=True, disabled=dis,
                  on_click=_start_job, kwargs={"section": "solvencia"})
        st.write("")
        st.button("🔎 Análisis Completo", type="primary", use_container_width=True, disabled=dis,
                  on_click=_start_job, kwargs={"do_all": True})

    # Ejecución
    if st.session_state["busy"]:
        with st.status("Procesando análisis…", expanded=True) as status:
            try:
                if st.session_state["job_all"]:
                    order = list(SECTION_SPECS.keys())
                    st.session_state.setdefault("fs_sections", {})
                    prog = st.progress(0.0)
                    for i, k in enumerate(order, start=1):
                        spec = SECTION_SPECS[k]
                        status.update(label=f"Analizando: {spec['titulo']}…")
                        try:
                            data, mode = run_section(
                                section_key=k,
                                model=model,
                                temperature=FIXED_TEMPERATURE,
                                max_chars=LOCAL_CONTEXT_MAX_CHARS,
                                file_ids=st.session_state["fs_file_ids"],
                            )
                            st.session_state["fs_sections"][k] = data
                            status.write(f"✓ {spec['titulo']} ({'File Search' if mode=='file_search' else 'Local'})")
                        except Exception as e:
                            st.session_state["fs_sections"][k] = {}
                            status.write(f"✗ {spec['titulo']}: {e}")
                        prog.progress(i / len(order))
                    status.update(label="Análisis completo finalizado", state="complete")
                else:
                    k = st.session_state["job"]
                    spec = SECTION_SPECS[k]
                    status.update(label=f"Analizando: {spec['titulo']}…")
                    data, mode = run_section(
                        section_key=k,
                        model=model,
                        temperature=FIXED_TEMPERATURE,
                        max_chars=LOCAL_CONTEXT_MAX_CHARS,
                        file_ids=st.session_state["fs_file_ids"],
                    )
                    st.session_state.setdefault("fs_sections", {})
                    st.session_state["fs_sections"][k] = data
                    status.update(label=f"Sección '{spec['titulo']}' completada", state="complete")
            finally:
                st.session_state["busy"] = False
                st.session_state["job"] = None
                st.session_state["job_all"] = False
                st.rerun()

    # Resultados (una sola vista, sin JSON visible)
    st.subheader("Resultados")
    fs_sections = st.session_state.get("fs_sections", {})
    if not fs_sections:
        st.info("Aún no hay resultados. Pulsa **Análisis Completo** o ejecuta alguna sección.")
    else:
        render_full_view(fs_sections)
        # Descargas (opcional: no mostramos JSON en pantalla)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Descargar análisis (Markdown)",
                data=_markdown_full(fs_sections=fs_sections),
                file_name="analisis_pliego.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                "Descargar JSON (Análisis completo)",
                data=json.dumps(fs_sections, indent=2, ensure_ascii=False),
                file_name="analisis_pliego.json",
                mime="application/json",
                use_container_width=True,
            )

# -----------------------------------------------------------------------------------
# Markdown consolidado para descarga
# -----------------------------------------------------------------------------------
def _markdown_full(fs_sections: Dict[str, Any]) -> str:
    out = ["# Análisis de Pliego – Resultado Completo\n"]
    def add(h): out.append(h if h.endswith("\n") else h + "\n")

    oc = fs_sections.get("objetivos_contexto", {})
    svs = fs_sections.get("servicios", {})
    im = fs_sections.get("importe", {})
    cv_all = fs_sections.get("criterios_valoracion", {})
    it = fs_sections.get("indice_tecnico", {})
    rx = fs_sections.get("riesgos_exclusiones", {})
    solv_all = fs_sections.get("solvencia", {}).get("solvencia", {})

    add("## Objetivos y contexto")
    add(f"- **Resumen**: {oc.get('resumen_servicios') or '—'}")
    if oc.get("objetivos"):
        add("**Objetivos**:")
        for o in oc["objetivos"]: add(f"- {o}")
    add(f"- **Alcance**: {oc.get('alcance') or '—'}\n")

    add("## Servicios solicitados")
    add(f"- **Resumen**: {svs.get('resumen_servicios') or '—'}")
    det = svs.get("servicios_detalle") or []
    if det:
        add("**Detalle:**")
        for d in det:
            add(f"- {d.get('nombre')}: {d.get('descripcion') or ''}")

    add("\n## Importe de licitación")
    imp_total = im.get("importe_total")
    moneda = im.get("moneda") or "€"
    add(f"- **Importe total**: {f'{imp_total:.2f} {moneda}' if isinstance(imp_total, (int,float)) else '—'}")
    add(f"- **IVA incluido**: {im.get('iva_incluido') if im.get('iva_incluido') is not None else '—'}")
    add(f"- **Tipo IVA**: {im.get('tipo_iva') if im.get('tipo_iva') is not None else '—'}")
    for d in (im.get("importes_detalle") or []):
        add(f"  - {d.get('concepto') or '—'}: {d.get('importe')} {d.get('moneda') or moneda} "
            f"({d.get('observaciones') or ''})")

    add("\n## Criterios de valoración")
    cv = cv_all.get("criterios_valoracion") or []
    if cv:
        for c in cv:
            add(f"- {c.get('nombre')} (peso: {c.get('peso_max')} {c.get('tipo') or ''}; "
                f"umbral: {c.get('umbral_minimo')}; método: {c.get('metodo_evaluacion')})")
            for s in (c.get("subcriterios") or []):
                add(f"  - {s.get('nombre')} (peso: {s.get('peso_max')} {s.get('tipo') or ''})")
        if cv_all.get("criterios_desempate"):
            add("**Desempate:**")
            for x in cv_all["criterios_desempate"]: add(f"- {x}")
    else:
        add("- —")

    add("\n## Índice de la respuesta técnica")
    req = it.get("indice_respuesta_tecnica") or []
    prop = it.get("indice_propuesto") or []
    add("**Solicitado**:")
    if req:
        for s in req: add(f"- {s.get('titulo')}")
    else:
        add("- —")
    add("\n**Propuesto**:")
    if prop:
        for s in prop: add(f"- {s.get('titulo')}")
    else:
        add("- —")

    add("\n## Riesgos y exclusiones")
    add(f"- **Riesgos y dudas**: {rx.get('riesgos_y_dudas') or '—'}")
    ex = rx.get("exclusiones") or []
    if ex:
        add("**Exclusiones:**")
        for e in ex: add(f"- {e}")
    mrx = rx.get("matriz_riesgos") or []
    if mrx:
        add("**Matriz de riesgos (PxI):**")
        for r in mrx:
            add(f"- {r.get('riesgo')}: P={r.get('probabilidad_1_5')} I={r.get('impacto_1_5')}, "
                f"C={r.get('criticidad_1_25')}; mit.: {r.get('mitigacion')}; resp.: {r.get('responsable')}")

    add("\n## Solvencia y acreditación")
    tec = solv_all.get("tecnica", [])
    eco = solv_all.get("economica", [])
    adm = solv_all.get("administrativa", [])
    if tec:
        add("**Técnica:**");  [add(f"- {x}") for x in tec]
    if eco:
        add("**Económica:**"); [add(f"- {x}") for x in eco]
    if adm:
        add("**Administrativa:**"); [add(f"- {x}") for x in adm]
    acr = solv_all.get("acreditacion") or []
    if acr:
        add("**Acreditación:**")
        for a in acr:
            add(f"- {a.get('requisito')}: doc={a.get('documento_necesario')}, "
                f"norma={a.get('norma_referencia')}, umbral={a.get('umbral')}")

    return "\n".join(out)

# -----------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
