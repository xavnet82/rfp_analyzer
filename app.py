
_OCR_ENABLED = True  # set False to disable OCR heuristic programáticamente

# --- Streamlit integration for OCR diagnostics ---
try:
    import streamlit as st
except Exception:
    st = None

# Ensure pytesseract is configured (Windows) and check engine availability
try:
    from ocr_helper import tesseract_available, configure_pytesseract_for_windows
    configure_pytesseract_for_windows()
    _HAS_TESSERACT = tesseract_available()
except Exception:
    _HAS_TESSERACT = False

# app.py
# -----------------------------------------------------------------------------------
# RFP Analyzer – Streamlit (consultoría TI) | "PDF completo" + fallback local
# - Modelos visibles: gpt-4o y gpt-4o-mini (temperatura configurable desde sidebar)
# - UX de una sola vista de análisis + una pestaña de "Registro (Prompts/Respuestas)"
# - Envío de PDFs como input_file a /responses (sin tools ni attachments)
# - Fallback local con selección de páginas relevantes y síntesis garantizada
# - Prompts robustos: JSON obligatorio, evidencias, discrepancias, no alucinar
# - Incluye “Formato y entrega de la oferta”
# - Log tab: expander para TODAS las secciones (incluida Formato/Entrega)
# - Índice técnico: ahora se renderiza título + descripción + subapartados
# -----------------------------------------------------------------------------------

import os
import io
import re
import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# -----------------------------------------------------------------------------------
# Config (intenta importar config.py; si no, usa defaults/env)
# -----------------------------------------------------------------------------------
APP_TITLE = "RFP Analyzer (Consultoría TI)"
OPENAI_MODEL_DEFAULT = "gpt-4o"
OPENAI_API_KEY = None
ADMIN_USER = "admin"
ADMIN_PASSWORD = "rfpanalyzer"
MAX_TOKENS_PER_REQUEST = 1800  # salida de cada sección (JSON)
DEFAULT_TEMPERATURE = 0.2

try:
    from config import (  # type: ignore
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
    pass

# secrets/env fallback
OPENAI_API_KEY = OPENAI_API_KEY or st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# Página (temprano)
st.set_page_config(page_title=APP_TITLE, layout="wide")

# -----------------------------------------------------------------------------------
# OpenAI SDK (v1.x)
# -----------------------------------------------------------------------------------
if not OPENAI_API_KEY:
    st.error("No se encontró OPENAI_API_KEY. Configura `st.secrets` o variable de entorno.")
    st.stop()

try:
    from openai import OpenAI, BadRequestError
except Exception as e:
    st.error(f"No se pudo importar `openai`: {e}")
    st.stop()

_oai = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------------------------------------------------------
# PDF parsing: intenta services.pdf_loader; si no, fallback PyPDF2
# -----------------------------------------------------------------------------------
def _fallback_extract_pdf_text(file_like: io.BytesIO) -> Tuple[List[str], str]:
    '''
    OCR-capable PDF text extractor.
    1) Try PyPDF2 extract_text per page.
    2) If total text is very small, run OCR with pypdfium2 + pytesseract (spa+eng).
    3) Return (pages, full_text).
    '''
    try:
        from PyPDF2 import PdfReader
        import io as _io
        b = file_like.read()
        reader = PdfReader(_io.BytesIO(b))
        pages: List[str] = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            pages.append(t)

        total_chars = sum(len(x) for x in pages)
        # Heuristic: if too little text, try OCR
        if _OCR_ENABLED and total_chars < 1500:
            try:
                # OCR path
                import pdfium  # pypdfium2
                # Warn in Streamlit if no Tesseract binary is found
                global _HAS_TESSERACT
                if (globals().get('st', None) is not None) and not _HAS_TESSERACT:
                    st.warning('OCR activado pero **no se encontró** el binario de Tesseract en el sistema. Instálalo o desactiva OCR. Se seguirá usando el texto extraído con PyPDF2 cuando exista.')
                from PIL import Image
                import pytesseract

                pdf = pdfium.PdfDocument(b)
                pages_ocr: List[str] = []
                for i in range(len(pdf)):
                    page = pdf.get_page(i)
                    # scale~2.0 ~= 150-200 DPI for most A4 PDFs
                    pil_image = page.render(scale=2.0).to_pil()
                    txt = pytesseract.image_to_string(pil_image, lang="spa+eng")
                    pages_ocr.append(txt or "")
                # If OCR improved things, adopt it
                if sum(len(x) for x in pages_ocr) > total_chars:
                    pages = pages_ocr
            except Exception:
                # OCR best-effort; keep PyPDF2 text if OCR fails
                pass

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

def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\r?\n\s*\r?\n", "\n\n", s)
    return s.strip()

# -----------------------------------------------------------------------------------
# Parámetros app
# -----------------------------------------------------------------------------------
AVAILABLE_MODELS = ["gpt-4o", "gpt-4o-mini"]
LOCAL_CONTEXT_MAX_CHARS = 40_000  # por sección (fallback local)

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
  "Eres analista sénior de licitaciones en España y consultor TI."
  " Respondes EXCLUSIVAMENTE con JSON VÁLIDO (UTF-8) y NADA MÁS."
  " Si una clave no aplica o no hay evidencia, devuélvela igualmente con valor null o []."
  " NUNCA inventes; usa SOLO información de los PDFs."
  " Normaliza números decimales con punto y moneda explícita (p. ej., EUR)."
  " Números sin separador de miles. Citas ≤ 180 caracteres."
  " Incluye referencias de página como enteros únicos en orden ascendente cuando existan."
  " Si hay versiones alternativas de un dato, usa 'discrepancias' para explicarlo brevemente."
  " Optimiza por: precisión factual > concisión > completitud."
)

SECTION_SPECS: Dict[str, Dict[str, str]] = {
  "objetivos_contexto": {
    "titulo": "Objetivos y contexto",
    "user_prompt": (
      "Extrae objetivos y contexto del pliego. Devuelve SIEMPRE las claves listadas."
      "\nSalida JSON EXACTA con claves:\n"
      "{"
      '  "resumen_servicios": str|null,'
      '  "objetivos": [str],'
      '  "alcance": str|null,'
      '  "referencias_paginas": [int],'
      '  "evidencias": [{"pagina": int, "cita": str}],'
      '  "discrepancias": [str]'
      "}"
      "\nReglas: no inventes; si no hay dato, usa null/[]."
    ),
  },
  "servicios": {
    "titulo": "Servicios solicitados (detalle)",
    "user_prompt": (
      "Lista servicios solicitados y detalles (entregables, SLAs/KPIs, etc.). Devuelve SIEMPRE las claves."
      "\nSalida JSON EXACTA:\n"
      "{"
      '  "resumen_servicios": str|null,'
      '  "servicios_detalle": ['
      '    {'
      '      "nombre": str,'
      '      "descripcion": str|null,'
      '      "entregables": [str],'
      '      "requisitos": [str],'
      '      "periodicidad": str|null,'
      '      "volumen": str|null,'
      '      "ubicacion_modalidad": str|null,'
      '      "sla_kpi": [{"nombre": str, "objetivo": str|null, "unidad": str|null, "metodo_medicion": str|null}],'
      '      "criterios_aceptacion": [str]'
      '    }'
      '  ],'
      '  "alcance": str|null,'
      '  "referencias_paginas": [int],'
      '  "evidencias": [{"pagina": int, "cita": str}],'
      '  "discrepancias": [str]'
      "}"
      "\nReglas: deduplica; no inventes; null/[] si no hay."
    ),
  },
  "importe": {
    "titulo": "Importe de licitación",
    "user_prompt": (
      "Extrae importes y condiciones (IVA, anualidades/prórrogas). Devuelve SIEMPRE las claves."
      "\nSalida JSON EXACTA:\n"
      "{"
      '  "importe_total": float|null,'
      '  "moneda": str|null,'
      '  "iva_incluido": bool|null,'
      '  "tipo_iva": float|null,'
      '  "importes_detalle": ['
      '    {"concepto": str|null, "importe": float|null, "moneda": str|null, "observaciones": str|null,'
      '     "periodo": {"tipo": "anualidad"|"prorroga"|null, "anyo": int|null, "duracion_meses": int|null}}'
      '  ],'
      '  "referencias_paginas": [int],'
      '  "evidencias": [{"pagina": int, "cita": str}],'
      '  "discrepancias": [str]'
      "}"
      "\nReglas: números con punto; si hay varias cifras, recoge todas en importes_detalle y usa discrepancias."
    ),
  },
  "criterios_valoracion": {
    "titulo": "Criterios de valoración",
    "user_prompt": (
      "Extrae criterios/subcriterios con pesos, tipo, umbrales, método y desempates. Devuelve SIEMPRE las claves."
      "\nSalida JSON EXACTA:\n"
      "{"
      '  "criterios_valoracion": ['
      '    {'
      '      "nombre": str,'
      '      "peso_max": float|null,'
      '      "tipo": "puntos"|"porcentaje"|null,'
      '      "umbral_minimo": float|null,'
      '      "metodo_evaluacion": str|null,'
      '      "subcriterios": ['
      '        {"nombre": str, "peso_max": float|null, "tipo": "puntos"|"porcentaje"|null, "observaciones": str|null}'
      '      ]'
      '    }'
      '  ],'
      '  "criterios_desempate": [str],'
      '  "referencias_paginas": [int],'
      '  "evidencias": [{"pagina": int, "cita": str}],'
      '  "discrepancias": [str]'
      "}"
      "\nReglas: conserva jerarquía; null/[] si no hay."
    ),
  },
  "indice_tecnico": {
    "titulo": "Índice de la respuesta técnica",
    "user_prompt": (
      # *** Prompt reforzado según tu enunciado ***
      "1) Analiza en detalle la propuesta e identifica, si existe, el índice solicitado literal para la respuesta técnica. "
      "2) Si no existiera, propón en base al pliego un índice alineado (implementable), que contenga al menos: contexto, "
      "nuestro enfoque, metodología, alcance y actividades, planificación con hitos, equipo y roles, governance, gestión de calidad y SLAs, "
      "gestión de riesgos/continuidad, ciberseguridad/compliance, sostenibilidad/accesibilidad y anexos. "
      "El índice propuesto NO debe ir vacío y debe ser accionable (con subapartados)."
      "\nSalida JSON EXACTA:\n"
      "{"
      '  "indice_respuesta_tecnica": ['
      '    {"titulo": str, "descripcion": str|null, "subapartados": [str]}'
      '  ],'
      '  "indice_propuesto": ['
      '    {"titulo": str, "descripcion": str|null, "subapartados": [str]}'
      '  ],'
      '  "trazabilidad": [{"propuesto": str, "solicitado_match": str|null}],'
      '  "referencias_paginas": [int],'
      '  "evidencias": [{"pagina": int, "cita": str}],'
      '  "discrepancias": [str]'
      "}"
      "\nReglas: si no hay índice literal, 'indice_respuesta_tecnica' puede ir [], pero 'indice_propuesto' DEBE incluir >=10 apartados con subapartados clave."
    ),
  },
  "riesgos_exclusiones": {
    "titulo": "Riesgos y exclusiones",
    "user_prompt": (
      "Identifica riesgos y exclusiones del pliego. Si no hay lista explícita, sintetiza riesgos compatibles con lo definido."
      " Devuelve SIEMPRE las claves."
      "\nSalida JSON EXACTA:\n"
      "{"
      '  "riesgos_y_dudas": str|null,'
      '  "exclusiones": [str],'
      '  "matriz_riesgos": ['
      '    {"riesgo": str, "probabilidad_1_5": int|null, "impacto_1_5": int|null,'
      '     "criticidad_1_25": int|null, "mitigacion": str|null, "responsable": str|null}'
      '  ],'
      '  "referencias_paginas": [int],'
      '  "evidencias": [{"pagina": int, "cita": str}],'
      '  "discrepancias": [str]'
      "}"
      "\nReglas: no inventes fuera del pliego; si infieres, debe ser coherente con lo que SÍ aparece."
    ),
  },
  "solvencia": {
    "titulo": "Criterios de solvencia",
    "user_prompt": (
      "Extrae solvencia técnica, económica y administrativa y cómo se acredita. Devuelve SIEMPRE las claves."
      "\nSalida JSON EXACTA:\n"
      "{"
      '  "solvencia": {'
      '    "tecnica": [str],'
      '    "economica": [str],'
      '    "administrativa": [str],'
      '    "acreditacion": [{"requisito": str, "documento_necesario": str|null, "norma_referencia": str|null, "umbral": str|null}]'
      '  },'
      '  "referencias_paginas": [int],'
      '  "evidencias": [{"pagina": int, "cita": str}],'
      '  "discrepancias": [str]'
      "}"
      "\nReglas: una condición por bullet; null/[] si no hay."
    ),
  },
  "formato_oferta": {
    "titulo": "Formato y entrega de la oferta",
    "user_prompt": (
      "Extrae requisitos de formato y entrega de la oferta (memoria técnica/administrativa): extensión máxima, tamaño de fuente, "
      "tipografía, interlineado, márgenes, estructura documental requerida, idioma, número de copias, formatos de archivo, tamaño máximo, "
      "firma digital y quién firma, paginación/numeración, etiquetado de sobres/archivos, canal de entrega (plataforma/sobre electrónico), "
      "plazo/fecha/hora y zona horaria, instrucciones de presentación y anexos obligatorios. Devuelve SIEMPRE las claves."
      "\nSalida JSON EXACTA:\n"
      "{"
      '  "formato_esperado": str|null,'
      '  "longitud_paginas": int|null,'
      '  "tipografia": {"familia": str|null, "tamano_min": float|null, "interlineado": float|null, "margenes": str|null},'
      '  "estructura_documental": [ {"titulo": str, "observaciones": str|null} ],'
      '  "requisitos_presentacion": [str],'
      '  "requisitos_archivo": {'
      '     "formatos_permitidos": [str],'
      '     "tamano_max_mb": float|null,'
      '     "firma_digital_requerida": bool|null,'
      '     "firmado_por": str|null'
      '  },'
      '  "idioma": str|null,'
      '  "copias": int|null,'
      '  "entrega": {'
      '     "canal": str|null,'
      '     "plazo": str|null,'
      '     "zona_horaria": str|null,'
      '     "instrucciones": [str]'
      '  },'
      '  "paginacion": {"requerida": bool|null, "formato": str|null},'
      '  "etiquetado": [str],'
      '  "anexos_obligatorios": [str],'
      '  "confidencialidad": str|null,'
      '  "referencias_paginas": [int],'
      '  "evidencias": [{"pagina": int, "cita": str}],'
      '  "discrepancias": [str]'
      "}"
      "\nReglas: no inventes; null/[] si no hay; convierte longitudes numéricas cuando sea posible."
    ),
  },
}

# -----------------------------------------------------------------------------------
# Palabras clave para selección local (recall)
# -----------------------------------------------------------------------------------
SECTION_KEYWORDS = {
  "objetivos_contexto": {"objeto del contrato": 5, "objeto": 3, "alcance": 4, "objetivo": 3,
                         "contexto": 3, "descripción del servicio": 4, "alcances": 3
    ,
"resumen_ejecutivo": {
    "duración del contrato": 5,
    "plazo de ejecución": 5,
    "vigencia": 3,
    "periodo de ejecución": 4,
    "periodo de prestación": 3,
    "prórrogas": 2,
    "fecha límite de presentación": 5,
    "plazo de presentación de ofertas": 5,
    "fecha y hora de presentación": 5,
    "presentación de proposiciones": 4,
    "fecha tope de entrega": 4,
    "sobres electrónicos": 2,
    "licitación electrónica": 2
}
},
  "servicios": {"servicios": 5, "actividades": 4, "tareas": 4, "entregables": 4,
                "nivel de servicio": 4, "sla": 3, "kpi": 3, "periodicidad": 3, "volumen": 3},
  "importe": {"presupuesto base": 6, "importe": 5, "precio": 4, "iva": 4,
              "base imponible": 4, "prórroga": 4, "anualidad": 4, "licitación": 4},
  "criterios_valoracion": {"criterios de valoración": 6, "criterios de adjudicación": 6,
                           "baremo": 5, "puntuación": 5, "puntos": 4, "porcentaje": 4,
                           "peso": 4, "umbral": 4, "desempate": 4, "fórmula": 4},
  "indice_tecnico": {"índice": 6, "indice": 6, "estructura": 5, "estructura mínima": 6,
                     "contenido de la oferta": 6, "contenido mínimo": 6, "memoria técnica": 5,
                     "documentación técnica": 5, "apartados": 4, "secciones": 4,
                     "instrucciones de preparación": 5, "formato de la propuesta": 5,
                     "orden de contenidos": 5, "capítulos": 4, "anexos": 3,
                     "presentación de ofertas": 4, "sobre técnico": 5},
  "riesgos_exclusiones": {"exclusiones": 7, "no incluye": 7, "quedan excluidos": 7,
                          "no serán objeto": 6, "limitaciones": 5, "incompatibilidades": 5,
                          "responsabilidad": 4, "exenciones": 4, "penalizaciones": 5,
                          "causas de exclusión": 6, "supuestos de exclusión": 6,
                          "condiciones especiales": 4, "garantías": 4, "plazos": 4,
                          "régimen sancionador": 5, "riesgos": 4, "restricciones": 4},
  "solvencia": {"solvencia técnica": 6, "solvencia económica": 6, "solvencia financiera": 5,
                "requisitos de solvencia": 6, "clasificación": 4, "experiencia": 4,
                "medios personales": 4, "medios materiales": 4, "acreditación": 5},
  "formato_oferta": {"formato": 6, "formato de la oferta": 7, "formato de la propuesta": 6,
                     "presentación de ofertas": 7, "presentacion de ofertas": 7, "presentación": 5,
                     "memoria técnica": 6, "longitud": 6, "páginas": 6, "paginas": 6, "extensión": 6, "extension": 6,
                     "tamaño de letra": 6, "tamano de letra": 6, "tipografía": 5, "tipografia": 5,
                     "interlineado": 5, "márgenes": 5, "margenes": 5, "fuente": 5, "tipo de letra": 5,
                     "etiquetado": 5, "rotulación": 5, "rotulacion": 5, "sobres": 6, "sobre electrónico": 6,
                     "plataforma": 6, "perfil del contratante": 5, "archivo pdf": 6, "pdf": 5, "docx": 4,
                     "firma electrónica": 6, "firma digital": 6, "firmado": 5,
                     "idioma": 5, "copia": 5, "copias": 5, "paginación": 5, "numeración": 5,
                     "fecha de entrega": 6, "plazo de presentación": 6, "hora": 5, "zona horaria": 4},
}

SECTION_CONTEXT_TUNING = {
    "indice_tecnico": {"max_chars": 80_000, "window": 2},
    "riesgos_exclusiones": {"max_chars": 60_000, "window": 2},
    "formato_oferta": {"max_chars": 60_000, "window": 2},
}

# -----------------------------------------------------------------------------------
# Utilidades OpenAI (Responses) + robustez
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
    a = dict(args)
    for _ in range(5):
        try:
            return _oai.responses.create(**a)
        except (BadRequestError, TypeError) as e:
            s = str(e)
            if _is_temperature_error(e) or ("temperature" in s and "unexpected" in s.lower()):
                a.pop("temperature", None); continue
            if _is_unsupported_param(e, "response_format"):
                a.pop("response_format", None); continue
            if _is_unsupported_param(e, "max_output_tokens"):
                val = a.pop("max_output_tokens", None)
                if val is not None:
                    a["max_completion_tokens"] = val
                continue
            raise

# -----------------------------------------------------------------------------------
# Logging de prompts/respuestas
# -----------------------------------------------------------------------------------
def _log_init():
    # Asegurar que existen todas las claves (incluida formato_oferta)
    st.session_state.setdefault("logs", {})
    for k in SECTION_SPECS.keys():
        st.session_state["logs"].setdefault(k, [])

def _log_event(section_key: str, fase: str, prompt: str, response_text: str, modelo: str):
    _log_init()
    st.session_state["logs"][section_key].append({
        "fase": fase,            # "input_file", "local_fallback", "synthesis", etc.
        "model": modelo,
        "prompt": prompt,
        "response": response_text,
    })

# -----------------------------------------------------------------------------------
# Envío a OpenAI con PDFs como input_file (sin tools)
# -----------------------------------------------------------------------------------
def _file_input_section_call(
    user_prompt: str,
    model: str,
    temperature: Optional[float],
    file_ids: List[str],
    section_key: Optional[str] = None,
) -> Dict[str, Any]:
    sys_msg = {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PREFIX}]}
    content = [{"type": "input_text", "text": user_prompt}]
    for fid in file_ids:
        content.append({"type": "input_file", "file_id": fid})
    usr_msg = {"role": "user", "content": content}

    log_prompt = f"[SYSTEM]\n{SYSTEM_PREFIX}\n\n[USER]\n{user_prompt}\n\n[FILES]\n" + "\n".join(file_ids)

    args = dict(
        model=model,
        input=[sys_msg, usr_msg],
        response_format={"type": "json_object"},
        max_output_tokens=MAX_TOKENS_PER_REQUEST,
        temperature=temperature,
    )
    rsp = _responses_create_robust(args)
    raw_text = _coalesce_text_from_responses(rsp) or json.dumps(rsp, default=str)
    _log_event(section_key or "desconocida", "input_file", log_prompt, raw_text, model)

    if not raw_text:
        try:
            raw_text = _extract_json_block(json.dumps(rsp, default=str))
        except Exception:
            return _force_jsonify_from_text(section_key, json.dumps(rsp, default=str), model, temperature)

    try:
        return _json_loads_robust(raw_text)
    except Exception:
        return _force_jsonify_from_text(section_key, raw_text, model, temperature)

# -----------------------------------------------------------------------------------
# Selección local de páginas relevantes (fallback)
# -----------------------------------------------------------------------------------
def _score_page(text: str, weights: dict) -> int:
    if not text:
        return 0
    t = text.lower()
    score = 0
    # Defensive: weights might contain non-numeric values (e.g., dicts)
    for kw, w in (weights or {}).items():
        if not isinstance(kw, str):
            continue
        # normalize weight
        if isinstance(w, (int, float)):
            ww = float(w)
        elif isinstance(w, dict):
            # try common keys
            for k in ("weight", "w", "score", "val"):
                if k in w and isinstance(w[k], (int, float)):
                    ww = float(w[k])
                    break
            else:
                ww = 1.0
        else:
            ww = 1.0
        try:
            score += t.count(kw.lower()) * ww
        except Exception:
            # if kw has regex specials or other issues, fallback to plain find loop
            cnt = 0
            start = 0
            while True:
                idx = t.find(kw.lower(), start)
                if idx == -1:
                    break
                cnt += 1
                start = idx + max(1, len(kw))
            score += cnt * ww

    # Bonus por posibles encabezados de secciones relevantes
    if any(h in t for h in [
        "criterios de valoración", "criterios de adjudicación", "presentación de ofertas",
        "formato de la oferta", "formato de la propuesta", "índice", "indice",
        "memoria técnica", "contenido mínimo", "contenido de la oferta",
        "objeto del contrato", "alcance del servicio", "contexto y objetivos"
    ]):
        score += 15
    try:
        return int(score)
    except Exception:
        return 0

def _select_relevant_spans(pages: List[str], section_key: str,
                           max_chars: int = LOCAL_CONTEXT_MAX_CHARS, window: int = 1) -> str:
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
        for j, txt in enumerate(pages[:2]):
            if txt:
                selected.append(f"[Pág {j+1}]\n{txt}")
                total += len(txt)
                if total >= max_chars:
                    break

    return "\n\n".join(selected)

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
    _log_event(section_key, "local_fallback", schema_hint + "\n\n[CONTEXTO]\n<<<…>>>", text, model)
    return _json_loads_robust(text)

def _synthesis_fill_section(section_key: str, model: str, temperature: float):
    """Síntesis garantizada para índice técnico / riesgos si no hay literal claro."""
    docs = st.session_state.get("fs_local_docs", [])
    if not docs:
        return None

    snippets = []
    for d in docs:
        pages = d["pages"]
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
            "'indice_propuesto' (NO vacío con >=10 apartados y subapartados), 'trazabilidad' (opcional), "
            "'referencias_paginas', 'evidencias', 'discrepancias'."
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
    _log_event(section_key, "synthesis", schema_hint + "\n\n[CONTEXTO]\n<<<…>>>", text, model)
    return _json_loads_robust(text)

# -----------------------------------------------------------------------------------
# Detección de “resultado vacío”
# -----------------------------------------------------------------------------------
REQUIRED_NONEMPTY = {
    "objetivos_contexto": ["resumen_servicios", "objetivos", "alcance"],
    "servicios": ["servicios_detalle", "resumen_servicios"],
    "importe": ["importe_total", "importes_detalle"],
    "criterios_valoracion": ["criterios_valoracion"],
    "indice_tecnico": ["indice_propuesto"],
    "riesgos_exclusiones": ["riesgos_y_dudas", "exclusiones", "matriz_riesgos"],
    "solvencia": ["solvencia"],
    "formato_oferta": ["formato_esperado", "requisitos_presentacion", "requisitos_archivo", "estructura_documental"],
}

def _is_effectively_empty(section_key: str, data: Dict[str, Any]) -> bool:
    if not isinstance(data, dict) or not data:
        return True
    keys = REQUIRED_NONEMPTY.get(section_key, [])
    for k in keys:
        v = data.get(k)
        if isinstance(v, str) and v.strip():
            return False
        if isinstance(v, (int, float)) and v is not None:
            return False
        if isinstance(v, (list, dict)) and len(v) > 0:
            return False
    return True

def _dedupe_sorted_pages(pages: List[int]) -> List[int]:
    try:
        return sorted(set(int(p) for p in pages if p is not None))
    except Exception:
        return pages or []

# -----------------------------------------------------------------------------------
# Forzar JSON si el modelo devolvió texto no parseable
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
    _log_event(section_key or "desconocida", "jsonify", "[NORMALIZADOR JSON]", text, model)
    return _json_loads_robust(text)

# -----------------------------------------------------------------------------------
# Ejecución de sección (PDF completo → local → síntesis)
# -----------------------------------------------------------------------------------
def run_section(section_key: str, model: str, temperature: float, max_chars: int, file_ids: List[str]) -> Tuple[Dict[str, Any], str]:
    # 1) input_file
    try:
        data = _file_input_section_call(
            user_prompt=_build_prompt_with_hints(section_key),
            model=model,
            temperature=temperature,
            file_ids=file_ids,
            section_key=section_key,
        )
        if _is_effectively_empty(section_key, data):
            raise RuntimeError("input_file devolvió salida vacía para esta sección.")
        if "referencias_paginas" in data:
            data["referencias_paginas"] = _dedupe_sorted_pages(data.get("referencias_paginas") or [])
        return data, "input_file"
    except Exception:
        # 2) Local
        data = _local_singlecall_section(section_key, model=model, temperature=temperature, max_chars=max_chars)

        # 3) Síntesis garantizada para índice/risks si sigue vacío
        if section_key in {"indice_tecnico", "riesgos_exclusiones"} and _is_effectively_empty(section_key, data):
            synth = _synthesis_fill_section(section_key, model=model, temperature=temperature)
            if isinstance(synth, dict):
                data = synth

        # Semilla mínima si aún vacío (índice propuesto estándar)
        if section_key == "indice_tecnico" and _is_effectively_empty(section_key, data):
            data = {
                "indice_respuesta_tecnica": [],
                "indice_propuesto": [
                    {"titulo": "Propuesta de valor", "descripcion": None, "subapartados": []},
                    {"titulo": "Contexto y objetivos", "descripcion": None, "subapartados": []},
                    {"titulo": "Alcance y supuestos", "descripcion": None, "subapartados": []},
                    {"titulo": "Metodología y governance", "descripcion": None, "subapartados": []},
                    {"titulo": "Plan de proyecto (hitos/plazos)", "descripcion": None, "subapartados": []},
                    {"titulo": "Recursos y organización", "descripcion": None, "subapartados": []},
                    {"titulo": "Gestión de calidad y SLAs", "descripcion": None, "subapartados": []},
                    {"titulo": "Gestión de riesgos y continuidad", "descripcion": None, "subapartados": []},
                    {"titulo": "Ciberseguridad y cumplimiento", "descripcion": None, "subapartados": []},
                    {"titulo": "Sostenibilidad y accesibilidad", "descripcion": None, "subapartados": []},
                    {"titulo": "Anexos", "descripcion": None, "subapartados": []},
                ],
                "trazabilidad": [],
                "referencias_paginas": [],
                "evidencias": [],
                "discrepancias": ["Índice literal no hallado; se propone estructura estándar alineada al pliego."],
            }

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
# Sidebar (modelo + temperatura)
# -----------------------------------------------------------------------------------
def sidebar_config() -> Tuple[str, float]:
    with st.sidebar:
        st.header("Configuración")
        if OPENAI_MODEL_DEFAULT in AVAILABLE_MODELS:
            idx = AVAILABLE_MODELS.index(OPENAI_MODEL_DEFAULT)
        else:
            idx = 0
        model = st.selectbox("Modelo OpenAI", AVAILABLE_MODELS, index=idx)
        temperature = st.slider(
            "Temperatura",
            min_value=0.0, max_value=1.0, value=DEFAULT_TEMPERATURE, step=0.05,
            help=(
                "Controla la aleatoriedad de la salida.\n\n"
                "- **Baja (0.0–0.3)**: más determinista/estable (recomendado para pliegos).\n"
                "- **Media (0.4–0.7)**: equilibrio entre variedad y consistencia.\n"
                "- **Alta (0.8–1.0)**: más creativa/variada, pero menos estable."
            ),
        )
        st.caption("Para licitaciones, usar baja (≈0.1–0.3) mejora consistencia y pegado al texto.")
    return model, float(temperature)

# -----------------------------------------------------------------------------------
# Render de resultados (UX – una sola vista, sin JSON visible)
# -----------------------------------------------------------------------------------
def _badge(text: str):
    st.markdown(
        f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;"
        f"background:#eef;border:1px solid #ccd;color:#334;font-size:12px'>{text}</span>",
        unsafe_allow_html=True
    )

def _render_index_block(items: List[Dict[str, Any]]):
    """Muestra título + descripción + subapartados (mejora UX índice técnico)."""
    if not items:
        st.info("Sin contenido.")
        return
    for i, s in enumerate(items, start=1):
        titulo = s.get("titulo") or f"Sección {i}"
        desc = s.get("descripcion")
        subs = s.get("subapartados") or []
        st.markdown(f"- **{titulo}**")
        if desc:
            st.caption(desc)
        if subs:
            st.write("\n".join([f"  • {x}" for x in subs]))

def render_full_view(fs_sections: Dict[str, Any]):
    oc = fs_sections.get("objetivos_contexto", {})
    objetivos = oc.get("objetivos") or []
    im = fs_sections.get("importe", {})
    imp_total = im.get("importe_total")
    moneda = im.get("moneda") or "EUR"
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
        evs = oc.get("evidencias") or []
        disc = oc.get("discrepancias") or []
        if evs or disc:
            colA, colB = st.columns(2)
            if evs:
                with colA:
                    st.caption("Evidencias")
                    for e in evs[:4]:
                        _badge(f"Pág {e.get('pagina')}")
                        st.write(f"“{(e.get('cita') or '')[:180]}”")
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
            try:
                import pandas as pd  # type: ignore
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
            except Exception:
                for d in detalle:
                    st.write(f"- **{d.get('nombre')}** — {d.get('descripcion') or ''}")
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
            try:
                import pandas as pd  # type: ignore
                st.dataframe(pd.DataFrame(det), use_container_width=True, hide_index=True)
            except Exception:
                for x in det:
                    st.write(f"- {x}")
        disc = im.get("discrepancias") or []
        if disc:
            st.caption("Discrepancias")
            st.write("\n".join([f"- {d}" for d in disc]))

    # Criterios de valoración
    cv_all = fs_sections.get("criterios_valoracion", {})
    cv = cv_all.get("criterios_valoracion") or []
    with st.expander("📊 Criterios de valoración", expanded=False):
        if cv:
            try:
                import pandas as pd  # type: ignore
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
            except Exception:
                for c in cv:
                    st.write(f"- {c.get('nombre')} (peso: {c.get('peso_max')} {c.get('tipo')})")
            dmp = cv_all.get("criterios_desempate") or []
            if dmp:
                st.markdown("**Criterios de desempate:**")
                st.write("\n".join([f"- {x}" for x in dmp]))
        else:
            st.info("No se encontraron criterios explícitos.")

    # Índice técnico (mejor render: título + descripción + subapartados)
    it = fs_sections.get("indice_tecnico", {})
    with st.expander("🗂️ Índice de la respuesta técnica", expanded=False):
        col1, col2 = st.columns(2)
        req = it.get("indice_respuesta_tecnica") or []
        prop = it.get("indice_propuesto") or []
        with col1:
            st.markdown("**Índice solicitado (literal)**")
            _render_index_block(req)
        with col2:
            st.markdown("**Índice propuesto (accionable)**")
            _render_index_block(prop)
        tr = it.get("trazabilidad") or []
        if tr:
            st.caption("Trazabilidad propuesto → solicitado")
            for t in tr[:12]:
                st.write(f"- **{t.get('propuesto')}** → {t.get('solicitado_match') or '—'}")
        disc = it.get("discrepancias") or []
        if disc:
            st.caption("Discrepancias")
            st.write("\n".join([f"- {d}" for d in disc]))

    # Formato y entrega de la oferta
    fmt = fs_sections.get("formato_oferta", {})
    with st.expander("🧾 Formato y entrega de la oferta", expanded=False):
        st.markdown(f"**Formato esperado:** {fmt.get('formato_esperado') or '—'}")
        lp = fmt.get("longitud_paginas")
        st.write(f"- **Longitud (pág.)**: {lp if isinstance(lp, (int, float)) else '—'}")
        tip = fmt.get("tipografia") or {}
        st.write(f"- **Tipografía**: {tip.get('familia') or '—'} / **Tamaño mínimo**: {tip.get('tamano_min') or '—'} / "
                 f"**Interlineado**: {tip.get('interlineado') or '—'} / **Márgenes**: {tip.get('margenes') or '—'}")
        est = fmt.get("estructura_documental") or []
        if est:
            st.markdown("**Estructura documental requerida/propuesta:**")
            st.write("\n".join([f"- {x.get('titulo')}" for x in est if x.get("titulo")]))
        rp = fmt.get("requisitos_presentacion") or []
        if rp:
            st.markdown("**Requisitos de presentación:**")
            st.write("\n".join([f"- {x}" for x in rp]))
        ra = fmt.get("requisitos_archivo") or {}
        st.write(f"- **Formatos permitidos**: {', '.join(ra.get('formatos_permitidos') or []) or '—'}")
        st.write(f"- **Tamaño máx (MB)**: {ra.get('tamano_max_mb') if ra.get('tamano_max_mb') is not None else '—'}")
        st.write(f"- **Firma digital requerida**: {ra.get('firma_digital_requerida') if ra.get('firma_digital_requerida') is not None else '—'}")
        st.write(f"- **Firmado por**: {ra.get('firmado_por') or '—'}")
        st.write(f"- **Idioma**: {fmt.get('idioma') or '—'}")
        st.write(f"- **Copias**: {fmt.get('copias') if fmt.get('copias') is not None else '—'}")
        ent = fmt.get("entrega") or {}
        st.write(f"- **Canal de entrega**: {ent.get('canal') or '—'}")
        st.write(f"- **Plazo/Fecha/Hora**: {ent.get('plazo') or '—'} / **Zona horaria**: {ent.get('zona_horaria') or '—'}")
        if ent.get("instrucciones"):
            st.markdown("**Instrucciones de entrega:**")
            st.write("\n".join([f"- {x}" for x in ent.get("instrucciones") or []]))
        pag = fmt.get("paginacion") or {}
        st.write(f"- **Paginación requerida**: {pag.get('requerida') if pag.get('requerida') is not None else '—'} "
                 f"/ **Formato**: {pag.get('formato') or '—'}")
        if fmt.get("etiquetado"):
            st.markdown("**Etiquetado/Rotulación:**")
            st.write("\n".join([f"- {x}" for x in (fmt.get("etiquetado") or [])]))
        if fmt.get("anexos_obligatorios"):
            st.markdown("**Anexos obligatorios:**")
            st.write("\n".join([f"- {x}" for x in (fmt.get("anexos_obligatorios") or [])]))
        if fmt.get("confidencialidad"):
            st.markdown(f"**Confidencialidad:** {fmt.get('confidencialidad')}")
        if fmt.get("discrepancias"):
            st.caption("Discrepancias")
            st.write("\n".join([f"- {d}" for d in (fmt.get("discrepancias") or [])]))

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
            try:
                import pandas as pd  # type: ignore
                st.caption("Matriz de riesgos (PxI)")
                st.dataframe(pd.DataFrame(mrx), use_container_width=True, hide_index=True)
            except Exception:
                for r in mrx:
                    st.write(f"- {r}")
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
            try:
                import pandas as pd  # type: ignore
                st.caption("Acreditación (cómo se demuestra)")
                st.dataframe(pd.DataFrame(acr), use_container_width=True, hide_index=True)
            except Exception:
                for a in acr:
                    st.write(f"- {a}")

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
    fmt = fs_sections.get("formato_oferta", {})
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
    moneda = im.get("moneda") or "EUR"
    add(f"- **Importe total**: {f'{imp_total:.2f} {moneda}' if isinstance(imp_total, (int,float)) else '—'}")
    add(f"- **IVA incluido**: {im.get('iva_incluido') if im.get('iva_incluido') is not None else '—'}")
    add(f"- **Tipo IVA**: {im.get('tipo_iva') if im.get('tipo_iva') is not None else '—'}")
    for d in (im.get("importes_detalle") or []):
        add(f"  - {d.get('concepto') or '—'}: {d.get('importe')} {d.get('moneda') or moneda} ({d.get('observaciones') or ''})")

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
        for s in req:
            add(f"- {s.get('titulo')}{': ' + (s.get('descripcion') or '') if s.get('descripcion') else ''}")
            for sub in s.get("subapartados") or []:
                add(f"  - {sub}")
    else:
        add("- —")
    add("\n**Propuesto**:")
    if prop:
        for s in prop:
            add(f"- {s.get('titulo')}{': ' + (s.get('descripcion') or '') if s.get('descripcion') else ''}")
            for sub in s.get("subapartados") or []:
                add(f"  - {sub}")
    else:
        add("- —")

    add("\n## Formato y entrega de la oferta")
    add(f"- **Formato esperado**: {fmt.get('formato_esperado') or '—'}")
    add(f"- **Longitud (pág.)**: {fmt.get('longitud_paginas') if fmt.get('longitud_paginas') is not None else '—'}")
    tip = fmt.get("tipografia") or {}
    add(f"- **Tipografía**: {tip.get('familia') or '—'}; tamaño mín.: {tip.get('tamano_min') or '—'}; "
        f"interlineado: {tip.get('interlineado') or '—'}; márgenes: {tip.get('margenes') or '—'}")
    if fmt.get("estructura_documental"):
        add("**Estructura documental:**")
        for x in fmt.get("estructura_documental") or []:
            add(f"- {x.get('titulo')}")
    if fmt.get("requisitos_presentacion"):
        add("**Requisitos de presentación:**")
        for x in fmt.get("requisitos_presentacion") or []:
            add(f"- {x}")
    ra = fmt.get("requisitos_archivo") or {}
    add(f"- **Formatos permitidos**: {', '.join(ra.get('formatos_permitidos') or []) or '—'}")
    add(f"- **Tamaño máx (MB)**: {ra.get('tamano_max_mb') if ra.get('tamano_max_mb') is not None else '—'}")
    add(f"- **Firma digital requerida**: {ra.get('firma_digital_requerida') if ra.get('firma_digital_requerida') is not None else '—'}")
    add(f"- **Firmado por**: {ra.get('firmado_por') or '—'}")
    add(f"- **Idioma**: {fmt.get('idioma') or '—'}")
    add(f"- **Copias**: {fmt.get('copias') if fmt.get('copias') is not None else '—'}")
    ent = fmt.get("entrega") or {}
    add(f"- **Canal de entrega**: {ent.get('canal') or '—'}; **Plazo/Fecha/Hora**: {ent.get('plazo') or '—'}; "
        f"**Zona horaria**: {ent.get('zona_horaria') or '—'}")
    if ent.get("instrucciones"):
        add("**Instrucciones de entrega:**")
        for x in ent.get("instrucciones") or []:
            add(f"- {x}")
    pag = fmt.get("paginacion") or {}
    add(f"- **Paginación**: requerida={pag.get('requerida') if pag.get('requerida') is not None else '—'}; "
        f"formato={pag.get('formato') or '—'}")
    if fmt.get("etiquetado"):
        add("**Etiquetado/Rotulación:**")
        for x in fmt.get("etiquetado") or []:
            add(f"- {x}")
    if fmt.get("anexos_obligatorios"):
        add("**Anexos obligatorios:**")
        for x in fmt.get("anexos_obligatorios") or []:
            add(f"- {x}")
    if fmt.get("confidencialidad"):
        add(f"- **Confidencialidad**: {fmt.get('confidencialidad')}")

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
            add(f"- {a.get('requisito')}: doc={a.get('documento_necesario')}, norma={a.get('norma_referencia')}, umbral={a.get('umbral')}")

    return "\n".join(out)

# -----------------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------------
def sidebar_and_header():
    login_gate()
    model, temperature = sidebar_config()
    st.title(APP_TITLE)
    st.caption(f"Analizador de pliegos (GPT-4o / 4o-mini). Temperatura actual: **{temperature:.2f}**.")
    return model, temperature

def main():
    model, temperature = sidebar_and_header()

    # Carga de PDFs
    files = st.file_uploader("Sube uno o varios PDFs del pliego", type=["pdf"], accept_multiple_files=True)
    if not files:
        st.info("Sube al menos un PDF para comenzar.")
        st.stop()

    # Tabulador: Análisis y Registro
    tab1, tab2 = st.tabs(["Análisis", "Registro (Prompts/Respuestas)"])

    with tab1:
        # Preparación de PDFs (OpenAI + local)
        if "fs_file_ids" not in st.session_state or st.button("Reindexar PDFs en OpenAI", use_container_width=True):
            st.session_state.pop("fs_sections", None)

            uploaded = [{"name": f.name, "bytes": f.read()} for f in files]

            with st.spinner("Subiendo PDF(s) a OpenAI…"):
                file_ids = []
                for u in uploaded:
                    up = _oai.files.create(
                        file=(u["name"], u["bytes"], "application/pdf"),
                        purpose="assistants",
                    )
                    file_ids.append(up.id)

            local_docs, char_stats = [], []
            for u in uploaded:
                parsed = parse_pdf_cached(u["name"], u["bytes"])
                local_docs.append({"name": u["name"], "pages": parsed["pages"]})
                char_stats.append((u["name"], len(parsed["pages"]), parsed["total_chars"]))

            st.session_state["fs_file_ids"] = file_ids
            st.session_state["fs_local_docs"] = local_docs
            st.session_state["char_stats"] = char_stats

            st.success("PDF(s) preparados (OpenAI + texto local).")

        # Diagnóstico
        with st.expander("Diagnóstico de extracción", expanded=False):
            for name, npages, nchar in st.session_state.get("char_stats", []):
                st.write(f"- **{name}**: {npages} páginas, {nchar} caracteres")
                if nchar < 1000:
                    st.warning(f"{name}: muy poco texto (posible escaneado sin OCR).")

        # Estado de ejecución
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
            st.button("Formato y entrega de la oferta",  use_container_width=True, disabled=dis,
                      on_click=_start_job, kwargs={"section": "formato_oferta"})
            st.button("🔎 Análisis Completo", type="primary", use_container_width=True, disabled=dis,
                      on_click=_start_job, kwargs={"do_all": True})

        # Ejecución (con status)
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
                                    temperature=temperature,
                                    max_chars=LOCAL_CONTEXT_MAX_CHARS,
                                    file_ids=st.session_state["fs_file_ids"],
                                )
                                st.session_state["fs_sections"][k] = data
                                status.write(f"✓ {spec['titulo']} ({mode})")
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
                            temperature=temperature,
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

        # Resultados
        st.subheader("Resultados")
        fs_sections = st.session_state.get("fs_sections", {})
        if not fs_sections:
            st.info("Aún no hay resultados. Pulsa **Análisis Completo** o ejecuta alguna sección.")
        else:
            render_full_view(fs_sections)
            # Descargas
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

    # Registro de prompts/respuestas (incluye Formato y Entrega)
    with tab2:
        st.caption("Vista de auditoría: prompt enviado y respuesta del modelo por sección/fase.")
        _log_init()
        st.button("🔄 Reinicializar/Normalizar listado de secciones", on_click=_log_init)

        # Mostrar en orden fijo (incluye formato_oferta)
        ordered = list(SECTION_SPECS.keys())
        for k in ordered:
            spec = SECTION_SPECS.get(k, {"titulo": k})
            v = st.session_state["logs"].get(k, [])
            with st.expander(f"🧭 {spec['titulo']} ({k}) — {len(v)} eventos", expanded=False):
                if not v:
                    st.info("Sin logs para esta sección todavía.")
                    continue
                for i, rec in enumerate(v, start=1):
                    st.markdown(f"**#{i} – Fase:** `{rec.get('fase')}` | **Modelo:** `{rec.get('model')}`")
                    st.markdown("**Prompt enviado:**")
                    st.code(rec.get("prompt") or "", language="markdown")
                    st.markdown("**Respuesta (cruda):**")
                    st.code(rec.get("response") or "", language="json")
                    st.divider()

# -----------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
