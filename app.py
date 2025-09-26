
# -*- coding: utf-8 -*-
import os
import io
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple

import streamlit as st

# ---- OpenAI SDK ----
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---- PDF parsing & OCR ----
from PyPDF2 import PdfReader
try:
    import pdfium  # pypdfium2
    from PIL import Image
    import pytesseract
    _HAS_OCR_DEPS = True
except Exception:
    _HAS_OCR_DEPS = False

# --------------------------
# Page & Theme
# --------------------------
st.set_page_config(page_title="RFP Analyzer – Resumen Ejecutivo", layout="wide")

def _apply_theme(dark: bool):
    # Basic light/dark styling using CSS injection
    if dark:
        st.markdown("""
        <style>
        .stApp { background-color: #0e1117; color: #e1e1e1; }
        .block-container { padding-top: 1.2rem; }
        header[data-testid="stHeader"] { background: rgba(0,0,0,0); }
        /* Text */
        h1, h2, h3, h4, h5, h6, label, p, span { color: #e1e1e1; }
        /* Metrics */
        div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"], div[data-testid="stMetricDelta"] {
            color: #e1e1e1 !important;
        }
        /* Inputs */
        .stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] div {
            color: #e1e1e1 !important;
        }
        /* Code blocks */
        pre, code { background: #161a22 !important; color: #e1e1e1 !important; }
        /* Dividers */
        hr { border-color: #30363d; }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp { background-color: #ffffff; color: #111111; }
        header[data-testid="stHeader"] { background: rgba(255,255,255,0.5); }
        div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"], div[data-testid="stMetricDelta"] {
            color: #111111 !important;
        }
        pre, code { background: #f6f8fa !important; color: #111111 !important; }
        </style>
        """, unsafe_allow_html=True)

# --------------------------
# Data structures & helpers
# --------------------------
@dataclass
class PDFDoc:
    name: str
    pages_text: List[str]
    full_text: str

_OCR_ENABLED = True  # Default

KEYWORDS_EXEC = {
    # duración / vigencia
    "duración del contrato": 6,
    "plazo de ejecución": 6,
    "vigencia": 4,
    "periodo de ejecución": 5,
    "periodo de prestación": 4,
    "prórrogas": 3,
    # oferta técnica (plazo)
    "fecha límite de presentación": 6,
    "plazo de presentación de ofertas": 6,
    "fecha y hora de presentación": 6,
    "presentación de proposiciones": 5,
    "fecha tope de entrega": 5,
    "sobre electrónico": 3,
    "sobres electrónicos": 3,
    "licitación electrónica": 3,
    "perfil del contratante": 3,
    # encabezados típicos
    "presentación de ofertas": 4,
    "criterios de adjudicación": 1,
}

def _score_text(text: str, weights: Dict[str, float]) -> float:
    if not text: return 0.0
    t = text.lower()
    score = 0.0
    for kw, w in (weights or {}).items():
        try:
            ww = float(w) if isinstance(w, (int, float)) else 1.0
        except Exception:
            ww = 1.0
        score += t.count(kw.lower()) * ww
    if any(h in t for h in [
        "presentación de ofertas", "índice", "indice",
        "objeto del contrato", "alcance del servicio",
        "duración del contrato", "plazo de ejecución", "vigencia"
    ]):
        score += 8
    return score

def _ocr_pdf_bytes(content_bytes: bytes) -> List[str]:
    """OCR fallback per page using pypdfium2 + pytesseract (spa+eng)."""
    pages_text: List[str] = []
    try:
        pdf = pdfium.PdfDocument(content_bytes)
    except Exception:
        return pages_text
    for i in range(len(pdf)):
        try:
            page = pdf.get_page(i)
            pil = page.render(scale=2.0).to_pil()
            txt = pytesseract.image_to_string(pil, lang="spa+eng")
            pages_text.append(txt or "")
        except Exception:
            pages_text.append("")
    return pages_text

def extract_pdf(file: io.BytesIO, filename: str) -> PDFDoc:
    b = file.read()
    # First try PyPDF2 text
    pages: List[str] = []
    try:
        reader = PdfReader(io.BytesIO(b))
        for p in reader.pages:
            try:
                t = p.extract_text() or ""
            except Exception:
                t = ""
            pages.append(t)
    except Exception:
        pages = []

    total_chars = sum(len(x) for x in pages)
    if _OCR_ENABLED and (total_chars < 1500) and _HAS_OCR_DEPS:
        # OCR fallback only if it improves content
        pages_ocr = _ocr_pdf_bytes(b)
        if sum(len(x) for x in pages_ocr) > total_chars:
            pages = pages_ocr

    return PDFDoc(name=filename, pages_text=pages, full_text="\\n".join(pages))

def select_relevant_spans(docs: List[PDFDoc], k_pages: int = 16) -> Tuple[str, List[Tuple[str,int,str]]]:
    """
    Score pages by executive keywords and return a concatenated context (bounded)
    plus a list of (docname, page_idx, snippet).
    """
    ranked: List[Tuple[float, str, int, str]] = []
    for d in docs:
        for i, t in enumerate(d.pages_text):
            sc = _score_text(t, KEYWORDS_EXEC)
            if sc > 0:
                snippet = (t[:800] + " ...") if len(t) > 800 else t
                ranked.append((sc, d.name, i, snippet))
    ranked.sort(key=lambda x: x[0], reverse=True)
    top = ranked[:k_pages]
    spans = []
    ctx_parts = []
    for _, dn, i, sn in top:
        spans.append((dn, i+1, sn))  # page number humanized
        ctx_parts.append(f"[{dn} | pág {i+1}]\\n{sn}")
    context = "\\n\\n".join(ctx_parts)
    return context, spans

def call_openai_extract(api_key: str, model: str, context: str) -> Dict:
    if OpenAI is None:
        raise RuntimeError("La librería 'openai' no está instalada en el entorno.")
    client = OpenAI(api_key=api_key)

    system = (
        "Eres un analista experto en licitaciones públicas españolas. "
        "Extrae del CONTEXTO los dos campos requeridos, sin inventar."
    )
    user = (
        "CONSIDERA EXCLUSIVAMENTE EL CONTEXTO SIGUIENTE (fragmentos de pliegos):\\n\\n"
        f"{context}\\n\\n"
        "Devuelve JSON con el esquema EXACTO:\\n"
        "{\\n"
        '  "plazo_contratacion": str|null,\\n'
        '  "fecha_entrega_oferta": str|null,\\n'
        '  "evidencias": [{"pagina": int, "cita": str}],\\n'
        '  "referencias_paginas": [int],\\n'
        '  "discrepancias": [str]\\n'
        "}\\n"
        "- 'plazo_contratacion': duración/vigencia del servicio, incluyendo prórrogas si se especifican.\\n"
        "- 'fecha_entrega_oferta': fecha (y hora si aparece) de la entrega/presentación de la oferta técnica.\\n"
        "- Normaliza fecha a ISO 8601 si es inequívoca; si no, deja null y explica en 'discrepancias'.\\n"
        "- Llena 'evidencias' con citas breves y número de página (si no se conoce, omite o usa 0)."
    )

    rsp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    txt = rsp.choices[0].message.content or "{}"
    try:
        data = json.loads(txt)
    except Exception:
        txt2 = txt.strip("` \\n")
        data = json.loads(txt2) if txt2.startswith("{") else {}
    return data

# --------------------------
# UI
# --------------------------
# Top bar with title and theme toggle (right aligned using columns)
col_title, col_spacer, col_toggle = st.columns([0.82, 0.06, 0.12])
with col_title:
    st.title("Análisis de Pliegos – Resumen Ejecutivo")
with col_toggle:
    # Toggle shows 🌙 when dark, ☀️ when light
    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = False
    dark_mode = st.toggle("🌙 / ☀️", value=st.session_state["dark_mode"],
                          help="Cambiar apariencia: oscuro / claro", key="dark_mode")
_apply_theme(st.session_state["dark_mode"])

# Sidebar config (no API key or uploader here)
with st.sidebar:
    st.header("Configuración")
    model = st.selectbox("Modelo", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"], index=0)
    ocr_enabled = st.toggle("Activar OCR (pypdfium2 + Tesseract)", value=_OCR_ENABLED,
                            help="Si el PDF es escaneado y PyPDF2 no extrae texto, se intentará OCR.")
    _OCR_ENABLED = ocr_enabled
    st.caption("La clave de OpenAI se toma de **st.secrets**.")

# Main area: file uploader and analyze button
st.markdown("#### Sube los PDFs del pliego")
up_files = st.file_uploader("PDF(s) del pliego (Administrativo/Técnico)", type=["pdf"], accept_multiple_files=True)
analyze = st.button("Analizar resumen ejecutivo", type="primary", disabled=not up_files)

# Read API key from Streamlit secrets
def _load_openai_key_from_secrets() -> str:
    # Accept several common layouts
    # 1) st.secrets["openai"]["api_key"]
    # 2) st.secrets["OPENAI_API_KEY"]
    # 3) st.secrets["openai_api_key"]
    try:
        if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
            return st.secrets["openai"]["api_key"]
    except Exception:
        pass
    for k in ("OPENAI_API_KEY", "openai_api_key"):
        try:
            if k in st.secrets:
                return st.secrets[k]
        except Exception:
            pass
    return ""

if analyze:
    api_key = _load_openai_key_from_secrets()
    if not api_key:
        st.error("No se encontró la clave en **st.secrets**. Añádela en `.streamlit/secrets.toml` como `OPENAI_API_KEY='sk-...'` o `openai.api_key='...'`.")
        st.stop()

    # Extract PDFs
    docs: List[PDFDoc] = []
    for f in up_files:
        try:
            docs.append(extract_pdf(io.BytesIO(f.read()), f.name))
        except Exception as e:
            st.warning(f"No se pudo procesar {f.name}: {e}")

    if not docs:
        st.error("No se pudo extraer ningún texto de los PDFs.")
        st.stop()

    # Select relevant pages
    context, spans = select_relevant_spans(docs, k_pages=18)

    with st.expander("Contexto relevante enviado al modelo", expanded=False):
        st.code(context[:30000])

    # OpenAI call
    try:
        data = call_openai_extract(api_key, model, context)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Metrics header
    st.markdown("### Resumen ejecutivo")
    c1, c2, c3, c4 = st.columns(4)
    imp_str = "—"  # placeholder; opcional: implementar extractor de importe total
    c1.metric("Importe total", imp_str)
    c2.metric("Duración del contrato", data.get("plazo_contratacion") or "—")
    c3.metric("Fecha límite oferta técnica", data.get("fecha_entrega_oferta") or "—")
    c4.metric("Páginas relevantes", len(spans))

    st.divider()

    # Details
    with st.expander("⏱️ Resumen ejecutivo (plazos clave)", expanded=True):
        st.write(f"- **Duración del contrato**: {data.get('plazo_contratacion') or '—'}")
        st.write(f"- **Fecha límite oferta técnica**: {data.get('fecha_entrega_oferta') or '—'}")
        evs = data.get("evidencias") or []
        if evs:
            st.caption("Evidencias")
            for e in evs[:10]:
                pg = e.get("pagina") or 0
                cita = (e.get("cita") or "")[:300]
                st.write(f"• pág {pg}: “{cita}”")
        disc = data.get("discrepancias") or []
        if disc:
            st.caption("Discrepancias")
            for d in disc:
                st.write(f"• {d}")
else:
    st.info("Sube tus PDFs en el área principal y pulsa **Analizar resumen ejecutivo** para extraer plazos clave.")
