
# -*- coding: utf-8 -*-
import os
import io
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

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
st.set_page_config(page_title="RFP Analyzer – Pliegos", layout="wide")

def _apply_theme(dark: bool):
    if dark:
        st.markdown("""
        <style>
        .stApp { background-color: #0e1117; color: #e1e1e1; }
        .block-container { padding-top: 1.2rem; }
        header[data-testid="stHeader"] { background: rgba(0,0,0,0); }
        h1, h2, h3, h4, h5, h6, label, p, span { color: #e1e1e1; }
        div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"], div[data-testid="stMetricDelta"] {
            color: #e1e1e1 !important;
        }
        pre, code { background: #161a22 !important; color: #e1e1e1 !important; }
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

# ---- Section keywords for retrieval ----
SECTION_KEYWORDS: Dict[str, Dict[str, float]] = {
    "resumen_ejecutivo": {
        "duración del contrato": 6, "plazo de ejecución": 6, "vigencia": 4,
        "periodo de ejecución": 5, "periodo de prestación": 4, "prórrogas": 3,
        "fecha límite de presentación": 6, "plazo de presentación de ofertas": 6,
        "fecha y hora de presentación": 6, "presentación de proposiciones": 5,
        "fecha tope de entrega": 5, "sobre electrónico": 3, "sobres electrónicos": 3,
        "licitación electrónica": 3, "perfil del contratante": 3, "presentación de ofertas": 4
    },
    "objetivos_contexto": {
        "objeto": 5, "objeto del contrato": 6, "objetivo": 5, "alcance": 5, "contexto": 4,
        "finalidad": 4, "justificación": 3, "marco": 2
    },
    "servicios": {
        "servicios": 5, "alcance": 4, "trabajos": 4, "tareas": 4, "actividades": 4,
        "entregables": 5, "responsabilidades": 3, "hitos": 3
    },
    "solvencia": {
        "solvencia técnica": 6, "solvencia económica": 6, "medios personales": 4,
        "experiencia": 4, "certificaciones": 3, "clasificación": 3, "requisitos mínimos": 5
    },
    "criterios_valoracion": {
        "criterios de adjudicación": 6, "criterios de valoración": 6, "baremo": 5,
        "puntuación": 5, "ponderación": 5, "subcriterios": 4, "fórmula": 4,
        "evaluación": 3, "criterios automáticos": 3, "criterios subjetivos": 3
    },
    "formato_oferta": {
        "formato de la oferta": 6, "contenido de la oferta": 6, "memoria técnica": 6,
        "estructura": 4, "extensión": 4, "índice": 4, "presentación de ofertas": 3,
        "soporte electrónico": 3, "plataforma de contratación": 3
    },
    "riesgos_exclusiones": {
        "penalizaciones": 5, "exclusiones": 5, "riesgos": 4, "causas de exclusión": 5,
        "incumplimientos": 4, "resolución del contrato": 3
    }
}

# ---- Section prompts/specs ----
SECTION_SPECS: Dict[str, Dict[str, str]] = {
    "resumen_ejecutivo": {
        "title": "Resumen ejecutivo (plazos clave)",
        "user_prompt": (
            "Del CONTEXTO, extrae dos campos clave:\n"
            "- plazo_contratacion: duración/vigencia/plazo de ejecución (con prórrogas si aplica)\n"
            "- fecha_entrega_oferta: fecha (y hora) límite para la entrega/presentación de la oferta técnica\n"
            "Devuelve JSON EXACTO:\n"
            "{\n"
            '  "plazo_contratacion": str|null,\n'
            '  "fecha_entrega_oferta": str|null,\n'
            '  "evidencias": [{"pagina": int, "cita": str}],\n'
            '  "referencias_paginas": [int],\n'
            '  "discrepancias": [str]\n'
            "}\n"
            "- Normaliza fecha a ISO 8601 si es inequívoca; si no, deja null y explica en 'discrepancias'.\n"
            "- No inventes; usa exclusivamente lo que aparece en el contexto."
        )
    },
    "objetivos_contexto": {
        "title": "Objetivos y Contexto",
        "user_prompt": (
            "Extrae objetivos y el contexto/alcance del contrato. Devuelve JSON con:\n"
            '{ "objetivos": [str], "contexto": [str], "referencias_paginas": [int], "evidencias": [{"pagina": int, "cita": str}] }'
        )
    },
    "servicios": {
        "title": "Servicios solicitados",
        "user_prompt": (
            "Lista servicios/trabajos/actividades y entregables requeridos. JSON:\n"
            '{ "servicios": [str], "entregables": [str], "referencias_paginas": [int], "evidencias": [{"pagina": int, "cita": str}] }'
        )
    },
    "solvencia": {
        "title": "Criterios de solvencia",
        "user_prompt": (
            "Resume solvencia técnica/económica, experiencia mínima, medios, certificaciones. JSON:\n"
            '{ "solvencia_tecnica": [str], "solvencia_economica": [str], "otros": [str], "referencias_paginas": [int], "evidencias": [{"pagina": int, "cita": str}] }'
        )
    },
    "criterios_valoracion": {
        "title": "Criterios de valoración",
        "user_prompt": (
            "Extrae criterios de adjudicación/valoración, con ponderaciones/puntuación, fórmulas y subcriterios. JSON:\n"
            '{ "criterios": [{"nombre": str, "peso": str|null, "tipo": "automatico|subjetivo|null"}], "referencias_paginas": [int], "evidencias": [{"pagina": int, "cita": str}] }'
        )
    },
    "formato_oferta": {
        "title": "Formato de la oferta",
        "user_prompt": (
            "Extrae estructura/índice/longitud de la memoria técnica, requisitos de soporte/entrega. JSON:\n"
            '{ "estructura": [str], "longitudes": [str], "requisitos_entrega": [str], "referencias_paginas": [int], "evidencias": [{"pagina": int, "cita": str}] }'
        )
    },
    "riesgos_exclusiones": {
        "title": "Riesgos y exclusiones",
        "user_prompt": (
            "Extrae penalizaciones/riesgos/exclusiones/causas de exclusión. JSON:\n"
            '{ "riesgos": [str], "exclusiones": [str], "penalizaciones": [str], "referencias_paginas": [int], "evidencias": [{"pagina": int, "cita": str}] }'
        )
    }
}

# --------------------------
# Retrieval & extraction
# --------------------------
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
    # bonus for headings
    if any(h in t for h in ["presentación de ofertas", "índice", "indice", "objeto del contrato", "alcance del servicio",
                            "duración del contrato", "plazo de ejecución", "vigencia", "criterios de adjudicación"]):
        score += 8
    return score

def _ocr_pdf_bytes(content_bytes: bytes) -> List[str]:
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
        pages_ocr = _ocr_pdf_bytes(b)
        if sum(len(x) for x in pages_ocr) > total_chars:
            pages = pages_ocr

    return PDFDoc(name=filename, pages_text=pages, full_text="\\n".join(pages))

def select_relevant_spans_for_section(docs: List[PDFDoc], section_key: str, k_pages: int = 18) -> Tuple[str, List[Tuple[str,int,str]]]:
    weights = SECTION_KEYWORDS.get(section_key, {})
    ranked: List[Tuple[float, str, int, str]] = []
    for d in docs:
        for i, t in enumerate(d.pages_text):
            sc = _score_text(t, weights) if weights else 0.0
            if sc > 0:
                snippet = (t[:800] + " ...") if len(t) > 800 else t
                ranked.append((sc, d.name, i, snippet))
    # fallback: if nothing scored, include first 4 pages across docs
    if not ranked:
        for d in docs:
            for i, t in enumerate(d.pages_text[:4]):
                snippet = (t[:800] + " ...") if len(t) > 800 else t
                ranked.append((0.1, d.name, i, snippet))
    ranked.sort(key=lambda x: x[0], reverse=True)
    top = ranked[:k_pages]
    spans = []
    ctx_parts = []
    for _, dn, i, sn in top:
        spans.append((dn, i+1, sn))
        ctx_parts.append(f"[{dn} | pág {i+1}]\\n{sn}")
    context = "\\n\\n".join(ctx_parts)
    return context, spans

def _load_openai_key_from_secrets() -> str:
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

def call_openai_json(api_key: str, model: str, system: str, user: str) -> Dict:
    if OpenAI is None:
        raise RuntimeError("La librería 'openai' no está instalada.")
    client = OpenAI(api_key=api_key)
    rsp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    txt = rsp.choices[0].message.content or "{}"
    try:
        return json.loads(txt)
    except Exception:
        txt2 = txt.strip("` \\n")
        return json.loads(txt2) if txt2.startswith("{") else {}

def analyze_section(section_key: str, docs: List[PDFDoc], model: str, api_key: str) -> Dict:
    spec = SECTION_SPECS[section_key]
    context, spans = select_relevant_spans_for_section(docs, section_key, k_pages=18)
    system = "Eres un analista experto en licitaciones públicas españolas. Devuelve JSON exacto según lo solicitado."
    user = "CONSIDERA EXCLUSIVAMENTE ESTE CONTEXTO (fragmentos de pliegos):\\n\\n" + context + "\\n\\n" + spec["user_prompt"]
    data = call_openai_json(api_key, model, system, user)
    # Añadir referencias si faltan
    if "referencias_paginas" in spec["user_prompt"] and not data.get("referencias_paginas"):
        data["referencias_paginas"] = [p for _, p, _ in spans][:10]
    return data

# --------------------------
# UI: Top bar with theme toggle
# --------------------------
col_title, col_spacer, col_toggle = st.columns([0.82, 0.06, 0.12])
with col_title:
    st.title("Análisis de Pliegos – Licitaciones")
with col_toggle:
    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = False
    st.session_state["dark_mode"] = st.toggle("🌙 / ☀️", value=st.session_state["dark_mode"],
                                              help="Cambiar apariencia: oscuro / claro", key="dark_mode")
_apply_theme(st.session_state["dark_mode"])

# Sidebar: general config only (no uploader, no api key here)
with st.sidebar:
    st.header("Configuración")
    model = st.selectbox("Modelo", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"], index=0)
    ocr_enabled = st.toggle("Activar OCR (pypdfium2 + Tesseract)", value=_OCR_ENABLED,
                            help="Si PyPDF2 extrae poco texto, intentar OCR.")
    _OCR_ENABLED = ocr_enabled
    st.caption("La clave de OpenAI se toma de **st.secrets**.")

# Main area: uploader and buttons
st.markdown("#### Sube los PDFs del pliego")
up_files = st.file_uploader("PDF(s) del pliego (Administrativo/Técnico)", type=["pdf"], accept_multiple_files=True)

cols = st.columns(6)
btn_exec = cols[0].button("Resumen ejecutivo")
btn_obj = cols[1].button("Objetivos y contexto")
btn_srv = cols[2].button("Servicios solicitados")
btn_sol = cols[3].button("Criterios de solvencia")
btn_cri = cols[4].button("Criterios de valoración")
btn_for = cols[5].button("Formato de la oferta")

cols2 = st.columns(6)
btn_rie = cols2[0].button("Riesgos y exclusiones")
btn_all = cols2[1].button("Análisis completo")

# Ensure session storage for section results
if "sections" not in st.session_state:
    st.session_state["sections"] = {}

def _ensure_docs(files) -> List[PDFDoc]:
    docs: List[PDFDoc] = []
    for f in files or []:
        try:
            docs.append(extract_pdf(io.BytesIO(f.read()), f.name))
        except Exception as e:
            st.warning(f"No se pudo procesar {f.name}: {e}")
    return docs

def _run_and_store(section_key: str, docs: List[PDFDoc], model: str, api_key: str):
    with st.spinner(f"Analizando {SECTION_SPECS[section_key]['title']}..."):
        res = analyze_section(section_key, docs, model, api_key)
    st.session_state["sections"][section_key] = res

def _top_metrics(sections: Dict[str, Dict]):
    st.markdown("### Resumen ejecutivo")
    c1, c2, c3, c4 = st.columns(4)
    imp_str = "—"  # placeholder (puede implementarse extractor de importe)
    c1.metric("Importe total", imp_str)
    rexe = sections.get("resumen_ejecutivo", {}) or {}
    c2.metric("Duración del contrato", rexe.get("plazo_contratacion") or "—")
    c3.metric("Fecha límite oferta técnica", rexe.get("fecha_entrega_oferta") or "—")
    # Mantener el indicador global de progreso
    c4.metric("Secciones completas", sum(1 for k, v in sections.items() if v))
    st.divider()

def _expand_section(title: str, data: Dict):
    with st.expander(title, expanded=False):
        st.json(data)

# Run actions
if any([btn_exec, btn_obj, btn_srv, btn_sol, btn_cri, btn_for, btn_rie, btn_all]):
    api_key = _load_openai_key_from_secrets()
    if not api_key:
        st.error("No se encontró la clave en **st.secrets**. Añádela en `.streamlit/secrets.toml`.")
        st.stop()
    if not up_files:
        st.error("Sube al menos un PDF del pliego.")
        st.stop()
    docs = _ensure_docs(up_files)

    if btn_all:
        for k in ["resumen_ejecutivo","objetivos_contexto","servicios","solvencia","criterios_valoracion","formato_oferta","riesgos_exclusiones"]:
            _run_and_store(k, docs, model, api_key)
    else:
        if btn_exec: _run_and_store("resumen_ejecutivo", docs, model, api_key)
        if btn_obj:  _run_and_store("objetivos_contexto", docs, model, api_key)
        if btn_srv:  _run_and_store("servicios", docs, model, api_key)
        if btn_sol:  _run_and_store("solvencia", docs, model, api_key)
        if btn_cri:  _run_and_store("criterios_valoracion", docs, model, api_key)
        if btn_for:  _run_and_store("formato_oferta", docs, model, api_key)
        if btn_rie:  _run_and_store("riesgos_exclusiones", docs, model, api_key)

# Header metrics (always show, based on last results if present)
_top_metrics(st.session_state["sections"])

# Expanders: show any computed sections
sec = st.session_state["sections"]
if sec.get("resumen_ejecutivo"):
    # show friendly executive section
    with st.expander("⏱️ Resumen ejecutivo (plazos clave)", expanded=True):
        rexe = sec["resumen_ejecutivo"]
        st.write(f"- **Duración del contrato**: {rexe.get('plazo_contratacion') or '—'}")
        st.write(f"- **Fecha límite oferta técnica**: {rexe.get('fecha_entrega_oferta') or '—'}")
        evs = rexe.get("evidencias") or []
        if evs:
            st.caption("Evidencias")
            for e in evs[:10]:
                pg = e.get("pagina") or 0
                cita = (e.get("cita") or "")[:300]
                st.write(f"• pág {pg}: “{cita}”")
        disc = rexe.get("discrepancias") or []
        if disc:
            st.caption("Discrepancias")
            for d in disc:
                st.write(f"• {d}")

for key, spec in SECTION_SPECS.items():
    if key == "resumen_ejecutivo":
        continue
    data = sec.get(key)
    if data:
        _expand_section("🧩 " + spec["title"], data)

# Context viewer (optional)
if sec:
    # Build a small context preview for the last analyzed section
    last_key = list(sec.keys())[-1]
    ctx, spans = select_relevant_spans_for_section([], last_key) if False else ("", [])
    # Not showing context here to avoid re-reading PDFs; kept as placeholder.
