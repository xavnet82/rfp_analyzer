
# -*- coding: utf-8 -*-
import os
import io
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import streamlit as st

# ---- OpenAI SDK (chat.completions for broad compatibility) ----
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
# Helpers
# --------------------------
st.set_page_config(page_title="RFP Analyzer - Resumen Ejecutivo", layout="wide")

@dataclass
class PDFDoc:
    name: str
    pages_text: List[str]
    full_text: str

_OCR_ENABLED = True  # Toggle programático

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
    if not text:
        return 0.0
    t = text.lower()
    score = 0.0
    for kw, w in (weights or {}).items():
        try:
            ww = float(w) if isinstance(w, (int, float)) else 1.0
        except Exception:
            ww = 1.0
        score += t.count(kw.lower()) * ww
    # small bonus for headings
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

    return PDFDoc(name=filename, pages_text=pages, full_text="\n".join(pages))

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
        ctx_parts.append(f"[{dn} | pág {i+1}]\n{sn}")
    context = "\n\n".join(ctx_parts)
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
        "CONSIDERA EXCLUSIVAMENTE EL CONTEXTO SIGUIENTE (fragmentos de pliegos):\n\n"
        f"{context}\n\n"
        "Devuelve JSON con el esquema EXACTO:\n"
        "{\n"
        '  "plazo_contratacion": str|null,\n'
        '  "fecha_entrega_oferta": str|null,\n'
        '  "evidencias": [{"pagina": int, "cita": str}],\n'
        '  "referencias_paginas": [int],\n'
        '  "discrepancias": [str]\n'
        "}\n"
        "- 'plazo_contratacion': duración/vigencia del servicio, incluyendo prórrogas si se especifican.\n"
        "- 'fecha_entrega_oferta': fecha (y hora si aparece) de la entrega/presentación de la oferta técnica.\n"
        "- Normaliza fecha a ISO 8601 si es inequívoca; si no, deja null y explica en 'discrepancias'.\n"
        "- Llena 'evidencias' con citas breves y número de página (si no se conoce, omite o usa 0)."
    )

    try:
        rsp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except Exception as e:
        raise RuntimeError(f"Error llamando a OpenAI: {e}")

    txt = rsp.choices[0].message.content or "{}"
    try:
        data = json.loads(txt)
    except Exception:
        # A veces el modelo envía backticks; intenta limpiar
        txt2 = txt.strip("` \n")
        data = json.loads(txt2) if txt2.startswith("{") else {}
    return data

# --------------------------
# UI
# --------------------------
st.title("Análisis de Pliegos – Resumen Ejecutivo")

with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input("OpenAI API key", type="password", help="Se usa localmente para consultar el modelo.")
    model = st.selectbox("Modelo", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"], index=0)
    ocr_enabled = st.toggle("Activar OCR (pypdfium2 + Tesseract)", value=_OCR_ENABLED,
                            help="Si el PDF es escaneado y PyPDF2 no extrae texto, se intentará OCR.")
    _OCR_ENABLED = ocr_enabled

    st.caption("Sube uno o varios PDFs del pliego (Administrativo/Técnico).")
    up_files = st.file_uploader("PDF(s) del pliego", type=["pdf"], accept_multiple_files=True)

analyze = st.button("Analizar resumen ejecutivo", type="primary", disabled=not up_files)

if analyze:
    if not api_key:
        st.error("Introduce tu OpenAI API key.")
        st.stop()

    docs: List[PDFDoc] = []
    for f in up_files:
        try:
            docs.append(extract_pdf(io.BytesIO(f.read()), f.name))
        except Exception as e:
            st.warning(f"No se pudo procesar {f.name}: {e}")

    if not docs:
        st.error("No se pudo extraer ningún texto de los PDFs.")
        st.stop()

    # Selección de páginas relevantes por keywords
    context, spans = select_relevant_spans(docs, k_pages=18)

    with st.expander("Contexto relevante que se envía al modelo", expanded=False):
        st.code(context[:30000])  # visualización truncada para evitar saturación

    # Llamada a OpenAI para extraer datos del resumen ejecutivo
    try:
        data = call_openai_extract(api_key, model, context)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Métricas cabecera
    st.markdown("### Resumen ejecutivo")
    c1, c2, c3, c4 = st.columns(4)
    imp_str = "—"  # si quieres, puedes detectar importe total en otra función
    c1.metric("Importe total", imp_str)
    c2.metric("Duración del contrato", data.get("plazo_contratacion") or "—")
    c3.metric("Fecha límite oferta técnica", data.get("fecha_entrega_oferta") or "—")
    # "Secciones completas" aquí no aplica; mostramos nº de páginas relevantes encontradas
    c4.metric("Páginas relevantes", len(spans))

    st.divider()

    # Panel de evidencias y discrepancias
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
    st.info("Sube tus PDFs y pulsa **Analizar resumen ejecutivo** para extraer plazos clave.")
