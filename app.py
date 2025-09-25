import io
import streamlit as st
from config import APP_TITLE, OPENAI_MODEL
from services.pdf_loader import extract_pdf_text
from services.openai_client import analyze_text_chunk
from services.schema import OfertaAnalizada
from utils.text import clean_text, chunk_text

st.set_page_config(page_title=APP_TITLE, layout="wide")
from components.ui import render_header, render_result

def main():
    with st.sidebar:
        st.header("Configuración")
        model = st.text_input("Modelo OpenAI", value=OPENAI_MODEL, help="p. ej. gpt-4o-mini, o el que tengas disponible")
        max_chars = st.slider("Tamaño aprox. de chunk (caracteres)", 6_000, 40_000, 12_000, step=1_000)
        st.caption("A mayor chunk, menos llamadas pero más tokens. Ajusta según coste/tiempo.")
    render_header(APP_TITLE)

    file = st.file_uploader("Sube el PDF del pliego", type=["pdf"])
    if not file:
        st.info("Sube un PDF para comenzar el análisis.")
        st.stop()

    # Guardar en buffer para pypdf
    with st.spinner("Extrayendo texto del PDF..."):
        pdf_bytes = file.read()
        buf = io.BytesIO(pdf_bytes)
        pages, full = extract_pdf_text(buf)
        pages = [clean_text(p) for p in pages]
        chunks = chunk_text(pages, max_chars=max_chars)

    st.success(f"PDF leído: {len(pages)} páginas; {len(chunks)} chunks para analizar.")
    if st.button("Analizar con OpenAI", type="primary"):
        result = None
        prog = st.progress(0)
        status = st.empty()
        for i, ch in enumerate(chunks, start=1):
            status.info(f"Analizando chunk {i}/{len(chunks)}...")
            result = analyze_text_chunk(result, ch, model=model)
            prog.progress(i / len(chunks))
        status.success("Análisis completado.")
        st.session_state["resultado"] = result

    if "resultado" in st.session_state and isinstance(st.session_state["resultado"], OfertaAnalizada):
        render_result(st.session_state["resultado"])

if __name__ == "__main__":
    main()
