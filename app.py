import io
import os, sys
ROOT = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import streamlit as st
from config import APP_TITLE, OPENAI_MODEL, OPENAI_API_KEY, ADMIN_USER, ADMIN_PASSWORD
from services.pdf_loader import extract_pdf_text
from services.openai_client import analyze_text_chunk, merge_offers
from services.schema import OfertaAnalizada
from utils.text import clean_text, chunk_text

def login_gate():
    # Si ya está autenticado, ofrecer cierre de sesión
    if st.session_state.get("is_auth", False):
        with st.sidebar:
            if st.button("Cerrar sesión"):
                st.session_state.clear()
                st.rerun()
        return True

    st.set_page_config(page_title=APP_TITLE, layout="wide")
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

st.set_page_config(page_title=APP_TITLE, layout="wide")
from components.ui import render_header, render_result

def main():
    # Gate de autenticación
    login_gate()

    with st.sidebar:
        st.header("Configuración")
        model = st.text_input("Modelo OpenAI", value=OPENAI_MODEL, help="p. ej. gpt-4o-mini, gpt-5-mini, etc.")
        max_chars = st.slider("Tamaño aprox. de chunk (caracteres)", 6_000, 40_000, 12_000, step=1_000)
        st.caption("A mayor chunk, menos llamadas pero más tokens. Ajusta según coste/tiempo.")
        if not OPENAI_API_KEY:
            st.error("No se encontró OPENAI_API_KEY. Configura `.env`, variables de entorno o `st.secrets` antes de analizar.")
            st.stop()

    render_header(APP_TITLE)

    files = st.file_uploader("Sube uno o varios PDFs del pliego (pliego general, anexos, etc.)", type=["pdf"], accept_multiple_files=True)
    if not files:
        st.info("Sube al menos un PDF para comenzar el análisis.")
        st.stop()

    # Preprocesar todos los PDFs y preparar chunks
    docs = []
    total_chunks = 0
    with st.spinner("Extrayendo texto de los PDFs..."):
        for file in files:
            pdf_bytes = file.read()
            buf = io.BytesIO(pdf_bytes)
            pages, full = extract_pdf_text(buf)
            pages = [clean_text(p) for p in pages]
            chunks = chunk_text(pages, max_chars=max_chars)
            docs.append({
                "name": file.name,
                "pages": pages,
                "chunks": chunks,
                "num_pages": len(pages)
            })
            total_chunks += len(chunks)

    st.success(f"Se han leído {len(docs)} fichero(s). Total páginas: {sum(d['num_pages'] for d in docs)}; total chunks: {total_chunks}.")

    if st.button("Analizar con OpenAI", type="primary"):
        aggregate_result = None
        per_file_results = {}
        prog = st.progress(0.0)
        status = st.empty()
        processed = 0

        for d in docs:
            status.info(f"Analizando: {d['name']} ({len(d['chunks'])} chunks)...")
            result = None
            for ch_idx, ch in enumerate(d["chunks"], start=1):
                try:
                    result = analyze_text_chunk(result, ch, model=model)
                except Exception as e:
                    st.error(f"Error analizando **{d['name']}**, chunk {ch_idx}/{len(d['chunks'])']}: {e}")
                    raise
                processed += 1
                prog.progress(processed / max(total_chunks, 1))
            per_file_results[d["name"]] = result
            aggregate_result = result if aggregate_result is None else merge_offers(aggregate_result, result)

        status.success("Análisis completado.")
        st.session_state["per_file_results"] = per_file_results
        st.session_state["aggregate_result"] = aggregate_result

    # Selector de vista (agregada o por fichero)
    per_file_results = st.session_state.get("per_file_results", {})
    aggregate_result = st.session_state.get("aggregate_result", None)

    if aggregate_result is None and not per_file_results:
        st.stop()

    options = ["Agregado (todos)"] + list(per_file_results.keys())
    choice = st.selectbox("Vista de resultados", options)

    if choice == "Agregado (todos)":
        if isinstance(aggregate_result, OfertaAnalizada):
            render_result(aggregate_result)
    else:
        r = per_file_results.get(choice)
        if isinstance(r, OfertaAnalizada):
            render_result(r)

    # Exportaciones múltiples (agregado + por fichero)
    if per_file_results or aggregate_result:
        with st.expander("Descargas por fichero y agregado"):
            if aggregate_result:
                st.download_button("Descargar JSON (Agregado)", aggregate_result.model_dump_json(indent=2),
                                   file_name="analisis_agregado.json", mime="application/json")
                st.download_button("Descargar Markdown (Agregado)", aggregate_result.to_markdown,
                                   file_name="analisis_agregado.md", mime="text/markdown")
            for name, r in per_file_results.items():
                st.download_button(f"Descargar JSON ({name})", r.model_dump_json(indent=2),
                                   file_name=f"analisis_{name}.json", mime="application/json")
                st.download_button(f"Descargar Markdown ({name})", r.to_markdown,
                                   file_name=f"analisis_{name}.md", mime="text/markdown")

if __name__ == "__main__":
    main()
