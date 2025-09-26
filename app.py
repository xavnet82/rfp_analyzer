import os, sys, io
ROOT = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
from config import (APP_TITLE, OPENAI_MODEL, OPENAI_API_KEY,
                    ADMIN_USER, ADMIN_PASSWORD, MODELS_CATALOG, OPENAI_TEMPERATURE)
from services.pdf_loader import extract_pdf_text
from services.openai_client import analyze_text_chunk, merge_offers
from services.schema import OfertaAnalizada
from utils.text import clean_text, chunk_text
from components.ui import render_header, render_result

st.set_page_config(page_title=APP_TITLE, layout="wide")

def login_gate():
    if st.session_state.get("is_auth", False):
        with st.sidebar:
            if st.button("Cerrar sesión"):
                st.session_state.clear(); st.rerun()
        return True

    st.title("Acceso")
    with st.form("login_form"):
        u = st.text_input("Usuario")
        p = st.text_input("Contraseña", type="password")
        submitted = st.form_submit_button("Entrar")
    if submitted:
        if u == ADMIN_USER and p == ADMIN_PASSWORD:
            st.session_state["is_auth"] = True
            st.success("Acceso concedido."); st.rerun()
        else:
            st.error("Credenciales inválidas.")
    st.stop()

def main():
    login_gate()

    with st.sidebar:
        st.header("Configuración")
        # Selector de modelo
        options = MODELS_CATALOG + ["Otro…"]
        modo = st.radio("Modo de análisis", ["Chunking local", "PDF completo (File Search)"], index=0,
                help="File Search sube el PDF a OpenAI y lo reutiliza en varias preguntas sin reenviar texto.")
        try:
            default_index = options.index(OPENAI_MODEL) if OPENAI_MODEL in options else 0
        except Exception:
            default_index = 0
        model_choice = st.selectbox("Modelo OpenAI", options=options, index=default_index,
                                    help="Selecciona un modelo conocido o usa 'Otro…' para escribir uno.")
        if model_choice == "Otro…":
            model = st.text_input("Modelo personalizado", value=OPENAI_MODEL)
        else:
            model = model_choice

        # Temperatura
        temperature = st.slider("Temperatura", min_value=0.0, max_value=2.0, value=float(OPENAI_TEMPERATURE), step=0.1)
        max_chars = st.slider("Tamaño aprox. de chunk (caracteres)", 6_000, 40_000, 12_000, step=1_000)
        st.caption("A mayor chunk, menos llamadas pero más tokens. "
                   "Nota: algunos modelos (p.ej., gpt-5*) ignoran temperaturas ≠ 1.0.")
        if not OPENAI_API_KEY:
            st.error("No se encontró OPENAI_API_KEY. Configura `.env`, variables de entorno o `st.secrets`.")
            st.stop()

    render_header(APP_TITLE)

    files = st.file_uploader("Sube uno o varios PDFs del pliego (pliego general, anexos, etc.)",
                             type=["pdf"], accept_multiple_files=True)
    if not files:
        st.info("Sube al menos un PDF para comenzar el análisis."); st.stop()

    docs = []
    total_chunks = 0
    with st.spinner("Extrayendo texto de los PDFs..."):
        for file in files:
            pdf_bytes = file.read()
            buf = io.BytesIO(pdf_bytes)
            pages, full = extract_pdf_text(buf)
            pages = [clean_text(p) for p in pages]
            chunks = chunk_text(pages, max_chars=max_chars)
            docs.append({"name": file.name, "pages": pages, "chunks": chunks, "num_pages": len(pages)})
            total_chunks += len(chunks)

    st.success(f"Se han leído {len(docs)} fichero(s). Total páginas: {sum(d['num_pages'] for d in docs)}; total chunks: {total_chunks}.")

    if st.button("Analizar con OpenAI", type="primary"):
        if modo == "Chunking local":
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
                        result = analyze_text_chunk(result, ch, model=model, temperature=temperature)
                    except Exception as e:
                        st.error(f"Error analizando **{d['name']}**, chunk {ch_idx}/{len(d['chunks'])}: {e}")
                        raise
                    processed += 1
                    prog.progress(processed / max(total_chunks, 1))
                per_file_results[d["name"]] = result
                aggregate_result = result if aggregate_result is None else merge_offers(aggregate_result, result)
    
            status.success("Análisis completado.")
            st.session_state["per_file_results"] = per_file_results
            st.session_state["aggregate_result"] = aggregate_result
         pass
            else:
                # --- NUEVO: PDF completo ---
                from services.file_search_client import create_vector_store_from_streamlit_files, analyze_with_file_search
                vs_id = create_vector_store_from_streamlit_files(files, name="RFP Vector Store")
                st.info("PDF(s) indexados en OpenAI. Ejecutando análisis…")
        
                # 1 llamada para obtener TODO el JSON estructurado
                raw = analyze_with_file_search(vector_store_id=vs_id, model=model, temperature=temperature)
        
                # Reutiliza tu parser + esquema existentes
                import json
                from services.schema import OfertaAnalizada
                data = json.loads(raw)
                result = OfertaAnalizada.model_validate(data)
        
                st.session_state["per_file_results"] = {"(vector_store)": result}
                st.session_state["aggregate_result"] = result
                st.success("Análisis completado con File Search.")
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
