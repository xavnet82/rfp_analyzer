# app.py
import os, sys, io

# Asegura que la raíz del proyecto está en el path de importación
ROOT = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
from config import (
    APP_TITLE,
    OPENAI_MODEL,
    OPENAI_API_KEY,
    ADMIN_USER,
    ADMIN_PASSWORD,
    MODELS_CATALOG,
    OPENAI_TEMPERATURE,
)

# Servicios (local chunking)
from services.pdf_loader import extract_pdf_text
from services.openai_client import analyze_text_chunk, merge_offers
from services.schema import OfertaAnalizada
from utils.text import clean_text, chunk_text
from components.ui import render_header, render_result

# Servicios (File Search / Vector Store) - opcional
try:
    from services.file_search_client import (
        create_vector_store_from_streamlit_files,
        analyze_with_file_search,
    )
    FS_AVAILABLE = True
except Exception:
    FS_AVAILABLE = False

# Configuración de la página (una sola vez)
st.set_page_config(page_title=APP_TITLE, layout="wide")


# ---------------------------
# Utilidades locales
# ---------------------------

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


def analyze_chunk_safe(current_result, ch_text, model, temperature):
    """
    Llama a analyze_text_chunk pasando temperatura; si el cliente aún no la soporta,
    reintenta sin el parámetro para no romper.
    """
    try:
        return analyze_text_chunk(current_result, ch_text, model=model, temperature=temperature)
    except TypeError:
        # Firma anterior sin 'temperature'
        return analyze_text_chunk(current_result, ch_text, model=model)


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

        # Temperatura (algunos modelos gpt-5* sólo aceptan 1.0; el cliente ya lo adapta si hace falta)
        temperature = st.slider(
            "Temperatura",
            min_value=0.0, max_value=2.0,
            value=float(OPENAI_TEMPERATURE), step=0.1
        )

        # Modo de análisis
        if FS_AVAILABLE:
            mode = st.radio(
                "Modo de análisis",
                ["Chunking local", "PDF completo (File Search)"],
                index=0,
                help="File Search sube los PDF a OpenAI y los reutiliza en varias consultas sin reenviar texto."
            )
        else:
            mode = "Chunking local"
            st.info("File Search no está disponible (módulo no encontrado). Usando 'Chunking local'.")

        # Chunking local: tamaño de chunk
        max_chars = st.slider(
            "Tamaño aprox. de chunk (caracteres)",
            6_000, 40_000, 12_000, step=1_000
        )
        st.caption(
            "A mayor chunk, menos llamadas pero más tokens. "
            "Nota: algunos modelos (p.ej., gpt-5*) ignoran temperaturas ≠ 1.0; el cliente se autoajusta."
        )

        # API Key
        if not OPENAI_API_KEY:
            st.error("No se encontró OPENAI_API_KEY. Configura `.env`, variables de entorno o `st.secrets`.")
            st.stop()

    return model, temperature, max_chars, mode


# ---------------------------
# App principal
# ---------------------------

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

    # 5) Ruta A: Chunking local
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

        if st.button("Analizar con OpenAI", type="primary"):
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

    # 6) Ruta B: PDF completo (File Search)
    else:
        if not FS_AVAILABLE:
            st.error("El modo 'PDF completo (File Search)' no está disponible: faltan dependencias.")
            st.stop()

        if st.button("Analizar con OpenAI (File Search)", type="primary"):
            try:
                # 1) Crear Vector Store e indexar PDFs en OpenAI
                vs_id = create_vector_store_from_streamlit_files(files, name="RFP Vector Store")
                st.info("PDF(s) indexados en OpenAI. Ejecutando análisis…")

                # 2) Una llamada para obtener el JSON completo
                raw = analyze_with_file_search(vector_store_id=vs_id, model=model, temperature=temperature)

                # 3) Parseo con tu esquema
                import json
                data = json.loads(raw)
                result = OfertaAnalizada.model_validate(data)

                # Guardar en sesión (resultado agregado único)
                st.session_state["per_file_results"] = {"(vector_store)": result}
                st.session_state["aggregate_result"] = result
                st.success("Análisis completado con File Search.")
            except Exception as e:
                st.error(f"Error en File Search: {e}")
                raise

    # 7) Visualización y descargas
    per_file_results = st.session_state.get("per_file_results", {})
    aggregate_result = st.session_state.get("aggregate_result", None)

    if not per_file_results and aggregate_result is None:
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

    # Exportaciones
    if per_file_results or aggregate_result:
        with st.expander("Descargas por fichero y agregado"):
            if aggregate_result:
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
            for name, r in per_file_results.items():
                st.download_button(
                    f"Descargar JSON ({name})",
                    r.model_dump_json(indent=2),
                    file_name=f"analisis_{name}.json",
                    mime="application/json"
                )
                st.download_button(
                    f"Descargar Markdown ({name})",
                    r.to_markdown,
                    file_name=f"analisis_{name}.md",
                    mime="text/markdown"
                )


if __name__ == "__main__":
    main()
