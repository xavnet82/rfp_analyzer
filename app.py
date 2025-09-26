# app.py
import os, sys, io, json, re
from typing import Optional, Dict, Any

# Asegura que la raíz del proyecto está en el path de importación
ROOT = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
from openai import OpenAI, BadRequestError

from config import (
    APP_TITLE,
    OPENAI_MODEL,
    OPENAI_API_KEY,
    OPENAI_TEMPERATURE,
    MODELS_CATALOG,
    ADMIN_USER,
    ADMIN_PASSWORD,
    MAX_TOKENS_PER_REQUEST,
)
from services.pdf_loader import extract_pdf_text
from services.schema import OfertaAnalizada
from services.openai_client import analyze_text_chunk, merge_offers  # modo chunking local
from utils.text import clean_text, chunk_text
from components.ui import render_header, render_result

# Configuración de la página (una sola vez)
st.set_page_config(page_title=APP_TITLE, layout="wide")


# =============================================================================
# Utilidades de autenticación
# =============================================================================
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


# =============================================================================
# Utilidades OpenAI – File Search (Responses API)
# Implementadas localmente para que no dependas de otros módulos
# =============================================================================
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
    m = re.search(r"\{.*\}", text, re.DOTALL)
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
    # 1) SDK recientes
    txt = getattr(rsp, "output_text", None)
    if txt:
        return txt
    # 2) Recorrer bloques (output / content / text)
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
    return None

def _is_temperature_error(e: Exception) -> bool:
    s = str(e)
    return ("temperature" in s) and ("Unsupported value" in s or "unsupported_value" in s or "does not support" in s)

def _is_unsupported_param(e: Exception, param: str) -> bool:
    s = str(e)
    return ("unsupported_parameter" in s or "Unexpected" in s or "unexpected" in s) and (param in s)

def _responses_create_robust(args: dict):
    """
    Llama a Responses API y se auto-adapta a variaciones del SDK/endpoint:
    - quita temperature si el modelo lo rechaza,
    - quita response_format si el SDK no lo soporta,
    - cambia max_output_tokens -> max_completion_tokens si es necesario.
    """
    a = dict(args)
    for _ in range(5):
        try:
            return _oai_client.responses.create(**a)
        except (BadRequestError, TypeError) as e:
            s = str(e)
            if _is_temperature_error(e) or ("unexpected keyword" in s and "temperature" in s):
                a.pop("temperature", None); continue
            if _is_unsupported_param(e, "response_format") or ("unexpected keyword" in s and "response_format" in s):
                a.pop("response_format", None); continue
            if _is_unsupported_param(e, "max_output_tokens") or ("unexpected keyword" in s and "max_output_tokens" in s):
                val = a.pop("max_output_tokens", None)
                if val is not None:
                    a["max_completion_tokens"] = val
                continue
            raise

def create_vector_store_from_streamlit_files(files, name: str = "RFP Vector Store"):
    """
    Sube los PDFs a OpenAI, los indexa en un Vector Store y
    devuelve (vector_store_id, file_ids) para usar 'attachments' si tu SDK
    no soporta tool_resources.
    """
    store = _oai_client.vector_stores.create(
        name=name,
        expires_after={"anchor": "last_active_at", "days": 2},  # opcional
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

# ---- Secciones: prompts específicos ----
SECTION_SPECS: Dict[str, Dict[str, str]] = {
    "objetivos_contexto": {
        "titulo": "Objetivos y contexto",
        "user_prompt": (
            "Sobre los PDFs aportados, extrae OBJETIVOS y CONTEXTO del pliego. "
            "Devuelve SOLO JSON con campos:\n"
            "{\n"
            '  "resumen_servicios": str,\n'
            '  "objetivos": [str],\n'
            '  "alcance": str|null,\n'
            '  "referencias_paginas": [int]\n'
            "}\n"
            "Usa números de página cuando sea posible."
        ),
    },
    "servicios": {
        "titulo": "Servicios solicitados (detalle)",
        "user_prompt": (
            "Sobre los PDFs, lista los SERVICIOS SOLICITADOS con detalle. "
            "Devuelve SOLO JSON con campos:\n"
            "{\n"
            '  "resumen_servicios": str,\n'
            '  "servicios_detalle": [{"nombre": str, "descripcion": str|null}],\n'
            '  "alcance": str|null,\n'
            '  "referencias_paginas": [int]\n'
            "}\n"
        ),
    },
    "importe": {
        "titulo": "Importe de licitación",
        "user_prompt": (
            "Extrae el IMPORTE DE LICITACIÓN. Devuelve SOLO JSON:\n"
            "{\n"
            '  "importe_total": float|null,\n'
            '  "moneda": str|null,\n'
            '  "importes_detalle": [{"concepto": str|null, "importe": float|null, "moneda": str|null, "observaciones": str|null}],\n'
            '  "referencias_paginas": [int]\n'
            "}\n"
            "Usa punto decimal. Si hay varios importes (base, IVA, prórrogas, anualidades), inclúyelos en importes_detalle."
        ),
    },
    "criterios_valoracion": {
        "titulo": "Criterios de valoración",
        "user_prompt": (
            "Extrae CRITERIOS DE VALORACIÓN y subcriterios con máximos/pesos. Devuelve SOLO JSON:\n"
            "{\n"
            '  "criterios_valoracion": [\n'
            '    {"nombre": str, "peso_max": float|null, "tipo": str|null,\n'
            '     "subcriterios": [{"nombre": str, "peso_max": float|null, "tipo": str|null, "observaciones": str|null}]}\n'
            '  ],\n'
            '  "referencias_paginas": [int]\n'
            "}\n"
            "Normaliza pesos a escala máxima si el pliego lo indica (p.ej. total 100)."
        ),
    },
    "indice_tecnico": {
        "titulo": "Índice de la respuesta técnica",
        "user_prompt": (
            "Extrae el ÍNDICE SOLICITADO de la respuesta técnica (literal del pliego) "
            "y propone un ÍNDICE ALINEADO. Devuelve SOLO JSON:\n"
            "{\n"
            '  "indice_respuesta_tecnica": [{"titulo": str, "descripcion": str|null, "subapartados": [str]}],\n'
            '  "indice_propuesto": [{"titulo": str, "descripcion": str|null, "subapartados": [str]}],\n'
            '  "referencias_paginas": [int]\n'
            "}\n"
        ),
    },
    "riesgos_exclusiones": {
        "titulo": "Riesgos y exclusiones",
        "user_prompt": (
            "Identifica RIESGOS, DUDAS, EXCLUSIONES o incompatibilidades del pliego. Devuelve SOLO JSON:\n"
            "{\n"
            '  "riesgos_y_dudas": str|null,\n'
            '  "exclusiones": [str],\n'
            '  "referencias_paginas": [int]\n'
            "}\n"
        ),
    },
    "solvencia": {
        "titulo": "Criterios de solvencia",
        "user_prompt": (
            "Extrae criterios de SOLVENCIA (técnica, económica, administrativa). Devuelve SOLO JSON:\n"
            "{\n"
            '  "solvencia": {\n'
            '    "tecnica": [str],\n'
            '    "economica": [str],\n'
            '    "administrativa": [str]\n'
            "  },\n"
            '  "referencias_paginas": [int]\n'
            "}\n"
        ),
    },
}

SYSTEM_PREFIX = (
    "Eres un analista experto en licitaciones públicas en España. "
    "Debes responder SIEMPRE con JSON válido (sin texto adicional). "
    "Si algo no aparece en los PDFs, devuelve null o listas vacías."
)

def _file_search_section_call(
    vector_store_id: str,
    user_prompt: str,
    model: str,
    temperature: Optional[float] = None,
    file_ids: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """
    Lanza una llamada a Responses+FileSearch para una sección concreta y devuelve dict.
    1º intento: tool_resources (vector store).
    Fallback si el SDK no lo soporta: attachments con file_ids.
    """

    model = (model or OPENAI_MODEL).strip()
    is_gpt5 = model.lower().startswith("gpt-5")

    # Input en formato "content parts" (más compatible entre SDKs)
    sys_msg = {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PREFIX}]}
    usr_msg = {"role": "user",   "content": [{"type": "input_text", "text": user_prompt}]}

    # ---------- Intento A: tool_resources (vector store) ----------
    args = dict(
        model=model,
        input=[sys_msg, usr_msg],
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
        response_format={"type": "json_object"},
        max_output_tokens=MAX_TOKENS_PER_REQUEST,
    )
    if temperature is not None:
        args["temperature"] = float(temperature)
    if is_gpt5:
        args.pop("temperature", None)
        args.pop("response_format", None)

    try:
        rsp = _responses_create_robust(args)
        text = _coalesce_text_from_responses(rsp)
        if not text:
            dump = json.dumps(rsp, default=str)
            text = _extract_json_block(dump)
        return _json_loads_robust(text)
    except (BadRequestError, TypeError) as e:
        # Si el SDK no soporta tool_resources, caemos al plan B (attachments)
        if "tool_resources" not in str(e):
            # otro error real -> propaga
            raise

    # ---------- Intento B: attachments (sin vector store) ----------
    if not file_ids:
        raise RuntimeError(
            "El SDK no soporta 'tool_resources' y no hay 'file_ids' para usar 'attachments'. "
            "Vuelve a indexar guardando file_ids."
        )

    args_b = dict(
        model=model,
        input=[sys_msg, usr_msg],
        tools=[{"type": "file_search"}],
        attachments=[{"file_id": fid, "tools": [{"type": "file_search"}]} for fid in file_ids],
        response_format={"type": "json_object"},
        max_output_tokens=MAX_TOKENS_PER_REQUEST,
    )
    if temperature is not None:
        args_b["temperature"] = float(temperature)
    if is_gpt5:
        args_b.pop("temperature", None)
        args_b.pop("response_format", None)

    rsp = _responses_create_robust(args_b)
    text = _coalesce_text_from_responses(rsp)
    if not text:
        dump = json.dumps(rsp, default=str)
        text = _extract_json_block(dump)
    return _json_loads_robust(text)



# =============================================================================
# Utilidades UI y Sidebar
# =============================================================================
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

        # Temperatura
        temperature = st.slider(
            "Temperatura",
            min_value=0.0, max_value=2.0,
            value=float(OPENAI_TEMPERATURE), step=0.1
        )

        # Modo
        mode = st.radio(
            "Modo de análisis",
            ["Chunking local", "PDF completo (File Search)"],
            index=0,
            help="File Search sube los PDF a OpenAI y los reutiliza en varias consultas sin reenviar texto."
        )

        # Chunking local: tamaño de chunk
        max_chars = st.slider(
            "Tamaño aprox. de chunk (caracteres)",
            6_000, 40_000, 12_000, step=1_000
        )
        st.caption(
            "A mayor chunk, menos llamadas pero más tokens. "
            "Nota: algunos modelos (p.ej., gpt-5*) ignoran temperaturas ≠ 1.0; el cliente se autoajusta."
        )

        if not OPENAI_API_KEY:
            st.error("No se encontró OPENAI_API_KEY. Configura `.env`, variables de entorno o `st.secrets`.")
            st.stop()

    return model, temperature, max_chars, mode


def analyze_chunk_safe(current_result, ch_text, model, temperature):
    """Compat con firmas antiguas: reintenta sin 'temperature' si el servicio no lo acepta."""
    try:
        return analyze_text_chunk(current_result, ch_text, model=model, temperature=temperature)
    except TypeError:
        return analyze_text_chunk(current_result, ch_text, model=model)


# =============================================================================
# App principal
# =============================================================================
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

    # ======================================
    # A) Modo Chunking local (como antes)
    # ======================================
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

        if st.button("Analizar con OpenAI (chunking local)", type="primary"):
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

    # ======================================
    # B) Modo PDF completo (File Search)
    # ======================================
    else:
        # (1) Preparar/crear Vector Store (solo 1 vez por subida)
        if "fs_vs_id" not in st.session_state or st.button("Reindexar PDFs en OpenAI"):
            try:
                st.session_state.pop("fs_sections", None)  # limpia resultados anteriores
            except KeyError:
                pass
            with st.spinner("Indexando PDFs en OpenAI (Vector Store)..."):
                vs_id, file_ids = create_vector_store_from_streamlit_files(files, name="RFP Vector Store")
            st.session_state["fs_vs_id"] = vs_id
            st.session_state["fs_file_ids"] = file_ids
            st.success("PDF(s) indexados en OpenAI.")
        
        vs_id = st.session_state.get("fs_vs_id")
        file_ids = st.session_state.get("fs_file_ids", [])
        if not vs_id:
            st.stop()
        
        st.info(f"Vector Store listo: `{vs_id}` con {len(file_ids)} archivo(s)")

        # (2) Botonera de secciones
        st.subheader("Análisis por secciones")
        c1, c2, c3 = st.columns(3)
        with c1:
            b_obj = st.button("Objetivos y contexto")
            b_serv = st.button("Servicios solicitados")
            b_imp = st.button("Importe de licitación")
        with c2:
            b_crit = st.button("Criterios de valoración")
            b_idx = st.button("Índice de la respuesta técnica")
            b_risk = st.button("Riesgos y exclusiones")
        with c3:
            b_solv = st.button("Criterios de solvencia")
            # espacio para futuros botones
            st.write("")

        if "fs_sections" not in st.session_state:
            st.session_state["fs_sections"] = {}

        # (3) Ejecutar sección solicitada
        def run_section(section_key: str):
            spec = SECTION_SPECS[section_key]
            with st.spinner(f"Analizando sección: {spec['titulo']}..."):
                data = _file_search_section_call(
                    vector_store_id=vs_id,
                    user_prompt=spec["user_prompt"],
                    model=model,
                    temperature=temperature,
                )
            st.session_state["fs_sections"][section_key] = data

        if b_obj:  run_section("objetivos_contexto")
        if b_serv: run_section("servicios")
        if b_imp:  run_section("importe")
        if b_crit: run_section("criterios_valoracion")
        if b_idx:  run_section("indice_tecnico")
        if b_risk: run_section("riesgos_exclusiones")
        if b_solv: run_section("solvencia")

        # (4) Mostrar resultados por sección (si existen)
        st.subheader("Resultados")
        for key, spec in SECTION_SPECS.items():
            if key in st.session_state["fs_sections"]:
                with st.expander(spec["titulo"], expanded=False):
                    st.json(st.session_state["fs_sections"][key])
                    st.download_button(
                        f"Descargar JSON – {spec['titulo']}",
                        json.dumps(st.session_state["fs_sections"][key], indent=2, ensure_ascii=False),
                        file_name=f"{key}.json",
                        mime="application/json",
                    )

        # (5) (Opcional) Fusionar a un resultado agregado del esquema principal
        if st.session_state["fs_sections"]:
            if st.button("Construir resultado agregado (map a esquema principal)"):
                # Mapear las secciones que encajan directamente en OfertaAnalizada
                payload: Dict[str, Any] = {
                    "resumen_servicios": "",
                    "objetivos": [],
                    "alcance": None,
                    "importe_total": None,
                    "moneda": None,
                    "importes_detalle": [],
                    "criterios_valoracion": [],
                    "indice_respuesta_tecnica": [],
                    "indice_propuesto": [],
                    "riesgos_y_dudas": None,
                    "referencias_paginas": [],
                }

                sec = st.session_state["fs_sections"]
                # objetivos/contexto
                oc = sec.get("objetivos_contexto", {})
                payload["resumen_servicios"] = oc.get("resumen_servicios") or payload["resumen_servicios"]
                payload["objetivos"] = oc.get("objetivos") or payload["objetivos"]
                payload["alcance"] = oc.get("alcance") or payload["alcance"]
                payload["referencias_paginas"] += oc.get("referencias_paginas", [])

                # servicios: solo añadimos resumen/alcance; el detalle queda fuera del esquema principal
                sv = sec.get("servicios", {})
                if isinstance(sv.get("resumen_servicios"), str) and len(sv["resumen_servicios"]) > len(payload["resumen_servicios"]):
                    payload["resumen_servicios"] = sv["resumen_servicios"]
                payload["alcance"] = payload["alcance"] or sv.get("alcance")
                payload["referencias_paginas"] += sv.get("referencias_paginas", [])

                # importe
                im = sec.get("importe", {})
                payload["importe_total"] = im.get("importe_total") or payload["importe_total"]
                payload["moneda"] = im.get("moneda") or payload["moneda"]
                payload["importes_detalle"] += im.get("importes_detalle", [])
                payload["referencias_paginas"] += im.get("referencias_paginas", [])

                # criterios
                cv = sec.get("criterios_valoracion", {})
                payload["criterios_valoracion"] += cv.get("criterios_valoracion", [])
                payload["referencias_paginas"] += cv.get("referencias_paginas", [])

                # índice
                ix = sec.get("indice_tecnico", {})
                payload["indice_respuesta_tecnica"] += ix.get("indice_respuesta_tecnica", [])
                payload["indice_propuesto"] += ix.get("indice_propuesto", [])
                payload["referencias_paginas"] += ix.get("referencias_paginas", [])

                # riesgos/exclusiones
                rx = sec.get("riesgos_exclusiones", {})
                # concatenamos exclusiones dentro de riesgos_y_dudas si existen
                ry = rx.get("riesgos_y_dudas")
                ex = rx.get("exclusiones", [])
                if ry or ex:
                    txt = (ry or "") + ("\nExclusiones: " + "; ".join(ex) if ex else "")
                    payload["riesgos_y_dudas"] = (payload["riesgos_y_dudas"] + "\n" + txt).strip() if payload["riesgos_y_dudas"] else txt
                payload["referencias_paginas"] += rx.get("referencias_paginas", [])

                # ordena y dedup páginas
                try:
                    payload["referencias_paginas"] = sorted(set(int(p) for p in payload["referencias_paginas"]))
                except Exception:
                    payload["referencias_paginas"] = []

                # Validar contra esquema principal
                try:
                    result = OfertaAnalizada.model_validate(payload)
                    st.session_state["aggregate_result"] = result
                    st.success("Resultado agregado construido.")
                except Exception as e:
                    st.error(f"No fue posible validar contra el esquema principal: {e}")

    # ======================================
    # Visualización y descargas (común)
    # ======================================
    per_file_results = st.session_state.get("per_file_results", {})
    aggregate_result = st.session_state.get("aggregate_result", None)

    # Vista de resultados (cuando hay agregado o por fichero del modo local)
    if aggregate_result is not None or per_file_results:
        options = ["Agregado (todos)"] + list(per_file_results.keys())
        choice = st.selectbox("Vista de resultados (modo chunking local / agregado File Search)", options)

        if choice == "Agregado (todos)":
            if isinstance(aggregate_result, OfertaAnalizada):
                render_result(aggregate_result)
        else:
            r = per_file_results.get(choice)
            if isinstance(r, OfertaAnalizada):
                render_result(r)

        # Exportaciones de agregado (si existe)
        if isinstance(aggregate_result, OfertaAnalizada):
            with st.expander("Descargas del resultado agregado"):
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


if __name__ == "__main__":
    main()
