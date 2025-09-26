import streamlit as st
from typing import List
from services.schema import OfertaAnalizada, SeccionIndice

def render_header(title: str):
    st.title(title)
    st.caption("Analiza pliegos PDF (uno o varios) y obtiene un resumen estructurado.")

def _render_indice(nombre: str, indice: List[SeccionIndice]):
    st.subheader(nombre)
    if indice:
        for i, sec in enumerate(indice, 1):
            with st.expander(f"{i}. {sec.titulo}", expanded=False):
                if sec.descripcion:
                    st.write(sec.descripcion)
                for j, sub in enumerate(sec.subapartados, 1):
                    st.write(f"{i}.{j} {sub}")
    else:
        st.write("No se detectó información en esta sección.")

def render_result(result: OfertaAnalizada):
    tabs = st.tabs(["Resumen", "Importes", "Criterios", "Índice", "Riesgos", "Exportar"])
    with tabs[0]:
        st.subheader("Servicios solicitados")
        st.write(result.resumen_servicios or "No disponible")

        cols = st.columns(2)
        with cols[0]:
            st.markdown("### Objetivos del pliego")
            if result.objetivos:
                for o in result.objetivos:
                    st.write(f"- {o}")
            else:
                st.write("No se detectaron objetivos explícitos.")
        with cols[1]:
            st.markdown("### Alcance y límites")
            st.write(result.alcance or "No disponible")

        if result.referencias_paginas:
            st.info("Referencias de páginas: " + ", ".join(map(str, result.referencias_paginas)))
    with tabs[1]:
        st.subheader("Importes")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Importe total", f"{result.importe_total if result.importe_total is not None else 'N/D'} {result.moneda or ''}".strip())
        with c2:
            st.write("Moneda detectada:", result.moneda or "N/D")
        if result.importes_detalle:
            st.write("Detalle:")
            for d in result.importes_detalle:
                st.write(f"- {d.concepto or 'Concepto'} — {d.importe if d.importe is not None else 'N/D'} {d.moneda or ''} {('· ' + d.observaciones) if d.observaciones else ''}")
        else:
            st.write("Sin detalle detectado.")
    with tabs[2]:
        st.subheader("Criterios de valoración")
        if result.criterios_valoracion:
            for c in result.criterios_valoracion:
                with st.expander(f"{c.nombre} (máx: {c.peso_max if c.peso_max is not None else 'N/D'}) · {c.tipo or 'N/D'}", expanded=False):
                    if c.subcriterios:
                        for s in c.subcriterios:
                            st.write(f"- {s.nombre} (máx: {s.peso_max if s.peso_max is not None else 'N/D'}; tipo: {s.tipo or 'N/D'}) {('· ' + s.observaciones) if s.observaciones else ''}")
                    else:
                        st.write("Sin subcriterios.")
        else:
            st.write("No se detectaron criterios.")
    with tabs[3]:
        sub = st.tabs(["Índice solicitado (literal)", "Índice propuesto (alineado)"])
        with sub[0]:
            _render_indice("Índice solicitado (del pliego)", result.indice_respuesta_tecnica)
        with sub[1]:
            _render_indice("Índice propuesto", result.indice_propuesto)
    with tabs[4]:
        st.subheader("Riesgos y dudas")
        st.write(result.riesgos_y_dudas or "No se detectaron riesgos/dudas.")
    with tabs[5]:
        st.subheader("Exportaciones")
        st.download_button("Descargar JSON", result.model_dump_json(indent=2), file_name="analisis_pliego.json", mime="application/json")
        st.download_button("Descargar Markdown", result.to_markdown, file_name="analisis_pliego.md", mime="text/markdown")
