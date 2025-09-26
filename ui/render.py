
# ui/render.py
from typing import Any, Dict, List
import streamlit as st
from utils.text import bullets

def badge(text: str):
    st.markdown(
        f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;"
        f"background:#eef;border:1px solid #ccd;color:#334;font-size:12px'>{text}</span>",
        unsafe_allow_html=True
    )

def render_index(items: List[Dict[str, Any]]):
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
            st.markdown(bullets(subs))

def resumen_ejecutivo(fs_sections: Dict[str, Any]):
    oc = fs_sections.get("objetivos_contexto", {})
    im = fs_sections.get("importe", {})
    fmt = fs_sections.get("formato_oferta", {})

    # Importe total
    imp_total = im.get("importe_total")
    moneda = im.get("moneda") or "EUR"
    imp_str = f"{imp_total:.2f} {moneda}" if isinstance(imp_total, (int, float)) else "—"

    # Duración del contrato (aprox) desde importes_detalle[].periodo.duracion_meses si existe
    dur = None
    for d in im.get("importes_detalle") or []:
        per = d.get("periodo") or {}
        if isinstance(per.get("duracion_meses"), (int, float)):
            dur = (dur or 0) + int(per["duracion_meses"])

    # Fecha máxima de presentación (del bloque formato/entrega.plazo si existe)
    entrega = (fmt.get("entrega") or {}).get("plazo")

    st.markdown("### Resumen ejecutivo")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Importe total", imp_str)
    c2.metric("Duración del contrato", f"{dur} meses" if dur else "—")
    c3.metric("Fecha máx. presentación", entrega or "—")
    c4.metric("Secciones completas", sum(1 for k, v in fs_sections.items() if v))
