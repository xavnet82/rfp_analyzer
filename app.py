
# app.py — versión consolidada y estable (tabs definidos correctamente)
import os, io, re, json
from datetime import datetime
from typing import Any, Dict, List

import streamlit as st
from openai import OpenAI

from services.pdf_loader import extract_pdf_text
from services.openai_client import create_client, responses_create_robust, coalesce_text_from_responses
from prompts.specs import SYSTEM_PREFIX, SECTION_SPECS, SECTION_CONTEXT_TUNING, SECTION_KEYWORDS
from utils.text import clean_text, bullets
from ui.render import render_index, resumen_ejecutivo

APP_TITLE = "Accenture RFP Analyzer"
AVAILABLE_MODELS = ["gpt-4o", "gpt-4o-mini"]
DEFAULT_TEMPERATURE = 0.2
LOCAL_CONTEXT_MAX_CHARS = 40000
MAX_TOKENS_PER_REQUEST = 1800

# Verbosidad → tokens/contexto
VERBOSITY_TOKENS = {1: 700, 2: 1000, 3: 1400, 4: 1800, 5: 2200}
VERBOSITY_CTX_MULT = {1: 0.6, 2: 0.8, 3: 1.0, 4: 1.2, 5: 1.4}


OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Falta OPENAI_API_KEY")
    st.stop()

client = create_client(OPENAI_API_KEY)

st.set_page_config(page_title=APP_TITLE, layout="wide")

# ---------------- Helpers ----------------
def login():
    admin_user = os.getenv("ADMIN_USER", "admin")
    admin_pass = os.getenv("ADMIN_PASSWORD", "rfpanalyzer")
    if st.session_state.get("auth"):
        return
    st.title("Acceso")
    with st.form("login"):
        u = st.text_input("Usuario")
        p = st.text_input("Contraseña", type="password")
        ok = st.form_submit_button("Entrar")
        if ok:
            if u == admin_user and p == admin_pass:
                st.session_state["auth"] = True
                st.success("Acceso concedido.")
                st.rerun()
            else:
                st.error("Credenciales inválidas.")
    st.stop()

def strip_code_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r'^\s*```(?:json)?\s*', '', s, flags=re.I)
    s = re.sub(r'\s*```\s*$', '', s)
    return s

def extract_json_block(s: str) -> str:
    m = re.search(r"\{[\s\S]*\}", s)
    if not m: raise RuntimeError("Sin JSON parseable.")
    return m.group(0)

def loads_robust(raw):
    if raw is None: raise RuntimeError("Vacío")
    if not isinstance(raw, str): return raw
    s = strip_code_fences(raw)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return json.loads(extract_json_block(s))

def score_page(text: str, weights: dict) -> int:
    if not text: return 0
    t = text.lower()
    return sum(t.count(k)*w for k, w in weights.items())

def select_relevant_spans(pages: List[str], section_key: str, max_chars: int = LOCAL_CONTEXT_MAX_CHARS, window: int = 1) -> str:
    tune = SECTION_CONTEXT_TUNING.get(section_key, {})
    max_chars = tune.get("max_chars", max_chars)
    window = tune.get("window", window)
    weights = SECTION_KEYWORDS.get(section_key, {})
    scored = [(score_page(p, weights), i) for i, p in enumerate(pages)]
    scored.sort(reverse=True)
    used, total, selected = set(), 0, []
    for sc, i in scored:
        if sc <= 0: break
        for j in range(max(0, i-window), min(len(pages), i+window+1)):
            if j in used: continue
            txt = pages[j]
            if not txt: continue
            if total + len(txt) > max_chars: break
            selected.append(f"[Pág {j+1}]\n{txt}")
            used.add(j)
            total += len(txt)
        if total >= max_chars: break
    if not selected:
        for j, txt in enumerate(pages[:2]):
            if txt:
                selected.append(f"[Pág {j+1}]\n{txt}")
                total += len(txt)
                if total >= max_chars: break
    return "\n\n".join(selected)

# ---------- Logging ----------
def _logs_init():
    st.session_state.setdefault("logs", {})
    for k in SECTION_SPECS.keys():
        st.session_state["logs"].setdefault(k, [])

def _log_event(section_key: str, model: str, temperature: float, prompt: str, response_text: str, mode: str):
    _logs_init()
    st.session_state["logs"][section_key].append({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "temperature": temperature,
        "mode": mode,  # "input_file" | "local_fallback"
        "prompt": prompt,
        "response": response_text,
    })

# ---------- Currency helpers ----------
def _to_float(x):
    try:
        if isinstance(x, str):
            x = x.replace('€','').replace('$','').replace('£','').replace('R$','')\
                 .replace('.', '').replace(',', '.').strip()
        return float(x)
    except Exception:
        return None

def _curr_symbol(c):
    if not c: return ""
    c = str(c).upper()
    return {"EUR":"€","USD":"$","GBP":"£","MXN":"$","COP":"$","CLP":"$","ARS":"$","BRL":"R$"}.get(c, c)

def fmt_money(value, currency="EUR"):
    v = _to_float(value)
    if v is None: return "—"
    s = _curr_symbol(currency)
    return f"{v:,.2f} {s}".replace(",", "_").replace(".", ",").replace("_", ".")

def render_extra_fields(title: str, data: dict, used_keys: set):
    if not isinstance(data, dict): 
        return
    extra = {k:v for k,v in data.items() if k not in used_keys and v not in (None, "", [], {})}
    if not extra:
        return
    with st.expander(title, expanded=False):
        for k, v in extra.items():
            header = k.replace("_"," ").capitalize()
            if isinstance(v, list):
                if v and isinstance(v[0], dict):
                    try:
                        import pandas as pd
                        st.dataframe(pd.DataFrame(v), use_container_width=True, hide_index=True)
                    except Exception:
                        for it in v:
                            st.markdown(f"- {it}")
                else:
                    for it in v:
                        st.markdown(f"- {it}")
            elif isinstance(v, dict):
                try:
                    import pandas as pd
                    st.dataframe(pd.DataFrame([v]), use_container_width=True, hide_index=True)
                except Exception:
                    st.markdown("; ".join(f"**{a}:** {b}" for a,b in v.items()))
            else:
                st.markdown(f"**{header}:** {v}")

# ---------- OpenAI calls ----------
def file_input_call(user_prompt: str, model: str, temperature: float, file_ids: List[str], section_key: str, max_tokens: int, verbosity: int):
    sys_msg = {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PREFIX}]}
    style = (
        "EJECUTIVO/CONCISO" if verbosity <= 2 else ("EQUILIBRADO" if verbosity == 3 else "DETALLADO/EXHAUSTIVO")
    )
    user_prompt = user_prompt + f"\n\n[NIVEL DE DETALLE]\n- Nivel: {verbosity} ({style}).\n- Si ejecutivo: síntesis en bullets cortos, sin adornos.\n- Si detallado: estructura clara, tablas/listas y referencias a páginas si procede."
    content = [{"type": "input_text", "text": user_prompt}]
    for fid in file_ids:
        content.append({"type": "input_file", "file_id": fid})
    usr_msg = {"role": "user", "content": content}

    prompt_text = f"[SYSTEM]\\n{SYSTEM_PREFIX}\\n\\n[USER]\\n{user_prompt}\\n\\n[FILES]\\n" + "\\n".join(file_ids)

    args = dict(
        model=model,
        input=[sys_msg, usr_msg],
        response_format={"type": "json_object"},
        max_output_tokens=max_tokens,
        temperature=temperature,
    )
    rsp = responses_create_robust(client, args)
    txt = coalesce_text_from_responses(rsp) or json.dumps(rsp, default=str)

    _log_event(section_key, f"{model} (v={verbosity}, maxtok={max_tokens})", temperature, prompt_text, (txt or "")[:200000], "input_file")

    return loads_robust(txt)

def local_call(section_key: str, model: str, temperature: float, max_chars: int, max_tokens: int, verbosity: int):
    docs = st.session_state.get("local_docs", [])
    contexts = []
    for d in docs:
        sel = select_relevant_spans(d["pages"], section_key, max_chars=int(max_chars * VERBOSITY_CTX_MULT.get(verbosity,1.0)))
        if sel: contexts.append(sel)
    context = "\\n\\n".join(contexts)[:120000]

    schema_hint = SECTION_SPECS[section_key]["user_prompt"]
    sys_msg = {"role": "system", "content": [{"type": "input_text", "text": "Responde SOLO con JSON válido."}]}
    usr_text = (
        "Extrae la sección solicitada según el siguiente esquema (JSON). "
        "Usa EXCLUSIVAMENTE el contexto que sigue.\\n\\n[ESQUEMA]\\n"
        f"{schema_hint}\\n\\n[CONTEXTO]\\n<<<\\n{context}\\n>>>\\n"
        "Responde solo con JSON válido."
    )
    usr_msg = {"role": "user", "content": [{"type": "input_text", "text": usr_text}]}

    args = dict(
        model=model,
        input=[sys_msg, usr_msg],
        response_format={"type": "json_object"},
        max_output_tokens=max_tokens,
        temperature=temperature,
    )
    rsp = responses_create_robust(client, args)
    txt = coalesce_text_from_responses(rsp) or json.dumps(rsp, default=str)

    _log_event(section_key, f"{model} (v={verbosity}, maxtok={max_tokens})", temperature, usr_text, (txt or "")[:200000], "local_fallback")

    return loads_robust(txt)

# ---------------- Main ----------------
def main():
    login()
    st.title(APP_TITLE)
    with st.sidebar:
        model = st.selectbox("Modelo OpenAI", AVAILABLE_MODELS, index=0)
        verbosity = st.slider(
            "Nivel de detalle de las respuestas",
            1, 5, 3,
            help=(
                "1 = muy ejecutivo y rápido (resúmenes cortos); 3 = equilibrio; 5 = muy detallado (más ejemplos, tablas y justificación)."
            ),
        )
        # Temperatura oculta (determinista)
        temperature = DEFAULT_TEMPERATURE
        

    _logs_init()

    files = st.file_uploader("Sube PDF(s)", type=["pdf"], accept_multiple_files=True)

    recargar = st.button("♻️ Recargar ficheros en OpenAI", help="Vuelve a subir los PDF y reinicia el análisis.")
    if recargar and not files:
        st.warning("Selecciona los PDF antes de recargar.")
    if (recargar and files) or ("file_ids" not in st.session_state and files):
        with st.spinner("Subiendo PDFs a OpenAI y extrayendo texto…"):
            fids, local_docs = [], []
            for f in files:
                content = f.read()
                up = client.files.create(file=(f.name, content, "application/pdf"), purpose="assistants")
                fids.append(up.id)
                pages, _ = extract_pdf_text(io.BytesIO(content))
                local_docs.append({"name": f.name, "pages": [clean_text(p) for p in pages]})
            st.session_state["file_ids"] = fids
            st.session_state["local_docs"] = local_docs
            st.session_state["sections"] = {}
            st.session_state["logs"] = {}   # limpiar logs
        st.success("Ficheros listos.")
    if "file_ids" not in st.session_state:
        st.stop()

    # ----- Tabs -----
    tabs = st.tabs(["Análisis", "Logs"])

    # ----- Tab 0: Análisis -----
    with tabs[0]:
        st.subheader("Ejecutar análisis")
        cols = st.columns(4)
        btns = {
            "objetivos_contexto": cols[0].button("Objetivos y contexto", use_container_width=True),
            "servicios": cols[1].button("Servicios", use_container_width=True),
            "importe": cols[2].button("Importe", use_container_width=True),
            "criterios_valoracion": cols[3].button("Criterios de valoración", use_container_width=True),
        }
        cols2 = st.columns(4)
        btns.update({
            "indice_tecnico": cols2[0].button("Índice técnico", use_container_width=True),
            "riesgos_exclusiones": cols2[1].button("Riesgos y exclusiones", use_container_width=True),
            "solvencia": cols2[2].button("Solvencia", use_container_width=True),
            "formato_oferta": cols2[3].button("Formato/Entrega", use_container_width=True),
        })
        all_btn = st.button("🔎 Análisis completo", type="primary", use_container_width=True)

        st.session_state.setdefault("sections", {})
        file_ids = st.session_state["file_ids"]

        def run_one(k: str, show_spinner: bool = True):
            spec = SECTION_SPECS[k]
            max_tokens = VERBOSITY_TOKENS.get(verbosity, MAX_TOKENS_PER_REQUEST)
            if show_spinner:
                with st.spinner(f"Analizando: {spec['titulo']}…"):
                    try:
                        data = file_input_call(SECTION_SPECS[k]["user_prompt"], model, temperature, file_ids, k, max_tokens, verbosity)
                    except Exception:
                        data = local_call(k, model, temperature, LOCAL_CONTEXT_MAX_CHARS, max_tokens, verbosity)
            else:
                try:
                    data = file_input_call(SECTION_SPECS[k]["user_prompt"], model, temperature, file_ids, k, max_tokens, verbosity)
                except Exception:
                    data = local_call(k, model, temperature, LOCAL_CONTEXT_MAX_CHARS, max_tokens, verbosity)
            st.session_state["sections"][k] = data

        for k, pressed in btns.items():
            if pressed:
                run_one(k)

        if all_btn:
            keys = list(SECTION_SPECS.keys())
            prog = st.progress(0.0, text="Ejecutando análisis completo…")
            for i, k in enumerate(keys, start=1):
                run_one(k, show_spinner=False)
                prog.progress(i/len(keys), text=f"Ejecutando análisis completo… ({i}/{len(keys)})")
            st.success("Análisis completo finalizado.")

        fs_sections = st.session_state.get("sections", {})
        if fs_sections:
            # Resumen ejecutivo
            resumen_ejecutivo(fs_sections)
            st.divider()

            # --------- Objetivos y contexto ---------
            oc = fs_sections.get("objetivos_contexto", {})
            with st.expander("🎯 Objetivos y contexto", expanded=True):
                st.markdown(f"**Resumen de servicios:** {oc.get('resumen_servicios') or '—'}")
                if oc.get("objetivos"):
                    st.markdown("**Objetivos**:")
                    st.markdown(bullets(oc.get("objetivos") or []))
                st.markdown(f"**Alcance:** {oc.get('alcance') or '—'}")
                if oc.get('referencias_paginas'):
                    st.markdown("**Referencias de páginas:**")
                    st.markdown(bullets([str(x) for x in (oc.get('referencias_paginas') or [])]))
                render_extra_fields("Detalles adicionales (objetivos/contexto)", oc, {'resumen_servicios','objetivos','alcance','referencias_paginas'})

            # --------- Servicios ---------
            svs = fs_sections.get("servicios", {})
            with st.expander("🧩 Servicios solicitados (detalle)"):
                st.markdown(f"**Resumen:** {svs.get('resumen_servicios') or '—'}")
                detalle = svs.get("servicios_detalle") or []
                if detalle:
                    try:
                        import pandas as pd
                        rows = []
                        for d in detalle:
                            rows.append({
                                "Servicio": d.get("nombre"),
                                "Descripción": d.get("descripcion"),
                                "Entregables": ", ".join(d.get("entregables") or []),
                                "SLA/KPI": ", ".join([sk.get("nombre") for sk in (d.get("sla_kpi") or []) if sk.get("nombre")]),
                                "Periodicidad": d.get("periodicidad"),
                                "Volumen": d.get("volumen"),
                                "Ubicación": d.get("ubicacion_modalidad"),
                            })
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    except Exception:
                        for d in detalle:
                            st.markdown(f"- **{d.get('nombre')}** — {d.get('descripcion') or ''}")
                else:
                    st.info("Sin servicios detallados explícitos en el texto analizado.")
                render_extra_fields("Detalles adicionales (servicios)", svs, {"resumen_servicios","servicios_detalle","alcance","referencias_paginas"})

            # --------- Importe ---------
            im = fs_sections.get("importe", {})
            with st.expander("💶 Importe de licitación"):
                imp_total = im.get("importe_total")
                moneda = im.get("moneda") or "EUR"
                st.markdown(f"**Importe total:** {fmt_money(imp_total, moneda)}")
                st.markdown(f"- **IVA incluido:** {im.get('iva_incluido') if im.get('iva_incluido') is not None else '—'}")
                st.markdown(f"- **Tipo IVA:** {im.get('tipo_iva') if im.get('tipo_iva') is not None else '—'}")

                det = im.get("importes_detalle") or []
                if det:
                    try:
                        import pandas as pd
                        rows = []
                        for d in det:
                            per = d.get("periodo") or {}
                            rows.append({
                                "Concepto": d.get("concepto"),
                                "Importe": _to_float(d.get("importe")),
                                "Importe (fmt)": fmt_money(d.get("importe"), d.get("moneda") or moneda),
                                "Moneda": (d.get("moneda") or moneda),
                                "Periodo": per.get("tipo"),
                                "Año": per.get("anyo"),
                                "Duración (meses)": per.get("duracion_meses"),
                                "Observaciones": d.get("observaciones"),
                            })
                        df = pd.DataFrame(rows)
                        if "Año" in df.columns and "Importe" in df.columns:
                            tot = df.groupby("Año", dropna=False)["Importe"].sum().reset_index()
                            tot["Total (fmt)"] = tot["Importe"].apply(lambda x: fmt_money(x, moneda))
                            st.markdown("**Totales por año**")
                            st.dataframe(tot[["Año","Total (fmt)"]], use_container_width=True, hide_index=True)
                        st.markdown("**Detalle de importes**")
                        st.dataframe(df.drop(columns=["Importe"], errors="ignore"), use_container_width=True, hide_index=True)
                    except Exception:
                        for d in det:
                            st.markdown(f"- **{d.get('concepto')}** — {d}")
                render_extra_fields("Detalles adicionales (importe)", im, {"importe_total","moneda","iva_incluido","tipo_iva","importes_detalle"})

            # --------- Criterios ---------
            cv_all = fs_sections.get("criterios_valoracion", {})
            with st.expander("📊 Criterios de valoración"):
                cv = cv_all.get("criterios_valoracion") or []
                if cv:
                    try:
                        import pandas as pd
                        rows = []
                        for c in cv:
                            rows.append({
                                "Criterio": c.get("nombre"),
                                "Peso máx": c.get("peso_max"),
                                "Tipo": c.get("tipo"),
                                "Umbral mín.": c.get("umbral_minimo"),
                                "Método eval.": c.get("metodo_evaluacion"),
                                "Subcriterios": "; ".join([sc.get("nombre") for sc in (c.get("subcriterios") or []) if sc.get("nombre")]),
                            })
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    except Exception:
                        for c in cv:
                            st.markdown(f"- {c.get('nombre')} (peso: {c.get('peso_max')} {c.get('tipo')})")
                    dmp = cv_all.get("criterios_desempate") or []
                    if dmp:
                        st.markdown("**Criterios de desempate:**")
                        st.markdown(bullets(dmp))
                else:
                    st.info("No se encontraron criterios explícitos.")
                render_extra_fields('Detalles adicionales (criterios)', cv_all, {'criterios_valoracion','criterios_desempate'})

            # --------- Índice técnico ---------
            it = fs_sections.get("indice_tecnico", {})
            with st.expander("🗂️ Índice de la respuesta técnica"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Índice solicitado (literal)**")
                    render_index(it.get("indice_respuesta_tecnica") or [])
                with col2:
                    st.markdown("**Índice propuesto (accionable)**")
                    render_index(it.get("indice_propuesto") or [])
                render_extra_fields('Detalles adicionales (índice técnico)', it, {'indice_respuesta_tecnica','indice_propuesto'})

            # --------- Formato y entrega ---------
            fmt = fs_sections.get("formato_oferta", {})
            with st.expander("🧾 Formato y entrega de la oferta"):
                st.markdown(f"**Formato esperado:** {fmt.get('formato_esperado') or '—'}")
                lp = fmt.get("longitud_paginas")
                st.markdown(f"- **Longitud (pág.)**: {lp if isinstance(lp, (int, float)) else '—'}")
                tip = fmt.get("tipografia") or {}
                st.markdown(f"- **Tipografía**: {tip.get('familia') or '—'} / **Tamaño mínimo**: {tip.get('tamano_min') or '—'} / "
                         f"**Interlineado**: {tip.get('interlineado') or '—'} / **Márgenes**: {tip.get('margenes') or '—'}")
                est = fmt.get("estructura_documental") or []
                if est:
                    st.markdown("**Estructura documental requerida/propuesta:**")
                    st.markdown(bullets([x.get("titulo") for x in est if x.get("titulo")]))
                rp = fmt.get("requisitos_presentacion") or []
                if rp:
                    st.markdown("**Requisitos de presentación:**")
                    st.markdown(bullets(rp))
                ra = fmt.get("requisitos_archivo") or {}
                st.markdown(f"- **Formatos permitidos**: {', '.join(ra.get('formatos_permitidos') or []) or '—'}")
                st.markdown(f"- **Tamaño máx (MB)**: {ra.get('tamano_max_mb') if ra.get('tamano_max_mb') is not None else '—'}")
                st.markdown(f"- **Firma digital requerida**: {ra.get('firma_digital_requerida') if ra.get('firma_digital_requerida') is not None else '—'}")
                st.markdown(f"- **Firmado por**: {ra.get('firmado_por') or '—'}")
                st.markdown(f"- **Idioma**: {fmt.get('idioma') or '—'}")
                st.markdown(f"- **Copias**: {fmt.get('copias') if fmt.get('copias') is not None else '—'}")
                ent = fmt.get("entrega") or {}
                st.markdown(f"- **Canal de entrega**: {ent.get('canal') or '—'}")
                st.markdown(f"- **Plazo/Fecha/Hora**: {ent.get('plazo') or '—'} / **Zona horaria**: {ent.get('zona_horaria') or '—'}")
                if ent.get("instrucciones"):
                    st.markdown("**Instrucciones de entrega:**")
                    st.markdown(bullets(ent.get("instrucciones") or []))
                render_extra_fields('Detalles adicionales (formato/entrega)', fmt, {'formato_esperado','longitud_paginas','tipografia','estructura_documental','requisitos_presentacion','requisitos_archivo','idioma','copias','entrega'})

            # --------- Riesgos ---------
            rx = fs_sections.get("riesgos_exclusiones", {})
            with st.expander("⚠️ Riesgos y exclusiones"):
                ry = rx.get("riesgos_y_dudas")
                st.markdown(f"**Riesgos y dudas (síntesis):** {ry or '—'}")
                ex = rx.get("exclusiones") or []
                if ex:
                    st.markdown("**Exclusiones:**")
                    st.markdown(bullets(ex))
                mrx = rx.get("matriz_riesgos") or []
                if mrx:
                    try:
                        import pandas as pd
                        st.caption("Matriz de riesgos (PxI)")
                        st.dataframe(pd.DataFrame(mrx), use_container_width=True, hide_index=True)
                    except Exception:
                        for r in mrx: st.markdown(f"- {r}")
                render_extra_fields('Detalles adicionales (riesgos/exclusiones)', rx, {'riesgos_y_dudas','exclusiones','matriz_riesgos'})

            # --------- Solvencia ---------
            solv_root = fs_sections.get("solvencia", {})
            solv = solv_root.get("solvencia", {})
            with st.expander("📜 Solvencia y acreditación"):
                col1, col2, col3 = st.columns(3)
                tec = solv.get("tecnica", [])
                eco = solv.get("economica", [])
                adm = solv.get("administrativa", [])
                with col1: st.markdown("**Técnica**"); st.markdown(bullets(tec) or "—")
                with col2: st.markdown("**Económica**"); st.markdown(bullets(eco) or "—")
                with col3: st.markdown("**Administrativa**"); st.markdown(bullets(adm) or "—")
                acr = solv.get("acreditacion") or []
                if acr:
                    try:
                        import pandas as pd
                        st.caption("Acreditación (cómo se demuestra)")
                        st.dataframe(pd.DataFrame(acr), use_container_width=True, hide_index=True)
                    except Exception:
                        for a in acr: st.markdown(f"- {a}")
                render_extra_fields('Detalles adicionales (solvencia)', solv_root, {'solvencia'})
        else:
            st.info("Realiza un análisis para ver resultados.")

    # ----- Tab 1: Logs -----
    with tabs[1]:
        st.subheader("Logs de prompts y respuestas")
        logs = st.session_state.get("logs", {})
        if not any(logs.values()):
            st.info("Aún no hay logs. Ejecuta alguna sección.")
        else:
            for k, spec in SECTION_SPECS.items():
                entries = logs.get(k, [])
                with st.expander(f"Sección: {spec['titulo']} ({len(entries)} registros)", expanded=False):
                    for i, e in enumerate(entries, start=1):
                        st.caption(f"[{e.get('ts')}] model={e.get('model')} temp={e.get('temperature')} mode={e.get('mode')}")
                        with st.popover(f"Prompt #{i}"):
                            st.code(e.get("prompt") or "", language="markdown")
                        with st.popover(f"Respuesta #{i}"):
                            st.code((e.get("response") or "")[:20000], language="json")

if __name__ == "__main__":
    main()
