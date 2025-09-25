from __future__ import annotations
import json
import re
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
from pydantic import ValidationError
from services.schema import OfertaAnalizada
from services.prompts import SYSTEM_PROMPT, USER_PROMPT
from config import OPENAI_API_KEY, OPENAI_MODEL, MAX_TOKENS_PER_REQUEST

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY no está configurada. Define la clave en `.env`, variables de entorno o st.secrets."
    )

client = OpenAI(api_key=OPENAI_API_KEY)

def _extract_json(text: str) -> str:
    """
    Extrae el primer bloque JSON de un texto.
    Fallback para clientes que no soportan response_format.
    """
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise RuntimeError("La respuesta del modelo no contiene JSON parseable.")
    return m.group(0)

@retry(wait=wait_exponential(multiplier=1, min=2, max=8), stop=stop_after_attempt(3))
def _call_openai(model: Optional[str], system: str, user: str) -> str:
    model = model or OPENAI_MODEL

    # 1) Intento con API Responses + response_format (si el cliente lo soporta)
    try:
        rsp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_output_tokens=MAX_TOKENS_PER_REQUEST,
            response_format={"type": "json_object"},
        )
        # Unificar acceso a texto
        try:
            return rsp.output_text  # clientes recientes
        except Exception:
            # Algunos clientes devuelven otra forma
            if hasattr(rsp, "choices") and rsp.choices:
                return rsp.choices[0].message.content
            return json.dumps(rsp)
    except (TypeError, AttributeError):
        # 2) Fallback a Chat Completions con response_format si está disponible
        try:
            rsp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.1,
                max_tokens=MAX_TOKENS_PER_REQUEST,
                response_format={"type": "json_object"},
            )
            return rsp.choices[0].message.content
        except TypeError:
            # 3) Cliente antiguo: sin response_format
            rsp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.1,
                max_tokens=MAX_TOKENS_PER_REQUEST,
            )
            text = rsp.choices[0].message.content
            return _extract_json(text)

def analyze_text_chunk(accumulated: Optional[OfertaAnalizada], chunk_text: str, model: Optional[str] = None) -> OfertaAnalizada:
    user = USER_PROMPT.format(doc_text=chunk_text)
    raw = _call_openai(model, SYSTEM_PROMPT, user)
    data = json.loads(raw)
    try:
        parsed = OfertaAnalizada.model_validate(data)
    except ValidationError as e:
        # Intento de reparación si vino string con JSON embebido
        if isinstance(data, str):
            parsed = OfertaAnalizada.model_validate(json.loads(data))
        else:
            raise e
    if accumulated is None:
        return parsed
    return merge_offers(accumulated, parsed)

def merge_offers(base: OfertaAnalizada, new: OfertaAnalizada) -> OfertaAnalizada:
    from copy import deepcopy
    out = deepcopy(base)
    # Merge con reglas deterministas
    if len(new.resumen_servicios) > len(out.resumen_servicios):
        out.resumen_servicios = new.resumen_servicios
    out.importe_total = new.importe_total or out.importe_total
    out.moneda = new.moneda or out.moneda

    seen = {(d.concepto, d.importe, d.moneda) for d in out.importes_detalle}
    for d in new.importes_detalle:
        key = (d.concepto, d.importe, d.moneda)
        if key not in seen:
            out.importes_detalle.append(d)
            seen.add(key)

    existing = {c.nombre: c for c in out.criterios_valoracion}
    for c in new.criterios_valoracion:
        if c.nombre in existing:
            ex = existing[c.nombre]
            ex.peso_max = ex.peso_max or c.peso_max
            ex.tipo = ex.tipo or c.tipo
            names = {s.nombre for s in ex.subcriterios}
            for s in c.subcriterios:
                if s.nombre not in names:
                    ex.subcriterios.append(s)
        else:
            out.criterios_valoracion.append(c)

    titles = {s.titulo: s for s in out.indice_respuesta_tecnica}
    for s in new.indice_respuesta_tecnica:
        if s.titulo in titles:
            ex = titles[s.titulo]
            ex.descripcion = ex.descripcion or s.descripcion
            subs = set(ex.subapartados)
            for sub in s.subapartados:
                if sub not in subs:
                    ex.subapartados.append(sub)
        else:
            out.indice_respuesta_tecnica.append(s)

    if new.riesgos_y_dudas and (not out.riesgos_y_dudas or len(new.riesgos_y_dudas) > len(out.riesgos_y_dudas)):
        out.riesgos_y_dudas = new.riesgos_y_dudas

    pages = set(out.referencias_paginas)
    for p in new.referencias_paginas:
        pages.add(p)
    out.referencias_paginas = sorted(pages)
    return out

