from __future__ import annotations
import json
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
from pydantic import ValidationError
from .schema import OfertaAnalizada
from .prompts import SYSTEM_PROMPT, USER_PROMPT
from ..config import OPENAI_API_KEY, OPENAI_MODEL, MAX_TOKENS_PER_REQUEST

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

@retry(wait=wait_exponential(multiplier=1, min=2, max=8), stop=stop_after_attempt(3))
def _call_openai(model: Optional[str], system: str, user: str) -> str:
    assert client is not None, "Falta OPENAI_API_KEY"
    model = model or OPENAI_MODEL
    # Usamos Responses API para mayor robustez
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
    # Unificamos acceso a texto
    try:
        return rsp.output_text
    except Exception:
        # Fallback si cambia la estructura
        return json.dumps(rsp.dict())
    
def analyze_text_chunk(accumulated: Optional[OfertaAnalizada], chunk_text: str, model: Optional[str] = None) -> OfertaAnalizada:
    user = USER_PROMPT.format(doc_text=chunk_text)
    raw = _call_openai(model, SYSTEM_PROMPT, user)
    data = json.loads(raw)
    try:
        parsed = OfertaAnalizada.model_validate(data)
    except ValidationError as e:
        # Intento de reparación trivial: si viene como string, parsear de nuevo
        if isinstance(data, str):
            parsed = OfertaAnalizada.model_validate(json.loads(data))
        else:
            raise e
    if accumulated is None:
        return parsed
    # Merge sencillo y determinista
    return merge_offers(accumulated, parsed)

def merge_offers(base: OfertaAnalizada, new: OfertaAnalizada) -> OfertaAnalizada:
    from copy import deepcopy
    out = deepcopy(base)
    # Reglas de merge: preferir campos no vacíos; concatenar listas con de-duplicado por clave
    if len(new.resumen_servicios) > len(out.resumen_servicios):
        out.resumen_servicios = new.resumen_servicios
    out.importe_total = new.importe_total or out.importe_total
    out.moneda = new.moneda or out.moneda
    # importes_detalle (dedupe por (concepto, importe, moneda))
    seen = {(d.concepto, d.importe, d.moneda) for d in out.importes_detalle}
    for d in new.importes_detalle:
        key = (d.concepto, d.importe, d.moneda)
        if key not in seen:
            out.importes_detalle.append(d)
            seen.add(key)
    # criterios (dedupe por nombre)
    existing = {c.nombre: c for c in out.criterios_valoracion}
    for c in new.criterios_valoracion:
        if c.nombre in existing:
            # merge subcriterios por nombre
            ex = existing[c.nombre]
            ex.peso_max = ex.peso_max or c.peso_max
            ex.tipo = ex.tipo or c.tipo
            names = {s.nombre for s in ex.subcriterios}
            for s in c.subcriterios:
                if s.nombre not in names:
                    ex.subcriterios.append(s)
        else:
            out.criterios_valoracion.append(c)
    # índice (dedupe por título)
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
    # riesgos
    if new.riesgos_y_dudas and (not out.riesgos_y_dudas or len(new.riesgos_y_dudas) > len(out.riesgos_y_dudas)):
        out.riesgos_y_dudas = new.riesgos_y_dudas
    # páginas
    pages = set(out.referencias_paginas)
    for p in new.referencias_paginas:
        pages.add(p)
    out.referencias_paginas = sorted(pages)
    return out
