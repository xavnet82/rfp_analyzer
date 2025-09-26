import json
import re
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI, BadRequestError
from pydantic import ValidationError

from services.schema import OfertaAnalizada
from services.prompts import SYSTEM_PROMPT, USER_PROMPT
from config import OPENAI_API_KEY, OPENAI_MODEL, MAX_TOKENS_PER_REQUEST

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY no está configurada.")

client = OpenAI(api_key=OPENAI_API_KEY)

DEFAULT_TEMPERATURE = 0.5  # se ignora si el modelo no lo soporta

def _coalesce_text_from_responses(rsp) -> Optional[str]:
    txt = getattr(rsp, "output_text", None)
    if txt:
        return txt
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
    choices = getattr(rsp, "choices", None)
    if choices:
        msg = getattr(choices[0], "message", None)
        if msg is not None:
            content = getattr(msg, "content", None)
            if content:
                return content
    return None

def _coalesce_text_from_chat(rsp) -> Optional[str]:
    choices = getattr(rsp, "choices", None)
    if not choices:
        return None
    msg = getattr(choices[0], "message", None)
    if msg is None:
        return None
    content = getattr(msg, "content", None)
    return content

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r'^\s*```(?:json)?\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s*```\s*$', '', s)
    return s

def _extract_json(text: str) -> str:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise RuntimeError("La respuesta del modelo no contiene JSON parseable.")
    return m.group(0)

def _loads_json_robust(raw):
    if raw is None:
        raise RuntimeError("Respuesta vacía del modelo.")
    if not isinstance(raw, str):
        return raw
    s = _strip_code_fences(raw)
    if not s:
        raise RuntimeError("El modelo devolvió cadena vacía.")
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        brace = _extract_json(s)
        return json.loads(brace)
    if isinstance(obj, str):
        inner = _strip_code_fences(obj)
        if not inner:
            raise RuntimeError("El modelo devolvió cadena vacía tras decodificar.")
        if inner.startswith("{") or inner.startswith("["):
            return json.loads(inner)
        brace = _extract_json(inner)
        return json.loads(brace)
    return obj

def _is_temperature_error(e: Exception) -> bool:
    s = str(e)
    return ("temperature" in s) and ("Unsupported value" in s or "unsupported_value" in s or "does not support" in s)

def _is_unsupported_param(e: Exception, param: str) -> bool:
    s = str(e)
    return ("unsupported_parameter" in s or "Unexpected" in s or "unexpected" in s) and (param in s)

def _responses_create_robust(args: dict):
    a = dict(args)
    for _ in range(5):
        try:
            return client.responses.create(**a)
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

@retry(wait=wait_exponential(multiplier=1, min=2, max=8),
       stop=stop_after_attempt(3),
       reraise=True)
def _call_openai(model: Optional[str], system: str, user: str, temperature: Optional[float] = None) -> str:
    model = (model or OPENAI_MODEL).strip()
    is_gpt5 = model.lower().startswith("gpt-5")

    # 1) Responses API (preferida; obligatoria para gpt-5*)
    eff_temp = DEFAULT_TEMPERATURE if temperature is None else float(temperature)
    args = dict(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_output_tokens=MAX_TOKENS_PER_REQUEST,
        temperature=eff_temp,
        response_format={"type": "json_object"},
    )
    if is_gpt5:
        # Muchos despliegues de gpt-5* no aceptan temperature != 1 ni response_format
        args.pop("temperature", None)
        args.pop("response_format", None)

    try:
        rsp = _responses_create_robust(args)
        text = _coalesce_text_from_responses(rsp)
        if not text:
            dump = json.dumps(rsp, default=str)
            try:
                return _extract_json(dump)
            except Exception:
                raise RuntimeError("Responses API devolvió salida sin texto utilizable.")
        return text
    except BadRequestError:
        if is_gpt5:
            raise

    # 2) Chat Completions (solo si NO es gpt-5*)
    from config import MAX_TOKENS_PER_REQUEST as TOK
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    for use_rf in (True, False):
        for token_param in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
            kwargs = dict(model=model, messages=messages)
            if temperature is not None:
                kwargs["temperature"] = float(temperature)
            kwargs[token_param] = TOK
            if use_rf:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                rsp = client.chat.completions.create(**kwargs)
                content = _coalesce_text_from_chat(rsp)
                if not content:
                    if "temperature" in kwargs:
                        kwargs_no_temp = dict(kwargs); kwargs_no_temp.pop("temperature", None)
                        rsp = client.chat.completions.create(**kwargs_no_temp)
                        content = _coalesce_text_from_chat(rsp)
                        if not content:
                            raise RuntimeError("Chat Completions sin temperature devolvió salida sin texto utilizable.")
                return content if use_rf else _extract_json(content)
            except BadRequestError as e:
                if _is_temperature_error(e):
                    try:
                        kwargs_no_temp = dict(kwargs); kwargs_no_temp.pop("temperature", None)
                        rsp = client.chat.completions.create(**kwargs_no_temp)
                        content = _coalesce_text_from_chat(rsp)
                        if not content:
                            raise RuntimeError("Chat Completions sin temperature devolvió salida sin texto utilizable.")
                        return content if use_rf else _extract_json(content)
                    except BadRequestError:
                        continue
                if _is_unsupported_param(e, token_param) or _is_unsupported_param(e, "response_format"):
                    continue
                raise

def analyze_text_chunk(accumulated: Optional[OfertaAnalizada],
                       chunk_text: str,
                       model: Optional[str] = None,
                       temperature: Optional[float] = None) -> OfertaAnalizada:
    user = USER_PROMPT.format(doc_text=chunk_text)
    raw = _call_openai(model, SYSTEM_PROMPT, user, temperature=temperature)
    data = _loads_json_robust(raw)
    parsed = OfertaAnalizada.model_validate(data)
    if accumulated is None:
        return parsed
    return merge_offers(accumulated, parsed)

def merge_offers(base: OfertaAnalizada, new: OfertaAnalizada) -> OfertaAnalizada:
    from copy import deepcopy
    out = deepcopy(base)

    if len((new.resumen_servicios or "")) > len((out.resumen_servicios or "")):
        out.resumen_servicios = new.resumen_servicios
    if new.objetivos:
        seen = set(out.objetivos)
        for o in new.objetivos:
            if o not in seen:
                out.objetivos.append(o); seen.add(o)
    out.alcance = out.alcance or new.alcance

    out.importe_total = new.importe_total or out.importe_total
    out.moneda = new.moneda or out.moneda
    seen_imp = {(d.concepto, d.importe, d.moneda) for d in (out.importes_detalle or [])}
    for d in (new.importes_detalle or []):
        key = (d.concepto, d.importe, d.moneda)
        if key not in seen_imp:
            out.importes_detalle.append(d); seen_imp.add(key)

    existing = {c.nombre: c for c in (out.criterios_valoracion or [])}
    for c in (new.criterios_valoracion or []):
        if c.nombre in existing:
            ex = existing[c.nombre]
            ex.peso_max = ex.peso_max or c.peso_max
            ex.tipo = ex.tipo or c.tipo
            names = {s.nombre for s in (ex.subcriterios or [])}
            for s in (c.subcriterios or []):
                if s.nombre not in names:
                    ex.subcriterios.append(s)
        else:
            out.criterios_valoracion.append(c)

    titles_sol = {s.titulo: s for s in (out.indice_respuesta_tecnica or [])}
    for s in (new.indice_respuesta_tecnica or []):
        if s.titulo in titles_sol:
            ex = titles_sol[s.titulo]
            ex.descripcion = ex.descripcion or s.descripcion
            subs = set(ex.subapartados or [])
            for sub in (s.subapartados or []):
                if sub not in subs:
                    ex.subapartados.append(sub)
        else:
            out.indice_respuesta_tecnica.append(s)

    titles_prop = {s.titulo: s for s in (out.indice_propuesto or [])}
    for s in (new.indice_propuesto or []):
        if s.titulo in titles_prop:
            ex = titles_prop[s.titulo]
            ex.descripcion = ex.descripcion or s.descripcion
            subs = set(ex.subapartados or [])
            for sub in (s.subapartados or []):
                if sub not in subs:
                    ex.subapartados.append(sub)
        else:
            out.indice_propuesto.append(s)

    if new.riesgos_y_dudas and (not out.riesgos_y_dudas or len(new.riesgos_y_dudas) > len(out.riesgos_y_dudas)):
        out.riesgos_y_dudas = new.riesgos_y_dudas

    pages = set(out.referencias_paginas or [])
    for p in (new.referencias_paginas or []):
        pages.add(p)
    out.referencias_paginas = sorted(pages)

    return out
