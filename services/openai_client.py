import json
import re
from pydantic import ValidationError
from services.schema import OfertaAnalizada
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI, BadRequestError
from pydantic import ValidationError

from services.schema import OfertaAnalizada
from services.prompts import SYSTEM_PROMPT, USER_PROMPT
from config import OPENAI_API_KEY, OPENAI_MODEL, MAX_TOKENS_PER_REQUEST
DEFAULT_TEMPERATURE = 0.1  # usado sólo si el modelo lo soporta

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY no está configurada. Define la clave en `.env`, variables de entorno o st.secrets."
    )

client = OpenAI(api_key=OPENAI_API_KEY)

def _is_temperature_error(e: Exception) -> bool:
    s = str(e)
    return ("temperature" in s) and ("Unsupported value" in s or "unsupported_value" in s or "does not support" in s)

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r'^\s*```(?:json)?\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s*```\s*$', '', s)
    return s

def _coalesce_text_from_responses(rsp) -> str | None:
    """
    Intenta extraer texto de múltiples formas del objeto devuelto por la Responses API,
    compatible con variaciones de SDK (output_text, output[*].content[*].text, choices...).
    """
    # 1) SDKs recientes
    txt = getattr(rsp, "output_text", None)
    if txt:
        return txt

    # 2) Recorrer bloques de salida
    out = getattr(rsp, "output", None)
    if out:
        texts = []
        for item in out:
            # item.content suele ser una lista de bloques con .type y .text
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for block in content:
                    # Algunas versiones usan .text; otras .value o dict-like
                    val = getattr(block, "text", None)
                    if val:
                        texts.append(val)
                    else:
                        # acceso dict-like
                        if isinstance(block, dict):
                            t = block.get("text") or block.get("value")
                            if t:
                                texts.append(t)
        if texts:
            return "\n".join(texts)

    # 3) Algunos SDKs dejan contenido en choices[0].message.content
    choices = getattr(rsp, "choices", None)
    if choices:
        msg = getattr(choices[0], "message", None)
        if msg is not None:
            content = getattr(msg, "content", None)
            if content:
                return content

    return None

def _coalesce_text_from_chat(rsp) -> str | None:
    """
    Extrae contenido de Chat Completions robustamente.
    """
    choices = getattr(rsp, "choices", None)
    if not choices:
        return None
    msg = getattr(choices[0], "message", None)
    if msg is None:
        return None
    content = getattr(msg, "content", None)
    if content:
        return content
    # Últimos intentos por si el SDK usa dicts internos raros
    if isinstance(msg, dict):
        return msg.get("content")
    return None

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


@retry(wait=wait_exponential(multiplier=1, min=2, max=8), stop=stop_after_attempt(3))
def _call_openai(model: Optional[str], system: str, user: str) -> str:
    model = model or OPENAI_MODEL

    # ---------- 1) Responses API (preferida) ----------
    try:
        args = dict(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            max_output_tokens=MAX_TOKENS_PER_REQUEST,
            temperature=DEFAULT_TEMPERATURE,
        )
        try:
            rsp = client.responses.create(**args)
        except BadRequestError as e:
            if _is_temperature_error(e):
                # Reintentar sin temperature con Responses
                args.pop("temperature", None)
                rsp = client.responses.create(**args)
            else:
                raise
        try:
            return rsp.output_text
        except Exception:
            if hasattr(rsp, "choices") and rsp.choices:
                return rsp.choices[0].message.content
            return json.dumps(rsp)
    except (TypeError, AttributeError, BadRequestError):
        # SDK antiguo o modelo que no soporta Responses/response_format -> fallback
        pass

    # ---------- 2) Chat Completions (fallback robusto) ----------
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    token_params_order = ("max_output_tokens", "max_completion_tokens", "max_tokens")
    response_format_options = (True, False)
    last_error = None

    for use_rf in response_format_options:
        for token_param in token_params_order:
            # 2.a) Intento con temperature
            kwargs = dict(
                model=model,
                messages=messages,
                temperature=DEFAULT_TEMPERATURE,
            )
            kwargs[token_param] = MAX_TOKENS_PER_REQUEST
            if use_rf:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                rsp = client.chat.completions.create(**kwargs)
                content = rsp.choices[0].message.content
                return content if use_rf else _extract_json(content)
            except BadRequestError as e:
                # Si el problema es 'temperature', reintentamos la MISMA combinación sin temperature
                if _is_temperature_error(e):
                    try:
                        kwargs_no_temp = dict(kwargs)
                        kwargs_no_temp.pop("temperature", None)
                        rsp = client.chat.completions.create(**kwargs_no_temp)
                        content = rsp.choices[0].message.content
                        return content if use_rf else _extract_json(content)
                    except BadRequestError as e2:
                        # Si ahora es otro unsupported_parameter (p.ej. max_tokens), seguimos iterando
                        msg2 = str(e2)
                        if "unsupported_parameter" in msg2 or "Unsupported parameter" in msg2:
                            last_error = e2
                            continue
                        raise
                # No es error de temperature: si es unsupported_parameter del token param, probar siguiente
                msg = str(e)
                if "unsupported_parameter" in msg or "Unsupported parameter" in msg:
                    last_error = e
                    continue
                # Otros 400: propagar
                raise
            except TypeError as e:
                # Firma del SDK no admite ese kwargs -> probar siguiente
                last_error = e
                continue

    raise RuntimeError(
        "No fue posible realizar la llamada a OpenAI con los parámetros disponibles. "
        "Actualiza `openai` (>=1.43) o ajusta el modelo en la barra lateral. "
        f"Último error: {last_error!r}"
    )


def analyze_text_chunk(accumulated: Optional[OfertaAnalizada], chunk_text: str, model: Optional[str] = None) -> OfertaAnalizada:
    user = USER_PROMPT.format(doc_text=chunk_text)
    raw = _call_openai(model, SYSTEM_PROMPT, user)

    data = _loads_json_robust(raw)  # <-- parseo unificado y robusto

    try:
        parsed = OfertaAnalizada.model_validate(data)
    except ValidationError as e:
        # Deja el error claro (incluye un snippet de la salida para depurar)
        snippet = (raw or "")[:400] if isinstance(raw, str) else str(raw)[:400]
        raise ValueError(f"La salida del modelo no cumple el esquema. Muestra (primeros 400 chars): {snippet}") from e

    if accumulated is None:
        return parsed
    return merge_offers(accumulated, parsed)


def merge_offers(base: OfertaAnalizada, new: OfertaAnalizada) -> OfertaAnalizada:
    from copy import deepcopy

    out = deepcopy(base)
    # Resumen y objetivos
    if len((new.resumen_servicios or "")) > len((out.resumen_servicios or "")):
        out.resumen_servicios = new.resumen_servicios
    if new.objetivos:
        seen = set(out.objetivos)
        for o in new.objetivos:
            if o not in seen:
                out.objetivos.append(o)
                seen.add(o)
    out.alcance = out.alcance or new.alcance

    # Importes
    out.importe_total = new.importe_total or out.importe_total
    out.moneda = new.moneda or out.moneda
    seen_imp = {(d.concepto, d.importe, d.moneda) for d in out.importes_detalle}
    for d in new.importes_detalle:
        key = (d.concepto, d.importe, d.moneda)
        if key not in seen_imp:
            out.importes_detalle.append(d)
            seen_imp.add(key)

    # Criterios
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

    # Índices
    titles_sol = {s.titulo: s for s in out.indice_respuesta_tecnica}
    for s in new.indice_respuesta_tecnica:
        if s.titulo in titles_sol:
            ex = titles_sol[s.titulo]
            ex.descripcion = ex.descripcion or s.descripcion
            subs = set(ex.subapartados)
            for sub in s.subapartados:
                if sub not in subs:
                    ex.subapartados.append(sub)
        else:
            out.indice_respuesta_tecnica.append(s)

    titles_prop = {s.titulo: s for s in out.indice_propuesto}
    for s in new.indice_propuesto:
        if s.titulo in titles_prop:
            ex = titles_prop[s.titulo]
            ex.descripcion = ex.descripcion or s.descripcion
            subs = set(ex.subapartados)
            for sub in s.subapartados:
                if sub not in subs:
                    ex.subapartados.append(sub)
        else:
            out.indice_propuesto.append(s)

    # Riesgos
    if new.riesgos_y_dudas and (
        not out.riesgos_y_dudas or len(new.riesgos_y_dudas) > len(out.riesgos_y_dudas)
    ):
        out.riesgos_y_dudas = new.riesgos_y_dudas

    # Páginas
    pages = set(out.referencias_paginas)
    for p in new.referencias_paginas:
        pages.add(p)
    out.referencias_paginas = sorted(pages)

    return out
