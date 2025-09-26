# services/file_search_client.py
from typing import List, Optional
from openai import OpenAI, BadRequestError
from tenacity import retry, wait_exponential, stop_after_attempt
import json

from config import OPENAI_API_KEY, OPENAI_MODEL, MAX_TOKENS_PER_REQUEST
from services.prompts import SYSTEM_PROMPT, USER_PROMPT

client = OpenAI(api_key=OPENAI_API_KEY)

def _is_temperature_error(e: Exception) -> bool:
    s = str(e)
    return ("temperature" in s) and ("Unsupported value" in s or "unsupported_value" in s or "does not support" in s)

def _is_unsupported_param(e: Exception, param: str) -> bool:
    s = str(e)
    return ("unsupported_parameter" in s or "Unexpected" in s or "unexpected" in s) and (param in s)

def _extract_json(text: str) -> str:
    import re
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise RuntimeError("La respuesta del modelo no contiene JSON parseable.")
    return m.group(0)

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
                        if v: parts.append(v)
        if parts:
            return "\n".join(parts)
    return None

def create_vector_store_from_streamlit_files(files, name: str = "RFP Vector Store"):
    """
    files: lista de st.uploaded_file o equivalentes (con .name y .read()).
    Devuelve el id del vector store con todos los ficheros indexados.
    """
    # 1) Crear vector store (puedes ajustar chunking_strategy si quieres)
    store = client.vector_stores.create(
        name=name,
        # opcional: caducidad automática para evitar costes de almacenamiento
        expires_after={"anchor": "last_active_at", "days": 2},
        # opcional: chunking_strategy para PDFs muy largos
        # chunking_strategy={"type": "static", "static": {"max_chunk_size_tokens": 1600, "chunk_overlap_tokens": 400}},
    )

    # 2) Subir cada PDF y añadirlo al vector store
    for f in files:
        data = f.read()
        up = client.files.create(
            file=(f.name, data, "application/pdf"),
            purpose="assistants"  # requerido para vector stores
        )
        # Espera a que se procese el fichero dentro del store
        client.vector_stores.files.create_and_poll(
            vector_store_id=store.id,
            file_id=up.id
        )
    return store.id

@retry(wait=wait_exponential(multiplier=1, min=2, max=8),
       stop=stop_after_attempt(3),
       reraise=True)
def analyze_with_file_search(
    vector_store_id: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None
) -> str:
    """
    Lanza UNA sola llamada a Responses API + file_search para obtener el JSON completo.
    Devuelve una cadena JSON.
    """
    model = (model or OPENAI_MODEL).strip()
    is_gpt5 = model.lower().startswith("gpt-5")

    # Construir argumentos base de Responses + herramienta file_search
    args = dict(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(doc_text="(usar los archivos adjuntos)")}
        ],
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
        response_format={"type": "json_object"},
        max_output_tokens=MAX_TOKENS_PER_REQUEST,
    )
    if temperature is not None:
        args["temperature"] = float(temperature)

    # Modelos gpt-5*: suelen rechazar temperature != 1 y, en ciertos SDKs, response_format
    if is_gpt5:
        args.pop("temperature", None)
        args.pop("response_format", None)

    try:
        rsp = client.responses.create(**args)
    except BadRequestError as e:
        # limpiamos parámetros problemáticos y reintentamos
        s = str(e)
        if _is_temperature_error(e):
            args.pop("temperature", None)
            rsp = client.responses.create(**args)
        elif _is_unsupported_param(e, "response_format"):
            args.pop("response_format", None)
            rsp = client.responses.create(**args)
        elif _is_unsupported_param(e, "max_output_tokens"):
            val = args.pop("max_output_tokens", None)
            if val is not None:
                args["max_completion_tokens"] = val
            rsp = client.responses.create(**args)
        else:
            raise

    text = _coalesce_text_from_responses(rsp)
    if not text:
        dump = json.dumps(rsp, default=str)
        return _extract_json(dump)
    return text
