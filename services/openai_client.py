
# services/openai_client.py
from typing import Any, Dict, List, Optional
from openai import OpenAI, BadRequestError

def create_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

def _is_temperature_error(msg: str) -> bool:
    return ("temperature" in msg and ("Unsupported value" in msg or "unsupported_value" in msg or "does not support" in msg))

def _is_unsupported_param(msg: str, param: str) -> bool:
    return (("unsupported_parameter" in msg or "Unknown parameter" in msg or "unexpected keyword" in msg) and (param in msg))

def responses_create_robust(client: OpenAI, args: Dict[str, Any]):
    a = dict(args)
    for _ in range(5):
        try:
            return client.responses.create(**a)
        except (BadRequestError, TypeError) as e:
            s = str(e)
            if _is_temperature_error(s):
                a.pop("temperature", None); continue
            if _is_unsupported_param(s, "response_format"):
                a.pop("response_format", None); continue
            if _is_unsupported_param(s, "max_output_tokens"):
                val = a.pop("max_output_tokens", None)
                if val is not None: a["max_completion_tokens"] = val
                continue
            raise

def coalesce_text_from_responses(rsp) -> Optional[str]:
    txt = getattr(rsp, "output_text", None)
    if txt: return txt
    out = getattr(rsp, "output", None)
    if out:
        parts = []
        for item in out:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for block in content:
                    val = getattr(block, "text", None)
                    if val: parts.append(val)
                    elif isinstance(block, dict):
                        v = block.get("text") or block.get("value")
                        if v: parts.append(v)
        if parts: return "\n".join(parts)
    msg = getattr(rsp, "message", None)
    if msg and isinstance(getattr(msg, "content", None), list):
        parts = []
        for block in msg.content:
            val = getattr(block, "text", None)
            if val: parts.append(val)
            elif isinstance(block, dict):
                v = block.get("text") or block.get("value")
                if v: parts.append(v)
        if parts: return "\n".join(parts)
    return None
