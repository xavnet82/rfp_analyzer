import os
from dotenv import load_dotenv

load_dotenv()

try:
    import streamlit as st
    _secrets = st.secrets if hasattr(st, "secrets") else {}
except Exception:
    _secrets = {}

def _get(key: str, default: str = "") -> str:
    return os.getenv(key) or _secrets.get(key, default)

OPENAI_API_KEY = _get("OPENAI_API_KEY", "")
OPENAI_MODEL = _get("OPENAI_MODEL", "gpt-4o-mini")
APP_TITLE = "Análisis de Pliegos – Licitaciones"
try:
    OPENAI_TEMPERATURE = float(_get("OPENAI_TEMPERATURE", "1.0"))
except Exception:
    OPENAI_TEMPERATURE = 1.0
try:
    MAX_TOKENS_PER_REQUEST = int(_get("MAX_TOKENS_PER_REQUEST", "2000"))
except Exception:
    MAX_TOKENS_PER_REQUEST = 2000

# Login (demo)
ADMIN_USER = _get("ADMIN_USER", "admin")
ADMIN_PASSWORD = _get("ADMIN_PASSWORD", "rfpanalyzer")

# Catálogo de modelos para el desplegable
_MODELS_CSV = _get("MODELS_CATALOG", "gpt-5-mini,gpt-4o,gpt-4o-mini,gpt-4.1,gpt-4.1-mini")
MODELS_CATALOG = [m.strip() for m in _MODELS_CSV.split(",") if m.strip()]
