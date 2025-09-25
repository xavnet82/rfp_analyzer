import os
from dotenv import load_dotenv

# 1) .env
load_dotenv()

# 2) st.secrets (si existe)
try:
    import streamlit as st
    _secrets = st.secrets if hasattr(st, "secrets") else {}
except Exception:
    _secrets = {}

def _get(key: str, default: str = "") -> str:
    return os.getenv(key) or _secrets.get(key, default)

OPENAI_API_KEY = _get("OPENAI_API_KEY", "")
OPENAI_MODEL = _get("OPENAI_MODEL", "gpt-4o-mini")
MAX_TOKENS_PER_REQUEST = int(_get("MAX_TOKENS_PER_REQUEST", "2000"))
APP_TITLE = "Análisis de Pliegos – Licitaciones"

# Credenciales básicas (solo demostración; no usar tal cual en producción)
ADMIN_USER = _get("ADMIN_USER", "admin")
ADMIN_PASSWORD = _get("ADMIN_PASSWORD", "rfpanalyzer")
