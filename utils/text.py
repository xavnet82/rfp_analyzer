
# utils/text.py
import re
from typing import List

def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\r?\n\s*\r?\n", "\n\n", s)
    return s.strip()

def bullets(items: List[str]) -> str:
    return "\n".join([f"- {x}" for x in items if str(x).strip()])
