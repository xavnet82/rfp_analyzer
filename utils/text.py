import re
from typing import List

def clean_text(s: str) -> str:
    s = s.replace('\x00', ' ')
    s = re.sub(r'[\r\t]', ' ', s)
    s = re.sub(r' +', ' ', s)
    return s.strip()

def chunk_text(pages: List[str], max_chars: int = 12000) -> List[str]:
    chunks, buf = [], ''
    for i, p in enumerate(pages, 1):
        if len(buf) + len(p) > max_chars and buf:
            chunks.append(buf); buf = p
        else:
            buf += ('\n\n' if buf else '') + p
    if buf:
        chunks.append(buf)
    return chunks
