from typing import List, Tuple
from pypdf import PdfReader

def extract_pdf_text(path_or_buf) -> Tuple[List[str], str]:
    reader = PdfReader(path_or_buf)
    pages_text = []
    for _i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        pages_text.append(t)
    full_text = "\n\n".join(pages_text)
    return pages_text, full_text
