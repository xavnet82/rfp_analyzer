
# services/pdf_loader.py
from typing import List, Tuple
import io

def extract_pdf_text(file_like: io.BytesIO) -> Tuple[List[str], str]:
    try:
        from PyPDF2 import PdfReader  # type: ignore
        reader = PdfReader(file_like)
        pages = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            pages.append(t)
        return pages, "\n".join(pages)
    except Exception as e:
        raise RuntimeError(f"Fallo al parsear PDF: {e}")
