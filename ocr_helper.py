\
"""
Helper to detect system Tesseract availability in Streamlit and provide friendly guidance.
"""
import shutil
import os

def tesseract_available() -> bool:
    # Looks for the 'tesseract' binary in PATH. On Windows, pytesseract may need a full path.
    exe = shutil.which("tesseract")
    return exe is not None

def configure_pytesseract_for_windows():
    """
    If running on Windows in Streamlit, users may need to set pytesseract.pytesseract.tesseract_cmd
    to the installed path, e.g. r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    This is a no-op unless the env var TESSERACT_CMD is set.
    """
    cmd = os.getenv("TESSERACT_CMD")
    if cmd:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = cmd
