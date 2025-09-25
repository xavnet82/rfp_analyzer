# App de análisis de pliegos (Streamlit + OpenAI) — Multi-fichero

Analiza uno o **varios** PDFs de un pliego (pliego general, anexos, etc.) y devuelve salida **estructurada**:
- Resumen detallado de servicios + **Objetivos** y **Alcance**.
- Importe total, detalle por lotes/renovaciones y moneda.
- Criterios de valoración y ponderaciones.
- **Índice solicitado (literal del pliego)** frente a **Índice propuesto (alineado)**.
- Exportación a JSON y Markdown para **cada fichero** y **agregado**.

## Requisitos
- Python 3.10+
- Clave de API de OpenAI (variable de entorno `OPENAI_API_KEY` o `st.secrets`).

## Instalación
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # y edita la clave/modelo
```

## Ejecución
```bash
streamlit run app.py
```

## Estructura
```
proyecto_analisis_pliegos/
├── app.py
├── config.py
├── requirements.txt
├── .env.example
├── services/
│   ├── openai_client.py
│   ├── pdf_loader.py
│   ├── prompts.py
│   └── schema.py
├── components/
│   └── ui.py
├── utils/
│   └── text.py
└── tests/
    └── test_schema.py
```

## Notas técnicas
- `pypdf` para extracción de texto. Añade OCR si el PDF es escaneado (p.ej. `pytesseract`). 
- **Multi-fichero**: agrega resultados por fichero y muestra visión **agregada**.
- `pydantic` valida el JSON. `tenacity` reintenta llamadas al LLM.
- Cliente OpenAI **robusto** a diferencias de SDK/modelo (`max_output_tokens` / `max_completion_tokens` / `max_tokens` y `response_format`).

