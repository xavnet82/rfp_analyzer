# App de análisis de pliegos (Streamlit + OpenAI)

Esta app permite cargar un PDF de un pliego de licitación, enviarlo a OpenAI para su análisis y mostrar un resultado **estructurado**:
- Resumen detallado de los servicios solicitados.
- Importe total, importes por lotes/renovaciones y moneda.
- Criterios de valoración y ponderaciones.
- Índice propuesto/solicitado de la **respuesta técnica**.
- Exportación a JSON y Markdown.

## Requisitos

- Python 3.10+
- Clave de API de OpenAI (variable de entorno `OPENAI_API_KEY`).

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate  # en Windows: .venv\Scripts\activate
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
- Se emplea `pypdf` para extracción de texto básica. Si el PDF es escaneado, considera integrar OCR (por ejemplo, `pytesseract`). 
- El envío a OpenAI aplica **chunking** para PDFs largos y agrega resultados.
- El formato final se valida con `pydantic` para minimizar errores de parseo.
- Control de coste: se permite elegir el modelo y un **límite de tokens aprox** (via truncado) desde la barra lateral.
- Exportaciones: JSON y Markdown.
