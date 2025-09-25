from services.schema import OfertaAnalizada

def test_basic_model():
    data = {
        "resumen_servicios": "Soporte y mantenimiento.",
        "objetivos": ["Mejorar la calidad", "Reducir costes"],
        "alcance": "Gobernanza y soporte 24x7",
        "importe_total": 100000.0,
        "moneda": "EUR",
        "importes_detalle": [],
        "criterios_valoracion": [
            {"nombre":"Técnica","peso_max":60.0,"tipo":"técnico","subcriterios":[]},
            {"nombre":"Económica","peso_max":40.0,"tipo":"precio","subcriterios":[]},
        ],
        "indice_respuesta_tecnica": [
            {"titulo":"Metodología","descripcion":"Metodología de trabajo","subapartados":["Gobernanza","Calidad"]}
        ],
        "indice_propuesto": [
            {"titulo":"Enfoque Metodológico","descripcion":"Alineado al solicitado","subapartados":["Gobernanza","Calidad"]}
        ],
        "riesgos_y_dudas": None,
        "referencias_paginas": [3,5]
    }
    m = OfertaAnalizada.model_validate(data)
    assert m.importe_total == 100000.0
    assert m.criterios_valoracion[0].nombre == "Técnica"
    assert m.indice_propuesto[0].titulo.startswith("Enfoque")
