
from typing import Dict

SYSTEM_PREFIX = (
    "Eres analista sénior de licitaciones en España y consultor TI especializado en redacción de ofertas. "
    "Trabajas con rigor jurídico y enfoque ejecutable de delivery. "
    "Respondes EXCLUSIVAMENTE con JSON VÁLIDO (UTF-8) y NADA MÁS. "
    "Si una clave no aplica o no hay evidencia, devuélvela igualmente con valor null o []. "
    "NUNCA inventes; usa SOLO información de los PDFs. "
    "Incluye referencias de página como enteros únicos en orden ascendente cuando existan. "
    "Normaliza: números decimales con punto (ej. 1234.56), SIN separador de miles; moneda explícita (ej. EUR). "
    "Si un importe aparece con coma o con símbolo (1.234,56 €), conviértelo a float y separa moneda. "
    "Cuando existan varias versiones del mismo dato, usa 'discrepancias' para explicarlas brevemente. "
    "Optimiza por: precisión factual > completitud > concisión. "
)

SECTION_SPECS: Dict[str, Dict[str, str]] = {
    "objetivos_contexto": {
        "titulo": "Objetivos y contexto",
        "user_prompt": (
            "Extrae detalle estratégico, objetivos y contexto del cliente identificado en el pliego. Devuelve SIEMPRE las claves listadas. "
            "Amplía el contexto incluyendo objeto del contrato, duración, lotes, prorrogas, etc y CPV si aparecieran.\n"
            "Salida JSON EXACTA con claves:\n"
            "{"
            '  "resumen_servicios": str|null,'
            '  "objetivos": [str],'
            '  "alcance": str|null,'
            '  "duracion_contrato": {"meses": int|null, "prorrogas": int|null, "observaciones": str|null},'
            '  "lotes": [{"numero": int|null, "nombre": str|null}],'
            '  "cpv": [str],'
            '  "fecha_limite_presentacion": str|null,'
            '  "referencias_paginas": [int],'
            '  "evidencias": [{"pagina": int, "cita": str}],'
            '  "discrepancias": [str]'
            "}\n"
            "Reglas: no inventes; si no hay dato, usa null/[]."
        ),
    },
    "servicios": {
        "titulo": "Servicios solicitados (detalle)",
        "user_prompt": (
            "Lista servicios solicitados de forma ACCIONABLE, desglosando TAREAS y ENTREGABLES por servicio. "
            "Incluye también SLAs/KPIs con objetivo, método de medida, frecuencia, y penalizaciones si existieran; "
            "criterios de aceptación; periodicidad, volumen y ubicación/modalidad; gobierno, reporting y herramientas; "
            "equipo y perfiles (FTE mínimos, seniority, certificaciones), horario y ventanas de servicio; límites fuera de alcance y supuestos; "
            "y requisitos de transición (entrada/salida) si fueran explícitos.\n"
            "Devuelve SIEMPRE las claves:\n"
            "{"
            '  "resumen_servicios": str|null,'
            '  "servicios_detalle": ['
            '    {'
            '      "nombre": str,'
            '      "descripcion": str|null,'
            '      "tareas": [str],'
            '      "actividades": [str],'
            '      "entregables": [str],'
            '      "criterios_aceptacion": [str],'
            '      "requisitos": [str],'
            '      "periodicidad": str|null,'
            '      "volumen": str|null,'
            '      "ubicacion_modalidad": str|null,'
            '      "horario": {"jornada": str|null, "ventanas_servicio": [str]},'
            '      "roles_perfiles": [{"rol": str, "seniority": str|null, "fte_min": float|null, "certificaciones": [str]}],'
            '      "herramientas_requeridas": [str],'
            '      "gobernanza_reportes": {"comites": [str], "periodicidad_reportes": str|null, "plantillas": [str]},'
            '      "fuera_alcance": [str],'
            '      "supuestos": [str],'
            '      "transicion": {"entrada": [str], "salida": [str]},'
            '      "sla_kpi": ['
            '         {"nombre": str, "objetivo": str|null, "unidad": str|null, "metodo_medicion": str|null, "frecuencia": str|null, "penalizaciones": str|null}'
            '      ]'
            '    }'
            '  ],'
            '  "alcance": str|null,'
            '  "referencias_paginas": [int],'
            '  "evidencias": [{"pagina": int, "cita": str}],'
            '  "discrepancias": [str]'
            "}\n"
            "Reglas: deduplica; no inventes; null/[] si no hay."
        ),
    },
    "importe": {
        "titulo": "Importe de licitación",
        "user_prompt": (
            "Extrae importes y condiciones: presupuesto base de licitación (PBL), valor estimado del contrato (VEC), "
            "IVA incluido/excluido y tipo, anualidades y prórrogas, desglose por lotes, conceptos y periodos, y precios unitarios si existieran. "
            "Incluye observaciones relevantes (revisión de precios, gastos excluidos/incluidos como viajes, garantías, etc.).\n"
            "Devuelve SIEMPRE las claves:\n"
            "{"
            '  "importe_total": float|null,'
            '  "moneda": str|null,'
            '  "iva_incluido": bool|null,'
            '  "tipo_iva": float|null,'
            '  "presupuesto_base_licitacion": float|null,'
            '  "valor_estimado_contrato": float|null,'
            '  "revisiones_precio": str|null,'
            '  "gastos_incluidos": [str],'
            '  "gastos_excluidos": [str],'
            '  "garantias": [str],'
            '  "desglose_por_lote": [{"lote": int|null, "nombre": str|null, "importe_total": float|null, "moneda": str|null}],'
            '  "precios_unitarios": [{"concepto": str|null, "precio_unitario": float|null, "unidad": str|null, "observaciones": str|null}],'
            '  "importes_detalle": ['
            '    {"concepto": str|null, "importe": float|null, "moneda": str|null, "observaciones": str|null,'
            '     "periodo": {"tipo": "anualidad"|"prorroga"|null, "anyo": int|null, "duracion_meses": int|null}}'
            '  ],'
            '  "referencias_paginas": [int],'
            '  "evidencias": [{"pagina": int, "cita": str}],'
            '  "discrepancias": [str]'
            "}\n"
            "Reglas: números con punto; si hay varias cifras, recoge todas en importes_detalle y utiliza discrepancias cuando no cuadren."
        ),
    },
    "criterios_valoracion": {
        "titulo": "Criterios de valoración",
        "user_prompt": (
            "Extrae criterios/subcriterios con pesos, tipo (puntos/porcentaje), umbrales mínimos, método/fórmula de evaluación, "
            "evidencia exigida y criterios de desempate. Incluye tablas de puntuación o rangos si aparecen.\n"
            "Devuelve SIEMPRE las claves:\n"
            "{"
            '  "criterios_valoracion": ['
            '    {'
            '      "nombre": str,'
            '      "peso_max": float|null,'
            '      "tipo": "puntos"|"porcentaje"|null,'
            '      "umbral_minimo": float|null,'
            '      "metodo_evaluacion": str|null,'
            '      "tabla_puntuacion": [{"rango": str, "puntos": float|null, "condicion": str|null}],'
            '      "evidencia_requerida": [str],'
            '      "subcriterios": ['
            '        {"nombre": str, "peso_max": float|null, "tipo": "puntos"|"porcentaje"|null, "observaciones": str|null}'
            '      ]'
            '    }'
            '  ],'
            '  "criterios_desempate": [str],'
            '  "referencias_paginas": [int],'
            '  "evidencias": [{"pagina": int, "cita": str}],'
            '  "discrepancias": [str]'
            "}\n"
            "Reglas: conserva jerarquía; null/[] si no hay."
        ),
    },
    "indice_tecnico": {
        "titulo": "Índice de la respuesta técnica",
        "user_prompt": (
            "1) Identifica el índice solicitado literal para la respuesta técnica (si existe) con sus subapartados. "
            "2) Si no existiera, propón un índice implementable de AL MENOS 12 apartados, con subapartados accionables, "
            "que cubra: contexto/objetivo, alcance y matriz de trazabilidad, enfoque/arquitectura, metodología y actividades, "
            "planificación y cronograma/milestones, equipo y roles (RACI), gobierno y reporting, calidad y SLAs, "
            "gestión de riesgos/seguridad/continuidad, ciberseguridad/compliance, sostenibilidad/accesibilidad, "
            "gestión del cambio/comunicación, indicadores/KPIs, anexos (currícula, casos de éxito, etc.). "
            "Añade trazabilidad entre el índice propuesto y el solicitado si hay correspondencia.\n"
            "Salida JSON EXACTA:\n"
            "{"
            '  "indice_respuesta_tecnica": [ {"titulo": str, "descripcion": str|null, "subapartados": [str]} ],'
            '  "indice_propuesto": [ {"titulo": str, "descripcion": str|null, "subapartados": [str]} ],'
            '  "trazabilidad": [{"propuesto": str, "solicitado_match": str|null}],'
            '  "referencias_paginas": [int],'
            '  "evidencias": [{"pagina": int, "cita": str}],'
            '  "discrepancias": [str]'
            "}\n"
            "Reglas: si no hay índice literal, 'indice_respuesta_tecnica' puede ir [], pero 'indice_propuesto' DEBE incluir >=12 apartados."
        ),
    },
    "riesgos_exclusiones": {
        "titulo": "Riesgos y exclusiones",
        "user_prompt": (
            "Identifica riesgos (probabilidad/impacto) y exclusiones explícitas. "
            "Si no hay listas textuales, deriva riesgos compatibles con el pliego (pero no inventes fuera de su alcance) "
            "y anota supuestos y restricciones si aparecen. "
            "Devuelve SIEMPRE las claves:\n"
            "{"
            '  "riesgos_y_dudas": str|null,'
            '  "exclusiones": [str],'
            '  "matriz_riesgos": ['
            '    {"riesgo": str, "probabilidad_1_5": int|null, "impacto_1_5": int|null,'
            '     "criticidad_1_25": int|null, "mitigacion": str|null, "responsable": str|null}'
            '  ],'
            '  "supuestos": [str],'
            '  "restricciones": [str],'
            '  "referencias_paginas": [int],'
            '  "evidencias": [{"pagina": int, "cita": str}],'
            '  "discrepancias": [str]'
            "}\n"
            "Reglas: coherencia con el pliego; null/[] si no hay."
        ),
    },
    "solvencia": {
        "titulo": "Criterios de solvencia",
        "user_prompt": (
            "Extrae requisitos de solvencia técnica, económica y administrativa, cómo se acreditan y cualquier certificación/ratio/experiencia mínima. "
            "Incluye clasificación empresarial si aplica, y requisitos de medios personales/materiales.\n"
            "Devuelve SIEMPRE las claves:\n"
            "{"
            '  "solvencia": {'
            '    "tecnica": [str],'
            '    "economica": [str],'
            '    "administrativa": [str],'
            '    "acreditacion": ['
            '       {"requisito": str, "documento_necesario": str|null, "norma_referencia": str|null, "umbral": str|null}'
            '    ],'
            '    "certificaciones_requeridas": [str],'
            '    "ratios_financieros": [str],'
            '    "clasificacion_empresarial": str|null,'
            '    "medios_personales": [str],'
            '    "medios_materiales": [str]'
            '  },'
            '  "referencias_paginas": [int],'
            '  "evidencias": [{"pagina": int, "cita": str}],'
            '  "discrepancias": [str]'
            "}\n"
            "Reglas: una condición por bullet; null/[] si no hay."
        ),
    },
    "formato_oferta": {
        "titulo": "Formato y entrega de la oferta",
        "user_prompt": (
            "Extrae requisitos de formato y entrega: extensión máxima (global y por secciones), tamaño de fuente, tipografía, interlineado y márgenes; "
            "estructura documental requerida; idioma; número de copias; formatos de archivo y tamaño máximo; firma digital y quién firma; "
            "paginación/numeración; etiquetado de sobres/archivos; plataforma/canal de entrega; "
            "fecha/hora límite de presentación y zona horaria; instrucciones de presentación; anexos obligatorios; confidencialidad. "
            "Devuelve SIEMPRE las claves:\n"
            "{"
            '  "formato_esperado": str|null,'
            '  "longitud_paginas": int|null,'
            '  "tipografia": {"familia": str|null, "tamano_min": float|null, "interlineado": float|null, "margenes": str|null},'
            '  "limites_por_seccion": [{"seccion": str, "max_paginas": int|null}],'
            '  "estructura_documental": [ {"titulo": str, "observaciones": str|null} ],'
            '  "requisitos_presentacion": [str],'
            '  "requisitos_archivo": {'
            '     "formatos_permitidos": [str],'
            '     "tamano_max_mb": float|null,'
            '     "firma_digital_requerida": bool|null,'
            '     "firmado_por": str|null,'
            '     "nomenclatura_archivos": str|null'
            '  },'
            '  "idioma": str|null,'
            '  "copias": int|null,'
            '  "entrega": {'
            '     "canal": str|null,'
            '     "plataforma": str|null,'
            '     "plazo": str|null,'
            '     "zona_horaria": str|null,'
            '     "instrucciones": [str]'
            '  },'
            '  "paginacion": {"requerida": bool|null, "formato": str|null},'
            '  "etiquetado": [str],'
            '  "anexos_obligatorios": [str],'
            '  "confidencialidad": str|null,'
            '  "referencias_paginas": [int],'
            '  "evidencias": [{"pagina": int, "cita": str}],'
            '  "discrepancias": [str]'
            "}\n"
            "Reglas: no inventes; null/[] si no hay; convierte longitudes numéricas cuando sea posible."
        ),
    },
}

SECTION_KEYWORDS = {
    "objetivos_contexto": {"objeto del contrato": 6, "objeto": 4, "alcance": 5, "objetivo": 5,
                           "contexto": 3, "descripción del servicio": 4, "duración": 5, "prórroga": 4,
                           "lote": 4, "cpv": 4, "presentación de ofertas": 5, "plazo de presentación": 6},
    "servicios": {"servicios": 6, "actividades": 5, "tareas": 6, "entregables": 6,
                  "nivel de servicio": 5, "sla": 5, "kpi": 5, "penaliz": 4,
                  "criterios de aceptación": 5, "periodicidad": 4, "volumen": 4,
                  "ubicación": 3, "gobernanza": 4, "report": 4, "herramientas": 3,
                  "ventanas de servicio": 4, "horario": 4, "transición": 4, "salida": 3, "entrada": 3,
                  "fuera de alcance": 5, "supuestos": 4, "perfiles": 4, "fte": 4},
    "importe": {"presupuesto base": 7, "valor estimado": 7, "importe": 5, "precio": 4, "iva": 5,
                "base imponible": 5, "prórroga": 5, "anualidad": 5, "licitación": 4,
                "lote": 5, "precio unitario": 6, "revisión de precios": 6, "garantía": 4, "gastos": 4},
    "criterios_valoracion": {"criterios de valoración": 7, "criterios de adjudicación": 7,
                             "baremo": 6, "puntuación": 6, "puntos": 5, "porcentaje": 5,
                             "peso": 5, "umbral": 5, "desempate": 5, "fórmula": 6, "tabla de puntuación": 6, "evidencia": 4},
    "indice_tecnico": {"índice": 7, "estructura": 6, "contenido de la oferta": 6, "contenido mínimo": 6,
                       "memoria técnica": 6, "apartados": 5, "secciones": 5, "instrucciones de preparación": 5,
                       "formato de la propuesta": 5, "presentación de ofertas": 4, "sobre técnico": 5, "anexos": 4},
    "riesgos_exclusiones": {"exclusiones": 7, "no incluye": 7, "quedan excluidos": 7,
                            "no serán objeto": 6, "limitaciones": 5, "incompatibilidades": 5,
                            "responsabilidad": 4, "penalizaciones": 5,
                            "causas de exclusión": 6, "condiciones especiales": 4, "riesgos": 5,
                            "supuestos": 4, "restricciones": 4},
    "solvencia": {"solvencia técnica": 7, "solvencia económica": 7, "solvencia financiera": 6,
                  "requisitos de solvencia": 7, "clasificación": 5, "experiencia": 5,
                  "medios personales": 6, "medios materiales": 5, "acreditación": 6,
                  "certificación": 5, "ratio": 4},
    "formato_oferta": {"formato": 6, "presentación de ofertas": 7, "longitud": 6, "páginas": 6,
                       "tamaño de letra": 6, "tipografía": 6, "interlineado": 5, "márgenes": 5,
                       "estructura documental": 6, "idioma": 5, "copias": 5, "sobres": 6, "sobre electrónico": 6,
                       "plataforma": 6, "perfil del contratante": 5, "archivo pdf": 6, "firma digital": 6,
                       "paginación": 5, "etiquetado": 5, "nomenclatura": 5, "fecha de presentación": 6}
}

SECTION_CONTEXT_TUNING = {
    "indice_tecnico": {"max_chars": 90000, "window": 2},
    "riesgos_exclusiones": {"max_chars": 70000, "window": 2},
    "formato_oferta": {"max_chars": 70000, "window": 2},
}
