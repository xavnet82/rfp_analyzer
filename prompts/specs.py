
# prompts/specs.py
from typing import Dict

SYSTEM_PREFIX = (
  "Eres analista sénior de licitaciones en España y consultor TI."
  " Respondes EXCLUSIVAMENTE con JSON VÁLIDO (UTF-8) y NADA MÁS."
  " Si una clave no aplica o no hay evidencia, devuélvela igualmente con valor null o []."
  " NUNCA inventes; usa SOLO información de los PDFs."
  " Normaliza números decimales con punto y moneda explícita (p. ej., EUR)."
  " Números sin separador de miles. Citas ≤ 180 caracteres."
  " Incluye referencias de página como enteros únicos en orden ascendente cuando existan."
  " Si hay versiones alternativas de un dato, usa 'discrepancias' para explicarlo brevemente."
  " Optimiza por: precisión factual > concisión > completitud."
)

SECTION_SPECS: Dict[str, Dict[str, str]] = {
  "objetivos_contexto": {
    "titulo": "Objetivos y contexto",
    "user_prompt": (
      "Extrae objetivos y contexto del pliego. Devuelve SIEMPRE las claves listadas."
      "\\nSalida JSON EXACTA con claves:\\n"
      "{"
      '  "resumen_servicios": str|null,'
      '  "objetivos": [str],'
      '  "alcance": str|null,'
      '  "referencias_paginas": [int],'
      '  "evidencias": [{"pagina": int, "cita": str}],'
      '  "discrepancias": [str]'
      "}"
      "\\nReglas: no inventes; si no hay dato, usa null/[]."
    ),
  },
  "servicios": {
    "titulo": "Servicios solicitados (detalle)",
    "user_prompt": (
      "Lista servicios solicitados y detalles (entregables, SLAs/KPIs, etc.). Devuelve SIEMPRE las claves."
      "\\nSalida JSON EXACTA:\\n"
      "{"
      '  "resumen_servicios": str|null,'
      '  "servicios_detalle": ['
      '    {'
      '      "nombre": str,'
      '      "descripcion": str|null,'
      '      "entregables": [str],'
      '      "requisitos": [str],'
      '      "periodicidad": str|null,'
      '      "volumen": str|null,'
      '      "ubicacion_modalidad": str|null,'
      '      "sla_kpi": [{"nombre": str, "objetivo": str|null, "unidad": str|null, "metodo_medicion": str|null}],'
      '      "criterios_aceptacion": [str]'
      '    }'
      '  ],'
      '  "alcance": str|null,'
      '  "referencias_paginas": [int],'
      '  "evidencias": [{"pagina": int, "cita": str}],'
      '  "discrepancias": [str]'
      "}"
      "\\nReglas: deduplica; no inventes; null/[] si no hay."
    ),
  },
  "importe": {
    "titulo": "Importe de licitación",
    "user_prompt": (
      "Extrae importes y condiciones (IVA, anualidades/prórrogas). Devuelve SIEMPRE las claves."
      "\\nSalida JSON EXACTA:\\n"
      "{"
      '  "importe_total": float|null,'
      '  "moneda": str|null,'
      '  "iva_incluido": bool|null,'
      '  "tipo_iva": float|null,'
      '  "importes_detalle": ['
      '    {"concepto": str|null, "importe": float|null, "moneda": str|null, "observaciones": str|null,'
      '     "periodo": {"tipo": "anualidad"|"prorroga"|null, "anyo": int|null, "duracion_meses": int|null}}'
      '  ],'
      '  "referencias_paginas": [int],'
      '  "evidencias": [{"pagina": int, "cita": str}],'
      '  "discrepancias": [str]'
      "}"
      "\\nReglas: números con punto; si hay varias cifras, recoge todas en importes_detalle y usa discrepancias."
    ),
  },
  "criterios_valoracion": {
    "titulo": "Criterios de valoración",
    "user_prompt": (
      "Extrae criterios/subcriterios con pesos, tipo, umbrales, método y desempates. Devuelve SIEMPRE las claves."
      "\\nSalida JSON EXACTA:\\n"
      "{"
      '  "criterios_valoracion": ['
      '    {'
      '      "nombre": str,'
      '      "peso_max": float|null,'
      '      "tipo": "puntos"|"porcentaje"|null,'
      '      "umbral_minimo": float|null,'
      '      "metodo_evaluacion": str|null,'
      '      "subcriterios": ['
      '        {"nombre": str, "peso_max": float|null, "tipo": "puntos"|"porcentaje"|null, "observaciones": str|null}'
      '      ]'
      '    }'
      '  ],'
      '  "criterios_desempate": [str],'
      '  "referencias_paginas": [int],'
      '  "evidencias": [{"pagina": int, "cita": str}],'
      '  "discrepancias": [str]'
      "}"
      "\\nReglas: conserva jerarquía; null/[] si no hay."
    ),
  },
  "indice_tecnico": {
    "titulo": "Índice de la respuesta técnica",
    "user_prompt": (
      "1) Analiza en detalle la propuesta e identifica, si existe, el índice solicitado literal para la respuesta técnica. "
      "2) Si no existiera, propón en base al pliego un índice alineado (implementable), que contenga al menos: contexto, "
      "nuestro enfoque, metodología, alcance y actividades, planificación con hitos, equipo y roles, governance, gestión de calidad y SLAs, "
      "gestión de riesgos/continuidad, ciberseguridad/compliance, sostenibilidad/accesibilidad y anexos. "
      "El índice propuesto NO debe ir vacío y debe ser accionable (con subapartados)."
      "\\nSalida JSON EXACTA:\\n"
      "{"
      '  "indice_respuesta_tecnica": ['
      '    {"titulo": str, "descripcion": str|null, "subapartados": [str]}'
      '  ],'
      '  "indice_propuesto": ['
      '    {"titulo": str, "descripcion": str|null, "subapartados": [str]}'
      '  ],'
      '  "trazabilidad": [{"propuesto": str, "solicitado_match": str|null}],'
      '  "referencias_paginas": [int],'
      '  "evidencias": [{"pagina": int, "cita": str}],'
      '  "discrepancias": [str]'
      "}"
      "\\nReglas: si no hay índice literal, 'indice_respuesta_tecnica' puede ir [], pero 'indice_propuesto' DEBE incluir >=10 apartados con subapartados clave."
    ),
  },
  "riesgos_exclusiones": {
    "titulo": "Riesgos y exclusiones",
    "user_prompt": (
      "Identifica riesgos y exclusiones del pliego. Si no hay lista explícita, sintetiza riesgos compatibles con lo definido."
      " Devuelve SIEMPRE las claves."
      "\\nSalida JSON EXACTA:\\n"
      "{"
      '  "riesgos_y_dudas": str|null,'
      '  "exclusiones": [str],'
      '  "matriz_riesgos": ['
      '    {"riesgo": str, "probabilidad_1_5": int|null, "impacto_1_5": int|null,'
      '     "criticidad_1_25": int|null, "mitigacion": str|null, "responsable": str|null}'
      '  ],'
      '  "referencias_paginas": [int],'
      '  "evidencias": [{"pagina": int, "cita": str}],'
      '  "discrepancias": [str]'
      "}"
      "\\nReglas: no inventes fuera del pliego; si infieres, debe ser coherente con lo que SÍ aparece."
    ),
  },
  "solvencia": {
    "titulo": "Criterios de solvencia",
    "user_prompt": (
      "Extrae solvencia técnica, económica y administrativa y cómo se acredita. Devuelve SIEMPRE las claves."
      "\\nSalida JSON EXACTA:\\n"
      "{"
      '  "solvencia": {'
      '    "tecnica": [str],'
      '    "economica": [str],'
      '    "administrativa": [str],'
      '    "acreditacion": [{"requisito": str, "documento_necesario": str|null, "norma_referencia": str|null, "umbral": str|null}]'
      '  },'
      '  "referencias_paginas": [int],'
      '  "evidencias": [{"pagina": int, "cita": str}],'
      '  "discrepancias": [str]'
      "}"
      "\\nReglas: una condición por bullet; null/[] si no hay."
    ),
  },
  "formato_oferta": {
    "titulo": "Formato y entrega de la oferta",
    "user_prompt": (
      "Extrae requisitos de formato y entrega de la oferta (memoria técnica/administrativa): extensión máxima, tamaño de fuente, "
      "tipografía, interlineado, márgenes, estructura documental requerida, idioma, número de copias, formatos de archivo, tamaño máximo, "
      "firma digital y quién firma, paginación/numeración, etiquetado de sobres/archivos, canal de entrega (plataforma/sobre electrónico), "
      "plazo/fecha/hora y zona horaria, instrucciones de presentación y anexos obligatorios. Devuelve SIEMPRE las claves."
      "\\nSalida JSON EXACTA:\\n"
      "{"
      '  "formato_esperado": str|null,'
      '  "longitud_paginas": int|null,'
      '  "tipografia": {"familia": str|null, "tamano_min": float|null, "interlineado": float|null, "margenes": str|null},'
      '  "estructura_documental": [ {"titulo": str, "observaciones": str|null} ],'
      '  "requisitos_presentacion": [str],'
      '  "requisitos_archivo": {'
      '     "formatos_permitidos": [str],'
      '     "tamano_max_mb": float|null,'
      '     "firma_digital_requerida": bool|null,'
      '     "firmado_por": str|null'
      '  },'
      '  "idioma": str|null,'
      '  "copias": int|null,'
      '  "entrega": {'
      '     "canal": str|null,'
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
      "}"
      "\\nReglas: no inventes; null/[] si no hay; convierte longitudes numéricas cuando sea posible."
    ),
  },
}

SECTION_KEYWORDS = {
  "objetivos_contexto": {"objeto del contrato": 5, "objeto": 3, "alcance": 4, "objetivo": 3,
                         "contexto": 3, "descripción del servicio": 4, "alcances": 3},
  "servicios": {"servicios": 5, "actividades": 4, "tareas": 4, "entregables": 4,
                "nivel de servicio": 4, "sla": 3, "kpi": 3, "periodicidad": 3, "volumen": 3},
  "importe": {"presupuesto base": 6, "importe": 5, "precio": 4, "iva": 4,
              "base imponible": 4, "prórroga": 4, "anualidad": 4, "licitación": 4},
  "criterios_valoracion": {"criterios de valoración": 6, "criterios de adjudicación": 6,
                           "baremo": 5, "puntuación": 5, "puntos": 4, "porcentaje": 4,
                           "peso": 4, "umbral": 4, "desempate": 4, "fórmula": 4},
  "indice_tecnico": {"índice": 6, "indice": 6, "estructura": 5, "estructura mínima": 6,
                     "contenido de la oferta": 6, "contenido mínimo": 6, "memoria técnica": 5,
                     "documentación técnica": 5, "apartados": 4, "secciones": 4,
                     "instrucciones de preparación": 5, "formato de la propuesta": 5,
                     "orden de contenidos": 5, "capítulos": 4, "anexos": 3,
                     "presentación de ofertas": 4, "sobre técnico": 5},
  "riesgos_exclusiones": {"exclusiones": 7, "no incluye": 7, "quedan excluidos": 7,
                          "no serán objeto": 6, "limitaciones": 5, "incompatibilidades": 5,
                          "responsabilidad": 4, "exenciones": 4, "penalizaciones": 5,
                          "causas de exclusión": 6, "supuestos de exclusión": 6,
                          "condiciones especiales": 4, "garantías": 4, "plazos": 4,
                          "régimen sancionador": 5, "riesgos": 4, "restricciones": 4},
  "solvencia": {"solvencia técnica": 6, "solvencia económica": 6, "solvencia financiera": 5,
                "requisitos de solvencia": 6, "clasificación": 4, "experiencia": 4,
                "medios personales": 4, "medios materiales": 4, "acreditación": 5},
  "formato_oferta": {"formato": 6, "formato de la oferta": 7, "formato de la propuesta": 6,
                     "presentación de ofertas": 7, "presentacion de ofertas": 7, "presentación": 5,
                     "memoria técnica": 6, "longitud": 6, "páginas": 6, "paginas": 6, "extensión": 6, "extension": 6,
                     "tamaño de letra": 6, "tamano de letra": 6, "tipografía": 5, "tipografia": 5,
                     "interlineado": 5, "márgenes": 5, "margenes": 5, "fuente": 5, "tipo de letra": 5,
                     "etiquetado": 5, "rotulación": 5, "rotulacion": 5, "sobres": 6, "sobre electrónico": 6,
                     "plataforma": 6, "perfil del contratante": 5, "archivo pdf": 6, "pdf": 5, "docx": 4,
                     "firma electrónica": 6, "firma digital": 6, "firmado": 5,
                     "idioma": 5, "copia": 5, "copias": 5, "paginación": 5, "numeración": 5,
                     "fecha de entrega": 6, "plazo de presentación": 6, "hora": 5, "zona horaria": 4},
}

SECTION_CONTEXT_TUNING = {
    "indice_tecnico": {"max_chars": 80000, "window": 2},
    "riesgos_exclusiones": {"max_chars": 60000, "window": 2},
    "formato_oferta": {"max_chars": 60000, "window": 2},
}
