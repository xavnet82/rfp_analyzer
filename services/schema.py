from typing import List, Optional
from pydantic import BaseModel, Field

class ImporteDetalle(BaseModel):
    concepto: Optional[str] = Field(None, description="Concepto o desglose, p.ej. 'Lote 1' o 'Renovación año 2'")
    importe: Optional[float] = Field(None, description="Importe numérico si es detectable")
    moneda: Optional[str] = Field(None, description="Moneda detectada, p.ej. EUR")
    observaciones: Optional[str] = None

class Subcriterio(BaseModel):
    nombre: str
    peso_max: Optional[float] = None
    tipo: Optional[str] = Field(None, description="técnico, automático, precio, etc.")
    observaciones: Optional[str] = None

class Criterio(BaseModel):
    nombre: str
    peso_max: Optional[float] = None
    tipo: Optional[str] = Field(None, description="técnico, automático, precio, etc.")
    subcriterios: List[Subcriterio] = Field(default_factory=list)

class SeccionIndice(BaseModel):
    titulo: str
    descripcion: Optional[str] = None
    subapartados: List[str] = Field(default_factory=list)

class OfertaAnalizada(BaseModel):
    resumen_servicios: str
    objetivos: List[str] = Field(default_factory=list, description="Lista de objetivos clave que busca el pliego")
    alcance: Optional[str] = Field(None, description="Resumen de alcance, límites y exclusiones si se detectan")
    importe_total: Optional[float] = None
    moneda: Optional[str] = None
    importes_detalle: List[ImporteDetalle] = Field(default_factory=list)
    criterios_valoracion: List[Criterio] = Field(default_factory=list)
    # Índice solicitado: literalmente el que demanda el pliego (respetar numeración y títulos)
    indice_respuesta_tecnica: List[SeccionIndice] = Field(default_factory=list)
    # Índice propuesto: normalizado y alineado al solicitado (para guiar la redacción)
    indice_propuesto: List[SeccionIndice] = Field(default_factory=list)
    riesgos_y_dudas: Optional[str] = None
    referencias_paginas: List[int] = Field(default_factory=list)

    @property
    def to_markdown(self) -> str:
        md = [f"""# Resumen de Servicios
{self.resumen_servicios}

## Objetivos
"""]
        if self.objetivos:
            for o in self.objetivos:
                md.append(f"- {o}")
        else:
            md.append("- N/D")
        if self.alcance:
            md.append("\n## Alcance\n" + self.alcance)
        md.append("""

# Importe
- Importe total: {} {}
""".format(self.importe_total if self.importe_total is not None else "N/D", self.moneda or "" ))

        if self.importes_detalle:
            md.append("## Detalle de importes")
            for d in self.importes_detalle:
                md.append(f"- {d.concepto or 'Concepto'}: {d.importe if d.importe is not None else 'N/D'} {d.moneda or ''} {('- ' + d.observaciones) if d.observaciones else ''}")
        md.append("\n# Criterios de valoración")
        for c in self.criterios_valoracion:
            md.append(f"- **{c.nombre}** (peso máx: {c.peso_max if c.peso_max is not None else 'N/D'}; tipo: {c.tipo or 'N/D'})")
            for s in c.subcriterios:
                md.append(f"  - {s.nombre} (máx: {s.peso_max if s.peso_max is not None else 'N/D'}; tipo: {s.tipo or 'N/D'}) {('· ' + s.observaciones) if s.observaciones else ''}")
        md.append("\n# Índice solicitado (literal del pliego)")
        if self.indice_respuesta_tecnica:
            for i, sec in enumerate(self.indice_respuesta_tecnica, 1):
                md.append(f"{i}. **{sec.titulo}** - {sec.descripcion or ''}")
                for j, sub in enumerate(sec.subapartados, 1):
                    md.append(f"   {i}.{j} {sub}")
        else:
            md.append("- N/D")
        md.append("\n# Índice propuesto (alineado)")
        if self.indice_propuesto:
            for i, sec in enumerate(self.indice_propuesto, 1):
                md.append(f"{i}. **{sec.titulo}** - {sec.descripcion or ''}")
                for j, sub in enumerate(sec.subapartados, 1):
                    md.append(f"   {i}.{j} {sub}")
        else:
            md.append("- N/D")
        if self.riesgos_y_dudas:
            md.append("\n# Riesgos y dudas\n" + self.riesgos_y_dudas)
        if self.referencias_paginas:
            md.append("\n_Páginas referenciadas:_ " + ", ".join(map(str, self.referencias_paginas)))
        return "\n".join(md)
