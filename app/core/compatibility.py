"""Reglas de compatibilidad entre columnas elegidas y evaluación de calidad de la configuración.

Dos responsabilidades:

1. Filtrar qué columnas tiene sentido ofrecer como "Observación/Antigüedad" dado el tipo de la
   columna de "Cohorte" elegida, y si la Granularidad temporal aplica o no — para que la interfaz
   solo muestre opciones compatibles en vez de dejar que el usuario arme combinaciones sin sentido
   (o, peor, matemáticamente inválidas: mezclar una escala de fechas con un valor numérico crudo
   produce edades sin significado, ver `core.cohort_engine`).
2. Dar una lectura rápida (🟢/🟡/🔴) de qué tan bien planteada está la configuración actual, con las
   razones explícitas, para guiar al usuario antes de generar el análisis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from utils.types import ColumnType

FECHA_FAMILY = "fecha"
PERIODO_FAMILY = "periodo"
NUMERICO_FAMILY = "numerico"
NOMINAL_FAMILY = "nominal"

_FAMILY_BY_TYPE = {
    ColumnType.FECHA: FECHA_FAMILY,
    ColumnType.PERIODO_ORDINAL: PERIODO_FAMILY,
    ColumnType.NUMERICO_CATEGORICO: NUMERICO_FAMILY,
    ColumnType.NUMERICO_CONTINUO: NUMERICO_FAMILY,
    ColumnType.CATEGORICO: NOMINAL_FAMILY,
    ColumnType.BOOLEANO: NOMINAL_FAMILY,
    ColumnType.IDENTIFICADOR: NOMINAL_FAMILY,
}

# PERIODO_ORDINAL usa una codificación interna propia (año×ciclo+parte, ver
# `cohort_engine._period_ordinal_key`) que NO está en la misma escala que un valor numérico crudo
# (NUMERICO_CATEGORICO/NUMERICO_CONTINUO) aunque ambos sean "números" — restarlos entre sí da
# antigüedades sin sentido (típicamente negativas y enormes, que además se descartan silenciosamente
# por el guard de edad>=0, dejando la cohorte vacía). Por eso PERIODO_ORDINAL forma su propia familia,
# separada de la familia "numérico" (NUMERICO_CATEGORICO ↔ NUMERICO_CONTINUO sí son compatibles entre
# sí: ambos usan el valor crudo tal cual, sin transformación).
_FECHA_TYPES = (ColumnType.FECHA,)
_PERIODO_TYPES = (ColumnType.PERIODO_ORDINAL,)
_NUMERICO_TYPES = (ColumnType.NUMERICO_CATEGORICO, ColumnType.NUMERICO_CONTINUO)
_ORDENADOS = _FECHA_TYPES + _PERIODO_TYPES + _NUMERICO_TYPES
_NOMINAL_TEMPORALMENTE_DEBIL = (ColumnType.CATEGORICO, ColumnType.BOOLEANO)


def ordinal_family(col_type: ColumnType) -> str:
    """A qué "escala" ordinal pertenece un tipo de columna: fecha, periodo, numérica o nominal."""
    return _FAMILY_BY_TYPE.get(col_type, NOMINAL_FAMILY)


def compatible_observation_types(cohort_type: ColumnType) -> tuple[ColumnType, ...]:
    """Tipos válidos para "Observación/Antigüedad" dado el tipo de la columna de Cohorte.

    Si la cohorte ya tiene su propia escala ordinal (fecha, periodo o numérica), la observación debe
    compartir esa misma escala — de lo contrario `edad = observación − cohorte` resta unidades
    incompatibles (p. ej. el ordinal interno de un periodo contra un semestre crudo) y el resultado
    no tiene sentido. Si la cohorte es nominal (sin escala propia, ej. Ciudad), cualquier columna
    ordenada sirve como observación, porque el origen se calcula por entidad, no por la cohorte.
    """
    family = ordinal_family(cohort_type)
    if family == FECHA_FAMILY:
        return _FECHA_TYPES
    if family == PERIODO_FAMILY:
        return _PERIODO_TYPES
    if family == NUMERICO_FAMILY:
        return _NUMERICO_TYPES
    return _ORDENADOS


def granularity_is_relevant(cohort_type: ColumnType, observation_type: ColumnType | None) -> bool:
    """La granularidad temporal (día/mes/...) solo aplica si cohorte u observación son de tipo Fecha."""
    return cohort_type == ColumnType.FECHA or observation_type == ColumnType.FECHA


@dataclass
class SelectionQuality:
    nivel: str  # "verde" | "amarillo" | "rojo"
    icono: str
    resumen: str
    razones: list[str] = field(default_factory=list)


def evaluate_selection(
    cohort_column: str,
    cohort_type: ColumnType,
    observation_column: str | None,
    entity_id_column: str | None,
    metric_column: str | None,
    metric_type: ColumnType | None,
) -> SelectionQuality:
    """Heurística explicable (no ML) de qué tan bien planteada está la configuración elegida."""
    razones_rojo: list[str] = []
    razones_amarillo: list[str] = []

    if metric_column is not None and metric_column == cohort_column:
        razones_rojo.append("La métrica y la cohorte son la misma columna: no aporta información nueva.")
    if metric_type == ColumnType.IDENTIFICADOR:
        razones_rojo.append("La métrica elegida es una columna identificadora (texto único por fila).")
    if observation_column is not None and observation_column == cohort_column and not entity_id_column:
        razones_rojo.append(
            "Cohorte y observación son la misma columna pero no hay ID de entidad: la antigüedad "
            "siempre será 0 (no hay forma de saber la primera aparición de cada registro)."
        )

    if razones_rojo:
        return SelectionQuality("rojo", "🔴", "No recomendado", razones_rojo)

    if observation_column is None:
        razones_amarillo.append("Sin columna de observación no hay evolución temporal (solo una foto estática).")
    if not entity_id_column:
        razones_amarillo.append(
            "Sin ID de entidad, la retención se basa en conteo de filas, no en entidades distintas."
        )
    if cohort_type in _NOMINAL_TEMPORALMENTE_DEBIL:
        razones_amarillo.append(
            "La cohorte es una categoría sin orden temporal propio: sirve para comparar segmentos, "
            "pero no mide antigüedad en el sentido clásico."
        )

    if razones_amarillo:
        return SelectionQuality("amarillo", "🟡", "Aceptable", razones_amarillo)

    return SelectionQuality(
        "verde",
        "🟢",
        "Excelente",
        ["Cohorte temporal, con observación compatible y ID de entidad: la configuración clásica de un análisis de cohortes."],
    )
