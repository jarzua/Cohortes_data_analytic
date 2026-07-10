"""Tipos compartidos del Motor Universal de Cohortes: enums y dataclasses de configuración."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ColumnType(str, Enum):
    """Clasificación semántica de una columna, inferida automáticamente por `core.profiling`."""

    FECHA = "fecha"
    PERIODO_ORDINAL = "periodo_ordinal"
    NUMERICO_CONTINUO = "numerico_continuo"
    NUMERICO_CATEGORICO = "numerico_categorico"
    CATEGORICO = "categorico"
    IDENTIFICADOR = "identificador"
    BOOLEANO = "booleano"


class Granularity(str, Enum):
    """Unidad temporal para truncar fechas y medir antigüedad entre periodos."""

    DIA = "Día"
    SEMANA = "Semana"
    MES = "Mes"
    TRIMESTRE = "Trimestre"
    SEMESTRE = "Semestre"
    ANIO = "Año"


class AggregationType(str, Enum):
    """Función de agregación aplicable a una métrica numérica."""

    CONTEO = "Conteo de registros"
    SUMA = "Suma"
    PROMEDIO = "Promedio"
    MEDIANA = "Mediana"
    MAXIMO = "Máximo"
    MINIMO = "Mínimo"


class EngineMode(str, Enum):
    """Modo de cálculo de antigüedad/retención del motor de cohortes."""

    EVENTO = "evento"
    SNAPSHOT = "snapshot"


class StatusBucket(str, Enum):
    """Bucket de negocio al que se mapea cada valor único de una columna de estado."""

    RETENIDO = "Retenido / Activo"
    CONVERTIDO = "Convertido"
    ABANDONO = "Abandono"
    PENDIENTE = "Pendiente / En proceso"
    IGNORAR = "Ignorar"


class MatrixView(str, Enum):
    """Modo de presentación de la matriz de cohortes."""

    ABSOLUTO = "Valores absolutos"
    PCT_COHORTE_INICIAL = "% respecto a cohorte inicial (retención)"
    PCT_TOTAL = "% respecto al total"


@dataclass
class ColumnProfile:
    """Resultado del perfilado automático de una columna."""

    name: str
    column_type: ColumnType
    n_unique: int
    n_nulls: int
    null_pct: float
    sample_values: list


@dataclass
class StatusMapping:
    """Mapeo definido por el usuario de valores de una columna de estado a buckets de negocio."""

    column: str
    value_to_bucket: dict = field(default_factory=dict)

    def bucket_for(self, value: object) -> StatusBucket:
        return self.value_to_bucket.get(value, StatusBucket.IGNORAR)


@dataclass
class CohortConfig:
    """Configuración completa elegida por el usuario para generar el análisis de cohortes."""

    cohort_column: str
    granularity: Granularity = Granularity.MES
    observation_column: str | None = None
    entity_id_column: str | None = None
    status_mapping: StatusMapping | None = None
    metric_column: str | None = None
    aggregation: AggregationType = AggregationType.CONTEO
    engine_mode_override: EngineMode | None = None
    cohort_label_column: str | None = None
    filters: dict = field(default_factory=dict)

    def cache_key(self) -> tuple:
        """Tupla hasheable estable para usar como parte de la key de `st.cache_data`."""
        status_key = None
        if self.status_mapping is not None:
            status_key = (
                self.status_mapping.column,
                tuple(sorted(self.status_mapping.value_to_bucket.items(), key=str)),
            )
        filters_key = tuple(sorted(((k, tuple(v)) for k, v in self.filters.items())))
        return (
            self.cohort_column,
            self.granularity.value,
            self.observation_column,
            self.entity_id_column,
            status_key,
            self.metric_column,
            self.aggregation.value,
            self.engine_mode_override.value if self.engine_mode_override else None,
            self.cohort_label_column,
            filters_key,
        )
