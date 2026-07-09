"""Motor de asignación de cohortes y cálculo de antigüedad.

Produce una tabla larga y "tidy" con una fila por registro de entrada, anotada con:
`entity_id`, `cohort_key` (etiqueta de la cohorte), `cohort_start_ordinal` (ancla numérica de la
cohorte), `age` (antigüedad respecto a esa ancla), `observation_ordinal`, y opcionalmente
`status_bucket` / `metric_value`. `core.metrics` consume esta tabla para construir las matrices.

Soporta dos combinaciones de columnas de cohorte/observación:

- Fecha ↔ Fecha (o Fecha ↔ ninguna): el caso clásico, la antigüedad se mide en unidades de la
  granularidad elegida (mes, trimestre, ...).
- Columna "nativa" (periodo ordinal tipo "2025-2", o numérica tipo "Semestre") ↔ columna nativa:
  la antigüedad es la diferencia directa en esas unidades propias, sin aplicar granularidad.
- Cohorte nominal (categórica, sin orden intrínseco, p.ej. Ciudad) combinada con una columna de
  observación con orden (fecha o nativa): el "inicio" de cada cohorte se toma como el mínimo valor
  de observación dentro de ese grupo.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from utils.types import CohortConfig, ColumnType, EngineMode, Granularity

_ORDINAL_NATIVE_TYPES = (
    ColumnType.PERIODO_ORDINAL,
    ColumnType.NUMERICO_CATEGORICO,
    ColumnType.NUMERICO_CONTINUO,
)
_NOMINAL_TYPES = (ColumnType.CATEGORICO, ColumnType.IDENTIFICADOR, ColumnType.BOOLEANO)

_FREQ_MAP = {
    Granularity.DIA: "D",
    Granularity.SEMANA: "W",
    Granularity.MES: "M",
    Granularity.TRIMESTRE: "Q",
    Granularity.ANIO: "Y",
}

_PERIOD_PARSE = re.compile(r"^(\d{4})[-/]?Q?([0-9]{1,2})$", re.IGNORECASE)


def _period_ordinal_key(series: pd.Series) -> pd.Series:
    """Convierte strings tipo '2025-2' / '2025-Q1' en un entero ordenable y con espaciado uniforme.

    El "ciclo" (2, 4 o 12) se infiere del valor máximo de la parte derecha observado en la columna,
    para que la diferencia entre periodos consecutivos sea siempre 1 (semestre, trimestre o mes).
    """
    extracted = series.astype(str).str.strip().str.extract(_PERIOD_PARSE)
    years = pd.to_numeric(extracted[0], errors="coerce")
    parts = pd.to_numeric(extracted[1], errors="coerce")
    max_part = parts.max()
    if pd.isna(max_part):
        cycle = 1
    elif max_part <= 2:
        cycle = 2
    elif max_part <= 4:
        cycle = 4
    else:
        cycle = 12
    return years * cycle + (parts - 1)


def _date_cohort_label_and_key(series: pd.Series, granularity: Granularity) -> tuple[pd.Series, pd.Series]:
    """Etiqueta legible + clave numérica ordenable para una columna de fecha truncada a `granularity`."""
    valid = series.notna()
    if granularity == Granularity.SEMESTRE:
        year = series.dt.year
        half = np.where(series.dt.month > 6, 2, 1)
        sort_key = pd.Series(year * 2 + (half - 1), index=series.index, dtype="float64")
        label = year.astype("Int64").astype(str) + "-S" + pd.Series(half, index=series.index).astype(str)
    else:
        freq = _FREQ_MAP[granularity]
        period = series.dt.to_period(freq)
        label = period.astype(str)
        sort_key = period.apply(lambda p: p.ordinal if pd.notna(p) else np.nan).astype("float64")
    label = label.where(valid)
    sort_key = sort_key.where(valid)
    return label, sort_key


def _native_ordinal(series: pd.Series, col_type: ColumnType) -> pd.Series:
    """Clave numérica ordenable para columnas de periodo-ordinal o numéricas ya nativas."""
    if col_type == ColumnType.PERIODO_ORDINAL:
        return _period_ordinal_key(series)
    return pd.to_numeric(series, errors="coerce")


def detect_engine_mode(df: pd.DataFrame, config: CohortConfig) -> EngineMode:
    """Detecta si el dataset es un log de eventos (varias filas por entidad) o un snapshot.

    Si el usuario fija `engine_mode_override`, se respeta. Si no hay columna de ID de entidad, o
    cada entidad aparece una sola vez, se asume snapshot; si una entidad aparece en más de una fila
    (evidencia de observaciones repetidas en distintos periodos), se asume modo evento.
    """
    if config.engine_mode_override is not None:
        return config.engine_mode_override
    if not config.entity_id_column:
        return EngineMode.SNAPSHOT
    n_rows = len(df)
    n_unique = df[config.entity_id_column].nunique(dropna=True)
    return EngineMode.EVENTO if n_unique < n_rows else EngineMode.SNAPSHOT


def compute_cohort_table(
    df: pd.DataFrame, types: dict[str, ColumnType], config: CohortConfig
) -> pd.DataFrame:
    """Construye la tabla larga [entity_id, cohort_key, cohort_start_ordinal, age, ...] del análisis."""
    cohort_col = config.cohort_column
    obs_col = config.observation_column
    cohort_type = types[cohort_col]

    if cohort_type in _NOMINAL_TYPES:
        label = df[cohort_col].astype(str)
        own_ordinal = None
    elif cohort_type == ColumnType.FECHA:
        label, own_ordinal = _date_cohort_label_and_key(df[cohort_col], config.granularity)
    elif cohort_type in _ORDINAL_NATIVE_TYPES:
        label = df[cohort_col].astype(str)
        own_ordinal = _native_ordinal(df[cohort_col], cohort_type)
    else:
        raise ValueError(f"Tipo de columna de cohorte no soportado: {cohort_type}")

    result = pd.DataFrame(index=df.index)
    result["cohort_key"] = label
    entity_id = df[config.entity_id_column] if config.entity_id_column else df.index.to_series()
    result["entity_id"] = entity_id.values

    if obs_col is None:
        result["age"] = 0
        result["cohort_start_ordinal"] = own_ordinal if own_ordinal is not None else 0.0
        result["observation_ordinal"] = result["cohort_start_ordinal"]
    else:
        obs_type = types[obs_col]
        if obs_type == ColumnType.FECHA:
            _, obs_ordinal = _date_cohort_label_and_key(df[obs_col], config.granularity)
        elif obs_type in _ORDINAL_NATIVE_TYPES:
            obs_ordinal = _native_ordinal(df[obs_col], obs_type)
        else:
            raise ValueError(
                "La columna de observación/antigüedad debe ser de tipo fecha, numérica o periodo "
                "ordinal (no una categoría nominal sin orden)."
            )
        result["observation_ordinal"] = obs_ordinal

        # El "origen" de la cohorte siempre se ancla por ENTIDAD (su primera aparición), no por la
        # fila individual: esto es lo que hace que "misma columna como cohorte y observación" (el
        # caso clásico: cohorte = mes de primer evento, edad = meses desde ese primer evento)
        # produzca antigüedad creciente en vez de 0 siempre. Si no hay `entity_id_column`, cada fila
        # es su propia "entidad" (sin repeticiones) y el mínimo por grupo no cambia nada.
        source_ordinal = own_ordinal if own_ordinal is not None else obs_ordinal
        cohort_start = source_ordinal.groupby(result["entity_id"]).transform("min")
        result["cohort_start_ordinal"] = cohort_start
        result["age"] = obs_ordinal - cohort_start

        # La etiqueta de cohorte también debe fijarse por entidad (la de su fila de arranque), no
        # tomar el valor de cada fila: si no, una misma entidad "saltaría" de cohorte en cada
        # observación posterior en vez de conservar su cohorte de origen.
        anchor = pd.DataFrame(
            {"entity": result["entity_id"].values, "ordinal": source_ordinal.values, "label": label.values}
        )
        first_label_by_entity = (
            anchor.sort_values("ordinal", kind="mergesort")
            .drop_duplicates(subset="entity", keep="first")
            .set_index("entity")["label"]
        )
        result["cohort_key"] = result["entity_id"].map(first_label_by_entity)

    if config.metric_column:
        result["metric_value"] = pd.to_numeric(df[config.metric_column], errors="coerce")
    if config.status_mapping:
        result["status_bucket"] = (
            df[config.status_mapping.column].map(lambda v: config.status_mapping.bucket_for(v).value)
        )

    result = result.dropna(subset=["cohort_key", "age"])
    result["age"] = result["age"].astype(int)
    # Antigüedad negativa (observación anterior al arranque de la cohorte) indica inconsistencia de
    # datos: se descarta en vez de distorsionar la matriz con columnas de "edad" sin sentido.
    result = result[result["age"] >= 0]
    return result
