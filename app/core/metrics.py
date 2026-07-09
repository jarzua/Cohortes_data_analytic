"""Construcción de matrices cohorte×edad: conteos, métricas, retención, abandono y conversión.

Todas las funciones parten de la tabla "tidy" que produce `core.cohort_engine.compute_cohort_table`.

Fórmulas principales (ver también la pestaña "Metodología" de la app):

- Modo evento: retención(cohorte, edad=N) = entidades distintas con `age == N` ÷ entidades
  distintas con `age == 0`. Cada fila de la tabla tidy es una observación puntual en un periodo.
- Modo snapshot: retención(cohorte, edad=N) = entidades con `age >= N` ÷ tamaño de la cohorte
  (entidades con `age >= 0`, o sea todas). Cada entidad aporta una sola fila con su antigüedad
  actual, así que "reached age N" se calcula como acumulado desde el final hacia el 0.
- Censura: en modo snapshot, una celda (cohorte, N) se marca NaN si la cohorte, dado su punto de
  arranque, no pudo alcanzar calendaricamente la edad N todavía (no es 0% de retención, es "aún no
  aplica") — se calcula comparando N contra `max_valid_age = ref_ordinal_global - cohort_start`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.types import AggregationType, EngineMode, MatrixView, StatusBucket

_AGG_FUNCS = {
    AggregationType.CONTEO: "count",
    AggregationType.SUMA: "sum",
    AggregationType.PROMEDIO: "mean",
    AggregationType.MEDIANA: "median",
    AggregationType.MAXIMO: "max",
    AggregationType.MINIMO: "min",
}


def _ordered_cohorts(tidy: pd.DataFrame) -> list:
    """Cohortes ordenadas por su ancla temporal/numérica; empates se ordenan alfabéticamente."""
    order_df = tidy.groupby("cohort_key")["cohort_start_ordinal"].first().reset_index()
    order_df = order_df.sort_values(["cohort_start_ordinal", "cohort_key"])
    return order_df["cohort_key"].tolist()


def _age_range(tidy: pd.DataFrame) -> list[int]:
    """Rango de edades 0..max. `cohort_engine` garantiza que no existan edades negativas."""
    max_age = int(tidy["age"].max())
    return list(range(0, max_age + 1))


def _apply_censoring(pivot: pd.DataFrame, tidy: pd.DataFrame) -> pd.DataFrame:
    """Enmascara con NaN las celdas donde la cohorte aún no pudo alcanzar esa edad (ver docstring)."""
    if "observation_ordinal" not in tidy.columns or tidy["observation_ordinal"].isna().all():
        return pivot
    reference_ordinal = tidy["observation_ordinal"].max()
    starts = tidy.groupby("cohort_key")["cohort_start_ordinal"].first().reindex(pivot.index)
    max_valid_age = reference_ordinal - starts
    age_cols = np.array(pivot.columns, dtype=float)
    mask = age_cols[np.newaxis, :] > max_valid_age.to_numpy()[:, np.newaxis]
    return pivot.mask(mask)


def build_count_matrix(tidy: pd.DataFrame, mode: EngineMode) -> pd.DataFrame:
    """Matriz cohorte×edad de entidades distintas, con la semántica correcta según el modo."""
    if tidy.empty:
        return pd.DataFrame()
    cohorts = _ordered_cohorts(tidy)
    ages = _age_range(tidy)

    exact = (
        tidy.groupby(["cohort_key", "age"])["entity_id"]
        .nunique()
        .unstack("age")
        .reindex(index=cohorts, columns=ages)
    )

    if mode == EngineMode.EVENTO:
        return exact

    # Modo snapshot: "alcanzaron la edad N" = acumulado desde la edad máxima hacia atrás.
    reached = exact.fillna(0).iloc[:, ::-1].cumsum(axis=1).iloc[:, ::-1]
    return _apply_censoring(reached, tidy)


def build_metric_matrix(tidy: pd.DataFrame, aggregation: AggregationType) -> pd.DataFrame:
    """Matriz cohorte×edad de la métrica seleccionada, agregada con la función elegida."""
    if tidy.empty or "metric_value" not in tidy.columns:
        return pd.DataFrame()
    cohorts = _ordered_cohorts(tidy)
    ages = _age_range(tidy)
    agg_func = _AGG_FUNCS[aggregation]
    pivot = (
        tidy.groupby(["cohort_key", "age"])["metric_value"]
        .agg(agg_func)
        .unstack("age")
        .reindex(index=cohorts, columns=ages)
    )
    return pivot


def to_percentage(matrix: pd.DataFrame, view: MatrixView) -> pd.DataFrame:
    """Transforma una matriz de valores absolutos a % según cohorte inicial o % del total."""
    if matrix.empty or view == MatrixView.ABSOLUTO:
        return matrix
    with np.errstate(divide="ignore", invalid="ignore"):
        if view == MatrixView.PCT_COHORTE_INICIAL:
            base = matrix.iloc[:, 0]
            return matrix.div(base, axis=0)
        if view == MatrixView.PCT_TOTAL:
            grand_total = float(np.nansum(matrix.values))
            if not grand_total:
                return matrix * np.nan
            return matrix / grand_total
    return matrix


def retention_matrix(tidy: pd.DataFrame, mode: EngineMode) -> pd.DataFrame:
    """% de entidades retenidas por cohorte y edad, relativo al tamaño de la cohorte (edad 0)."""
    counts = build_count_matrix(tidy, mode)
    return to_percentage(counts, MatrixView.PCT_COHORTE_INICIAL)


def churn_matrix(tidy: pd.DataFrame, mode: EngineMode) -> pd.DataFrame:
    """Tasa de abandono (1 - retención) por cohorte y edad; conserva NaN de celdas censuradas."""
    retention = retention_matrix(tidy, mode)
    if retention.empty:
        return retention
    return 1 - retention


def status_summary(tidy: pd.DataFrame) -> pd.DataFrame:
    """Tabla cohorte × bucket de estado con conteo de entidades distintas."""
    if "status_bucket" not in tidy.columns:
        return pd.DataFrame()
    return (
        tidy.groupby(["cohort_key", "status_bucket"])["entity_id"]
        .nunique()
        .unstack("status_bucket")
        .fillna(0)
    )


def conversion_rates(tidy: pd.DataFrame) -> pd.Series:
    """% de entidades por cohorte cuyo estado mapea a Convertido o Retenido/Activo."""
    counts = status_summary(tidy)
    if counts.empty:
        return pd.Series(dtype=float)
    counts = counts.drop(columns=[StatusBucket.IGNORAR.value], errors="ignore")
    total = counts.sum(axis=1)
    converted_cols = [c for c in (StatusBucket.CONVERTIDO.value, StatusBucket.RETENIDO.value) if c in counts.columns]
    converted = counts[converted_cols].sum(axis=1) if converted_cols else pd.Series(0.0, index=counts.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (converted / total).replace([np.inf, -np.inf], np.nan)


def abandono_rates(tidy: pd.DataFrame) -> pd.Series:
    """% de entidades por cohorte cuyo estado mapea a Abandono/Churn."""
    counts = status_summary(tidy)
    if counts.empty:
        return pd.Series(dtype=float)
    counts = counts.drop(columns=[StatusBucket.IGNORAR.value], errors="ignore")
    total = counts.sum(axis=1)
    abandono_col = StatusBucket.ABANDONO.value
    abandono = counts[abandono_col] if abandono_col in counts.columns else pd.Series(0.0, index=counts.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (abandono / total).replace([np.inf, -np.inf], np.nan)


def executive_kpis(tidy: pd.DataFrame, mode: EngineMode) -> dict:
    """KPIs agregados para el Dashboard Ejecutivo."""
    if tidy.empty:
        return {}
    counts = build_count_matrix(tidy, mode)
    retention = to_percentage(counts, MatrixView.PCT_COHORTE_INICIAL)
    row_means = retention.mean(axis=1, skipna=True) if not retention.empty else pd.Series(dtype=float)

    avg_retention = float(np.nanmean(retention.values)) if retention.size else float("nan")
    avg_abandono = 1 - avg_retention if not np.isnan(avg_retention) else float("nan")

    return {
        "total_registros": int(tidy["entity_id"].nunique()),
        "cohortes_activas": int(len(counts.index)) if not counts.empty else 0,
        "retencion_promedio": avg_retention,
        "abandono_promedio": avg_abandono,
        "mejor_cohorte": row_means.idxmax() if not row_means.empty and row_means.notna().any() else None,
        "peor_cohorte": row_means.idxmin() if not row_means.empty and row_means.notna().any() else None,
        "tamano_cohortes": counts.iloc[:, 0] if not counts.empty else pd.Series(dtype=float),
    }
