"""Perfilado automático de un DataFrame: tipos de columna, resumen, nulos y estadísticas.

La clasificación de tipos es puramente estructural (dtype + cardinalidad + patrones de texto),
nunca por nombre de columna, para que el motor funcione igual con cualquier dataset.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

from utils.types import ColumnProfile, ColumnType

IDENTIFIER_UNIQUE_RATIO = 0.9
IDENTIFIER_MIN_UNIQUE = 50
NUMERIC_CATEGORICAL_MAX_UNIQUE = 15
CATEGORICAL_MAX_UNIQUE = 200
_SAMPLE_SIZE = 500

# Patrones de "periodo" (año-semestre, año-trimestre, año-mes corto) verificados ANTES que el
# parseo genérico de fechas: "2025-2" es ambiguo (¿semestre 2 de 2025, o febrero 2025?) y pandas lo
# interpretaría como fecha; para el vocabulario de esta app (Semestre/Trimestre) es más útil tratarlo
# como periodo ordinal.
_PERIOD_PATTERNS = [
    re.compile(r"^\d{4}[-/][1-4]$"),
    re.compile(r"^\d{4}[-/]?Q[1-4]$", re.IGNORECASE),
    re.compile(r"^\d{4}[-/](0?[1-9]|1[0-2])$"),
]


def _looks_like_period(series: pd.Series) -> bool:
    values = series.dropna().astype(str).str.strip()
    if values.empty:
        return False
    sample = values if len(values) <= _SAMPLE_SIZE else values.sample(_SAMPLE_SIZE, random_state=0)
    matches = sample.apply(lambda v: any(p.match(v) for p in _PERIOD_PATTERNS))
    return bool(matches.mean() >= 0.9)


def _parses_as_date(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return False  # evita interpretar enteros (ids, edades, semestres) como fechas/epoch
    non_null = series.dropna()
    if non_null.empty:
        return False
    sample = non_null if len(non_null) <= _SAMPLE_SIZE else non_null.sample(_SAMPLE_SIZE, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample.astype(str), errors="coerce")
    return bool(parsed.notna().mean() >= 0.9)


def infer_column_types(df: pd.DataFrame) -> dict[str, ColumnType]:
    """Clasifica cada columna del dataset en un `ColumnType`."""
    n_rows = len(df)
    types: dict[str, ColumnType] = {}
    for col in df.columns:
        series = df[col]
        n_unique = int(series.nunique(dropna=True))

        if pd.api.types.is_bool_dtype(series):
            types[col] = ColumnType.BOOLEANO
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            types[col] = ColumnType.FECHA
            continue
        if pd.api.types.is_numeric_dtype(series):
            if n_unique <= NUMERIC_CATEGORICAL_MAX_UNIQUE:
                types[col] = ColumnType.NUMERICO_CATEGORICO
            else:
                types[col] = ColumnType.NUMERICO_CONTINUO
            continue

        # object / string
        if _looks_like_period(series):
            types[col] = ColumnType.PERIODO_ORDINAL
            continue
        if _parses_as_date(series):
            types[col] = ColumnType.FECHA
            continue

        unique_ratio = (n_unique / n_rows) if n_rows else 0.0
        if unique_ratio >= IDENTIFIER_UNIQUE_RATIO and n_unique > IDENTIFIER_MIN_UNIQUE:
            types[col] = ColumnType.IDENTIFICADOR
        else:
            types[col] = ColumnType.CATEGORICO
    return types


def coerce_column_types(df: pd.DataFrame, types: dict[str, ColumnType]) -> pd.DataFrame:
    """Aplica los tipos inferidos al DataFrame: castea fechas y baja `category` en categóricas.

    Convertir a `category` reduce memoria y acelera los `groupby` posteriores en datasets grandes.
    """
    df = df.copy()
    for col, col_type in types.items():
        if col_type == ColumnType.FECHA and not pd.api.types.is_datetime64_any_dtype(df[col]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                df[col] = pd.to_datetime(df[col], errors="coerce")
        elif col_type in (ColumnType.CATEGORICO, ColumnType.NUMERICO_CATEGORICO, ColumnType.PERIODO_ORDINAL):
            df[col] = df[col].astype("category")
    return df


def build_column_profiles(df: pd.DataFrame, types: dict[str, ColumnType]) -> list[ColumnProfile]:
    """Construye el resumen de estructura del dataset: tipo, únicos, nulos, ejemplos."""
    n_rows = len(df)
    profiles = []
    for col in df.columns:
        series = df[col]
        n_nulls = int(series.isna().sum())
        sample_values = series.dropna().unique()[:5].tolist()
        profiles.append(
            ColumnProfile(
                name=col,
                column_type=types[col],
                n_unique=int(series.nunique(dropna=True)),
                n_nulls=n_nulls,
                null_pct=(n_nulls / n_rows * 100) if n_rows else 0.0,
                sample_values=sample_values,
            )
        )
    return profiles


def dataset_summary_table(profiles: list[ColumnProfile]) -> pd.DataFrame:
    """Tabla resumen lista para mostrar en la UI (una fila por columna)."""
    return pd.DataFrame(
        [
            {
                "Columna": p.name,
                "Tipo detectado": p.column_type.value,
                "Únicos": p.n_unique,
                "Nulos": p.n_nulls,
                "% Nulos": round(p.null_pct, 2),
                "Ejemplos": ", ".join(str(v) for v in p.sample_values),
            }
            for p in profiles
        ]
    )


def descriptive_stats(df: pd.DataFrame, types: dict[str, ColumnType]) -> pd.DataFrame:
    """Estadísticas descriptivas básicas para columnas numéricas (continuas y categóricas-numéricas)."""
    numeric_cols = [
        c for c, t in types.items() if t in (ColumnType.NUMERICO_CONTINUO, ColumnType.NUMERICO_CATEGORICO)
    ]
    if not numeric_cols:
        return pd.DataFrame()
    stats = df[numeric_cols].describe().T
    stats = stats.rename(
        columns={
            "count": "Conteo",
            "mean": "Promedio",
            "std": "Desv. Estándar",
            "min": "Mínimo",
            "25%": "P25",
            "50%": "Mediana",
            "75%": "P75",
            "max": "Máximo",
        }
    )
    return stats.round(2)


def get_columns_by_type(types: dict[str, ColumnType], *wanted: ColumnType) -> list[str]:
    """Lista de nombres de columna cuyo tipo está entre los solicitados, en orden original."""
    return [c for c, t in types.items() if t in wanted]
