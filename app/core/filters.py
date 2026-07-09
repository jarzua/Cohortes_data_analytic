"""Filtros dinámicos: un widget por columna filtrable, combinables entre sí (AND)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.types import ColumnType

_MULTISELECT_TYPES = (
    ColumnType.CATEGORICO,
    ColumnType.NUMERICO_CATEGORICO,
    ColumnType.BOOLEANO,
    ColumnType.PERIODO_ORDINAL,
)
MAX_MULTISELECT_OPTIONS = 300


def get_filterable_columns(types: dict[str, ColumnType]) -> dict[str, ColumnType]:
    """Columnas candidatas a filtro: categóricas, numérico-categóricas, fechas y numéricas continuas."""
    return {
        c: t
        for c, t in types.items()
        if t in _MULTISELECT_TYPES or t in (ColumnType.FECHA, ColumnType.NUMERICO_CONTINUO)
    }


def render_filters(df: pd.DataFrame, types: dict[str, ColumnType], key_prefix: str = "filter") -> dict:
    """Dibuja un widget de filtro por cada columna filtrable y devuelve las selecciones activas."""
    filterable = get_filterable_columns(types)
    selections: dict = {}
    if not filterable:
        st.caption("No se detectaron columnas filtrables en este dataset.")
        return selections

    for col, col_type in filterable.items():
        widget_key = f"{key_prefix}_{col}"
        if col_type in _MULTISELECT_TYPES:
            options = sorted(df[col].dropna().unique().tolist(), key=str)
            if not options or len(options) > MAX_MULTISELECT_OPTIONS:
                continue
            selected = st.multiselect(col, options=options, default=[], key=widget_key)
            if selected:
                selections[col] = ("isin", selected)
        elif col_type == ColumnType.FECHA:
            valid = df[col].dropna()
            if valid.empty:
                continue
            min_d, max_d = valid.min().date(), valid.max().date()
            if min_d == max_d:
                continue
            date_range = st.date_input(
                col, value=(min_d, max_d), min_value=min_d, max_value=max_d, key=widget_key
            )
            if isinstance(date_range, tuple) and len(date_range) == 2 and date_range != (min_d, max_d):
                selections[col] = ("date_range", date_range)
        elif col_type == ColumnType.NUMERICO_CONTINUO:
            valid = df[col].dropna()
            if valid.empty:
                continue
            lo, hi = float(valid.min()), float(valid.max())
            if lo == hi:
                continue
            value = st.slider(col, min_value=lo, max_value=hi, value=(lo, hi), key=widget_key)
            if value != (lo, hi):
                selections[col] = ("range", value)
    return selections


def apply_filters(df: pd.DataFrame, selections: dict) -> pd.DataFrame:
    """Aplica todas las selecciones de filtro combinadas con AND."""
    if not selections:
        return df
    mask = pd.Series(True, index=df.index)
    for col, (kind, value) in selections.items():
        if kind == "isin":
            mask &= df[col].isin(value)
        elif kind == "date_range":
            start, end = value
            mask &= df[col].dt.date.between(start, end)
        elif kind == "range":
            lo, hi = value
            mask &= df[col].between(lo, hi)
    return df[mask]
