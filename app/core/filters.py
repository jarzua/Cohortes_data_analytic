"""Filtros dinámicos: un widget por columna filtrable, combinables entre sí (AND) y en cascada.

"En cascada" significa que elegir un valor en un filtro reduce las opciones disponibles en los
demás (ej. elegir Facultad="Ingeniería" acota la lista de Programa a los de esa facultad), sin
necesidad de declarar a mano qué columna depende de cuál — cada widget calcula sus opciones sobre
el DataFrame ya acotado por los demás filtros activos.
"""

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


def _sort_options(values: list) -> list:
    """Orden natural (numérico si son números, alfabético si son texto); nunca mezcla ambos."""
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=str)


def _collect_active_selections(filterable: dict[str, ColumnType], key_prefix: str) -> dict:
    """Lee del `session_state` (de la corrida anterior) qué filtros están activos ahora mismo."""
    active: dict = {}
    for col, col_type in filterable.items():
        value = st.session_state.get(f"{key_prefix}_{col}")
        if not value:
            continue
        if col_type in _MULTISELECT_TYPES and isinstance(value, list):
            active[col] = ("isin", value)
        elif col_type == ColumnType.FECHA and isinstance(value, tuple) and len(value) == 2:
            active[col] = ("date_range", value)
        elif col_type == ColumnType.NUMERICO_CONTINUO and isinstance(value, tuple) and len(value) == 2:
            active[col] = ("range", value)
    return active


def _sanitize_multiselect_state(key: str, options: list) -> None:
    """Quita del estado guardado cualquier valor que ya no esté entre las opciones acotadas.

    Streamlit lanza una excepción si el valor previamente seleccionado de un `multiselect` no está
    en la lista de `options` actual — algo que puede pasar aquí porque las opciones cambian según
    los demás filtros activos (cascada). Hay que sanear el estado ANTES de crear el widget.
    """
    if key in st.session_state:
        current = st.session_state[key]
        valid = [v for v in current if v in options]
        if valid != current:
            st.session_state[key] = valid


def _sanitize_date_state(key: str, min_d, max_d) -> None:
    if key in st.session_state:
        current = st.session_state[key]
        if isinstance(current, tuple) and len(current) == 2:
            start, end = current
            if start < min_d or end > max_d or start > end:
                st.session_state[key] = (min_d, max_d)


def _sanitize_range_state(key: str, lo: float, hi: float) -> None:
    if key in st.session_state:
        current = st.session_state[key]
        if isinstance(current, tuple) and len(current) == 2:
            c_lo, c_hi = current
            if c_lo < lo or c_hi > hi or c_lo > c_hi:
                st.session_state[key] = (lo, hi)


def render_filters(df: pd.DataFrame, types: dict[str, ColumnType], key_prefix: str = "filter") -> dict:
    """Dibuja un widget de filtro por cada columna filtrable y devuelve las selecciones activas."""
    filterable = get_filterable_columns(types)
    selections: dict = {}
    if not filterable:
        st.caption("No se detectaron columnas filtrables en este dataset.")
        return selections

    st.caption("Los filtros se acotan entre sí: elegir un valor reduce las opciones de los demás.")
    active = _collect_active_selections(filterable, key_prefix)

    for col, col_type in filterable.items():
        widget_key = f"{key_prefix}_{col}"
        other_selections = {c: v for c, v in active.items() if c != col}
        scoped_df = apply_filters(df, other_selections)

        if col_type in _MULTISELECT_TYPES:
            options = _sort_options(scoped_df[col].dropna().unique().tolist())
            if not options or len(options) > MAX_MULTISELECT_OPTIONS:
                continue
            _sanitize_multiselect_state(widget_key, options)
            selected = st.multiselect(col, options=options, default=[], key=widget_key)
            if selected:
                selections[col] = ("isin", selected)
        elif col_type == ColumnType.FECHA:
            valid = scoped_df[col].dropna()
            if valid.empty:
                continue
            min_d, max_d = valid.min().date(), valid.max().date()
            if min_d == max_d:
                continue
            _sanitize_date_state(widget_key, min_d, max_d)
            date_range = st.date_input(
                col, value=(min_d, max_d), min_value=min_d, max_value=max_d, key=widget_key
            )
            if isinstance(date_range, tuple) and len(date_range) == 2 and date_range != (min_d, max_d):
                selections[col] = ("date_range", date_range)
        elif col_type == ColumnType.NUMERICO_CONTINUO:
            valid = scoped_df[col].dropna()
            if valid.empty:
                continue
            lo, hi = float(valid.min()), float(valid.max())
            if lo == hi:
                continue
            _sanitize_range_state(widget_key, lo, hi)
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
