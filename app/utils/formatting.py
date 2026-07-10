"""Helpers de formato numérico usados en tablas, KPIs y gráficos."""

from __future__ import annotations

import math


def format_pct(value: float | None, decimals: int = 1, empty: str = "N/A") -> str:
    """Formatea una fracción (0-1) como porcentaje legible, o `empty` si no aplica (ej. censurado)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return empty
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float | None, decimals: int = 0, empty: str = "N/A") -> str:
    """Formatea un número con separador de miles, o `empty` si no aplica (ej. censurado)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return empty
    if decimals == 0:
        return f"{value:,.0f}"
    return f"{value:,.{decimals}f}"
