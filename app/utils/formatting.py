"""Helpers de formato numérico usados en tablas, KPIs y gráficos."""

from __future__ import annotations

import math


def format_pct(value: float | None, decimals: int = 1) -> str:
    """Formatea una fracción (0-1) como porcentaje legible, o 'N/A' si no aplica."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float | None, decimals: int = 0) -> str:
    """Formatea un número con separador de miles, o 'N/A' si no aplica."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    if decimals == 0:
        return f"{value:,.0f}"
    return f"{value:,.{decimals}f}"
