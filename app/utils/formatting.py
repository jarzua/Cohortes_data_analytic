"""Helpers de formato numérico usados en tablas, KPIs y gráficos."""

from __future__ import annotations

import math


def safe_divide(numerator: float, denominator: float) -> float | None:
    """División segura: retorna None (no NaN) cuando el denominador es 0 o inválido."""
    if denominator is None or denominator == 0 or (isinstance(denominator, float) and math.isnan(denominator)):
        return None
    if numerator is None or (isinstance(numerator, float) and math.isnan(numerator)):
        return None
    return numerator / denominator


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


def format_currency(value: float | None, symbol: str = "$", decimals: int = 0) -> str:
    """Formatea un valor monetario con separador de miles."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{symbol}{value:,.{decimals}f}"
