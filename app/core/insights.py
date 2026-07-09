"""Generación de observaciones automáticas (reglas deterministas, no ML) a partir de las matrices.

Todas las reglas son explicables: ranking simple, pendiente de una regresión lineal
(`numpy.polyfit`) para tendencias, y z-score para anomalías. El lenguaje es neutral de dominio
("cohorte", "entidad") para que sirva igual con datos de estudiantes, clientes o suscriptores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.formatting import format_pct


@dataclass
class Insight:
    titulo: str
    mensaje: str
    severidad: str  # "info" | "exito" | "alerta"


def _linear_trend(y: np.ndarray) -> float | None:
    """Pendiente de una regresión lineal simple de `y` contra su índice de orden (0..n-1)."""
    valid = ~np.isnan(y)
    if valid.sum() < 2:
        return None
    x = np.arange(len(y))[valid]
    slope, _ = np.polyfit(x, y[valid], 1)
    return float(slope)


def generate_insights(
    retention: pd.DataFrame,
    kpis: dict,
    conversion: pd.Series | None = None,
    abandono_status: pd.Series | None = None,
) -> list[Insight]:
    """Construye la lista de observaciones/recomendaciones para la pestaña de Insights."""
    insights: list[Insight] = []
    if retention.empty or not kpis:
        return insights

    if kpis.get("mejor_cohorte") is not None:
        insights.append(
            Insight(
                "Mejor cohorte",
                f"La cohorte **{kpis['mejor_cohorte']}** muestra la retención promedio más alta "
                "a lo largo de su ciclo de vida.",
                "exito",
            )
        )
    if kpis.get("peor_cohorte") is not None and kpis.get("peor_cohorte") != kpis.get("mejor_cohorte"):
        insights.append(
            Insight(
                "Cohorte con oportunidad de mejora",
                f"La cohorte **{kpis['peor_cohorte']}** muestra la retención promedio más baja; "
                "conviene revisar qué cambió en su proceso de adquisición u onboarding.",
                "alerta",
            )
        )

    sizes = kpis.get("tamano_cohortes")
    if sizes is not None and len(sizes) >= 3:
        slope = _linear_trend(sizes.to_numpy(dtype=float))
        if slope is not None and slope > 0:
            insights.append(
                Insight(
                    "Tendencia de crecimiento",
                    "El tamaño de las cohortes muestra una tendencia creciente a lo largo del tiempo.",
                    "exito",
                )
            )
        elif slope is not None and slope < 0:
            insights.append(
                Insight(
                    "Tendencia de contracción",
                    "El tamaño de las cohortes muestra una tendencia decreciente a lo largo del tiempo.",
                    "alerta",
                )
            )

    row_means = retention.mean(axis=1, skipna=True)
    if len(row_means) >= 3:
        slope = _linear_trend(row_means.to_numpy(dtype=float))
        if slope is not None and slope > 0.01:
            insights.append(
                Insight(
                    "Retención mejorando",
                    "Las cohortes más recientes retienen mejor que las más antiguas: la tendencia "
                    "de retención promedio es positiva.",
                    "exito",
                )
            )
        elif slope is not None and slope < -0.01:
            insights.append(
                Insight(
                    "Retención empeorando",
                    "Las cohortes más recientes retienen peor que las más antiguas: la tendencia "
                    "de retención promedio es negativa.",
                    "alerta",
                )
            )

    if len(row_means.dropna()) >= 3:
        mean_val = row_means.mean(skipna=True)
        std_val = row_means.std(skipna=True)
        if std_val and std_val > 0:
            z_scores = (row_means - mean_val) / std_val
            for cohort, z in z_scores[z_scores.abs() > 2].items():
                direction = "muy por encima" if z > 0 else "muy por debajo"
                insights.append(
                    Insight(
                        f"Anomalía detectada: {cohort}",
                        f"La cohorte **{cohort}** tiene una retención promedio {direction} del resto "
                        f"(z-score {z:.1f}).",
                        "alerta",
                    )
                )

    if retention.shape[1] >= 2:
        step_drop = (retention.iloc[:, 0] - retention.iloc[:, 1]).mean(skipna=True)
        if pd.notna(step_drop) and step_drop > 0.3:
            insights.append(
                Insight(
                    "Punto crítico temprano",
                    f"En promedio se pierde {format_pct(step_drop)} de cada cohorte entre el periodo "
                    "0 y el 1. Es el punto de mayor fuga y el de mayor impacto si se corrige.",
                    "alerta",
                )
            )

    if conversion is not None and not conversion.empty:
        avg_conv = conversion.mean(skipna=True)
        if pd.notna(avg_conv):
            insights.append(
                Insight(
                    "Conversión general",
                    f"La tasa de conversión promedio entre cohortes es {format_pct(avg_conv)}.",
                    "info",
                )
            )

    if abandono_status is not None and not abandono_status.empty:
        avg_ab = abandono_status.mean(skipna=True)
        if pd.notna(avg_ab) and avg_ab > 0.3:
            insights.append(
                Insight(
                    "Abandono elevado",
                    f"La tasa de abandono promedio basada en estado es {format_pct(avg_ab)}, por "
                    "encima del 30%. Se recomienda priorizar acciones de retención.",
                    "alerta",
                )
            )

    if not insights:
        insights.append(
            Insight(
                "Sin hallazgos relevantes",
                "No se detectaron patrones destacables con la configuración actual.",
                "info",
            )
        )
    return insights
