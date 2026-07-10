"""Construcción de figuras Plotly: heatmap de cohortes, curvas de retención/abandono y KPIs."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# Paleta validada (ver skill de dataviz): orden categórico fijo, nunca ciclado por índice arbitrario.
CATEGORICAL_COLORS = [
    "#2a78d6",  # 1 blue
    "#1baf7a",  # 2 aqua
    "#eda100",  # 3 yellow
    "#008300",  # 4 green
    "#4a3aa7",  # 5 violet
    "#e34948",  # 6 red
    "#e87ba4",  # 7 magenta
    "#eb6834",  # 8 orange
]
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
MUTED_INK = "#898781"
GRIDLINE = "rgba(137,135,129,0.25)"

_BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=MUTED_INK),
    margin=dict(l=10, r=10, t=50, b=10),
)


def heatmap_figure(matrix: pd.DataFrame, title: str, as_percentage: bool) -> go.Figure:
    """Heatmap cohorte×edad. `as_percentage` controla el formato de las anotaciones y el hover."""
    if matrix.empty:
        return go.Figure()
    z = matrix.values
    text = [
        [("-" if pd.isna(v) else (f"{v * 100:.0f}%" if as_percentage else f"{v:,.0f}")) for v in row]
        for row in z
    ]
    hover = "%{y} · edad %{x}<br>Valor: %{text}<extra></extra>"
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=[str(c) for c in matrix.columns],
            y=[str(i) for i in matrix.index],
            text=text,
            texttemplate="%{text}",
            hovertemplate=hover,
            colorscale=[[i / (len(SEQUENTIAL_BLUE) - 1), c] for i, c in enumerate(SEQUENTIAL_BLUE)],
            showscale=True,
            xgap=2,
            ygap=2,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Antigüedad (periodos desde el inicio de la cohorte)",
        yaxis_title="Cohorte",
        yaxis=dict(autorange="reversed", gridcolor=GRIDLINE),
        xaxis=dict(gridcolor=GRIDLINE, type="category"),
        **_BASE_LAYOUT,
    )
    return fig


def _lines_figure(matrix: pd.DataFrame, title: str, y_title: str, as_percentage: bool) -> go.Figure:
    """Una línea por cohorte a lo largo de la antigüedad. Colores categóricos en orden fijo."""
    if matrix.empty:
        return go.Figure()
    fig = go.Figure()
    for i, (cohort, row) in enumerate(matrix.iterrows()):
        color = CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)]
        y_values = row.values
        fig.add_trace(
            go.Scatter(
                x=list(matrix.columns),
                y=y_values,
                mode="lines+markers",
                name=str(cohort),
                line=dict(width=2, color=color),
                marker=dict(size=8, color=color),
                hovertemplate=(
                    f"Cohorte {cohort}<br>Edad %{{x}}<br>"
                    + ("%{y:.1%}" if as_percentage else "%{y:,.0f}")
                    + "<extra></extra>"
                ),
                connectgaps=False,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Antigüedad (periodos desde el inicio de la cohorte)",
        yaxis_title=y_title,
        yaxis=dict(gridcolor=GRIDLINE, tickformat=".0%" if as_percentage else None),
        xaxis=dict(gridcolor=GRIDLINE, type="category"),
        legend=dict(title="Cohorte", orientation="v"),
        hovermode="x unified",
        **_BASE_LAYOUT,
    )
    return fig


def retention_curves_figure(retention_matrix: pd.DataFrame) -> go.Figure:
    """Curva de retención (%) por cohorte a lo largo de la antigüedad."""
    return _lines_figure(retention_matrix, "Curva de Retención por Cohorte", "Retención", as_percentage=True)


def churn_curve_figure(churn_matrix: pd.DataFrame) -> go.Figure:
    """Curva de abandono (%) por cohorte a lo largo de la antigüedad."""
    return _lines_figure(churn_matrix, "Curva de Abandono por Cohorte", "Abandono", as_percentage=True)


def cohort_size_bar_figure(sizes: pd.Series) -> go.Figure:
    """Barras con el tamaño (entidades en la edad 0) de cada cohorte. Serie única: sin leyenda."""
    if sizes.empty:
        return go.Figure()
    fig = go.Figure(
        data=go.Bar(
            x=[str(i) for i in sizes.index],
            y=sizes.values,
            marker_color=CATEGORICAL_COLORS[0],
            hovertemplate="Cohorte %{x}<br>Tamaño: %{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Tamaño de Cohorte (entidades en el periodo de origen)",
        xaxis_title="Cohorte",
        yaxis_title="Entidades",
        xaxis=dict(gridcolor=GRIDLINE, type="category"),
        yaxis=dict(gridcolor=GRIDLINE),
        **_BASE_LAYOUT,
    )
    return fig


def status_distribution_figure(status_summary: pd.DataFrame) -> go.Figure:
    """Barras apiladas de distribución de buckets de estado por cohorte."""
    if status_summary.empty:
        return go.Figure()
    fig = go.Figure()
    for i, status in enumerate(status_summary.columns):
        color = CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)]
        fig.add_trace(
            go.Bar(
                x=[str(i) for i in status_summary.index],
                y=status_summary[status].values,
                name=str(status),
                marker_color=color,
                hovertemplate=f"{status}<br>Cohorte %{{x}}: %{{y:,.0f}}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Distribución de Estados por Cohorte",
        xaxis_title="Cohorte",
        yaxis_title="Entidades",
        barmode="stack",
        xaxis=dict(gridcolor=GRIDLINE, type="category"),
        yaxis=dict(gridcolor=GRIDLINE),
        legend=dict(title="Estado"),
        **_BASE_LAYOUT,
    )
    return fig
