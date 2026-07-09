"""Motor Universal de Cohortes — punto de entrada de la aplicación Streamlit.

Ejecutar con: `streamlit run app.py` (desde la carpeta `app/`).
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from core import cohort_engine, data_loader, filters, insights, metrics, profiling, visualizations
from utils.formatting import format_number, format_pct
from utils.types import (
    AggregationType,
    CohortConfig,
    ColumnType,
    EngineMode,
    Granularity,
    MatrixView,
    StatusBucket,
    StatusMapping,
)

st.set_page_config(page_title="Motor Universal de Cohortes", page_icon="📊", layout="wide")

NONE_LABEL = "(Ninguno)"
COHORT_COLUMN_TYPES = (
    ColumnType.FECHA,
    ColumnType.PERIODO_ORDINAL,
    ColumnType.NUMERICO_CATEGORICO,
    ColumnType.CATEGORICO,
    ColumnType.BOOLEANO,
)
OBSERVATION_COLUMN_TYPES = (
    ColumnType.FECHA,
    ColumnType.PERIODO_ORDINAL,
    ColumnType.NUMERICO_CATEGORICO,
    ColumnType.NUMERICO_CONTINUO,
)
METRIC_COLUMN_TYPES = (ColumnType.NUMERICO_CONTINUO, ColumnType.NUMERICO_CATEGORICO)
STATUS_COLUMN_TYPES = (ColumnType.CATEGORICO, ColumnType.NUMERICO_CATEGORICO, ColumnType.BOOLEANO)
MAX_STATUS_VALUES = 30

_KEYWORD_BUCKETS = [
    (("desert", "churn", "cancel", "baja", "abandon", "retir", "perdid"), StatusBucket.ABANDONO),
    (("gradu", "convert", "complet", "gan", "cerr"), StatusBucket.CONVERTIDO),
    (("activ", "matricul", "vigente", "reten", "true", "1"), StatusBucket.RETENIDO),
    (("pend", "inscri", "proceso", "aplaz", "suspend", "trial", "prueba"), StatusBucket.PENDIENTE),
]


def _guess_bucket(value: object) -> StatusBucket:
    text = str(value).strip().lower()
    for keywords, bucket in _KEYWORD_BUCKETS:
        if any(k in text for k in keywords):
            return bucket
    return StatusBucket.IGNORAR


@st.cache_data(show_spinner="Cargando y perfilando el archivo...")
def _load_and_profile(file_bytes: bytes, filename: str, sheet_name: str | None):
    buffer = io.BytesIO(file_bytes)
    df = data_loader.load_dataset(buffer, filename, sheet_name=sheet_name)
    types = profiling.infer_column_types(df)
    df = profiling.coerce_column_types(df, types)
    return df, types


def _options_for(types: dict[str, ColumnType], wanted: tuple[ColumnType, ...]) -> list[str]:
    return [c for c, t in types.items() if t in wanted]


# ---------------------------------------------------------------------------
# Sidebar: carga de archivo
# ---------------------------------------------------------------------------
st.sidebar.title("📊 Motor Universal de Cohortes")
uploaded_file = st.sidebar.file_uploader("Carga un archivo Excel o CSV", type=["csv", "xlsx", "xls"])

if uploaded_file is None:
    st.title("📊 Motor Universal de Cohortes")
    st.info(
        "Carga un archivo Excel o CSV desde el panel lateral para comenzar. La aplicación detecta "
        "automáticamente columnas de fecha, categóricas y numéricas, y te permite construir un "
        "análisis de cohortes completo sin configuración previa."
    )
    st.stop()

file_bytes = uploaded_file.getvalue()
sheet_name = None
if data_loader.is_excel(uploaded_file.name):
    sheets = data_loader.list_sheets(io.BytesIO(file_bytes))
    if len(sheets) > 1:
        sheet_name = st.sidebar.selectbox("Hoja de Excel", options=sheets)
    else:
        sheet_name = sheets[0]

try:
    df, types = _load_and_profile(file_bytes, uploaded_file.name, sheet_name)
except ValueError as exc:
    st.error(f"No se pudo cargar el archivo: {exc}")
    st.stop()

st.sidebar.success(f"{df.shape[0]:,} filas × {df.shape[1]} columnas cargadas.")

tabs = st.tabs(
    [
        "📂 Datos",
        "⚙️ Configuración",
        "🔢 Matriz de Cohortes",
        "📈 Retención y Abandono",
        "📊 Dashboard Ejecutivo",
        "🧠 Insights",
        "📐 Metodología",
    ]
)

# ---------------------------------------------------------------------------
# Tab 1: Datos
# ---------------------------------------------------------------------------
with tabs[0]:
    st.header("Estructura del Dataset")
    profiles = profiling.build_column_profiles(df, types)
    st.dataframe(profiling.dataset_summary_table(profiles), use_container_width=True)

    total_nulls = sum(p.n_nulls for p in profiles)
    col1, col2, col3 = st.columns(3)
    col1.metric("Filas", format_number(df.shape[0]))
    col2.metric("Columnas", df.shape[1])
    col3.metric("Valores nulos totales", format_number(total_nulls))

    st.subheader("Estadísticas Descriptivas (columnas numéricas)")
    desc = profiling.descriptive_stats(df, types)
    if desc.empty:
        st.caption("No se detectaron columnas numéricas.")
    else:
        st.dataframe(desc, use_container_width=True)

    st.subheader("Vista previa")
    st.dataframe(df.head(50), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2: Configuración
# ---------------------------------------------------------------------------
with tabs[1]:
    st.header("Configuración del Análisis de Cohortes")

    cohort_options = _options_for(types, COHORT_COLUMN_TYPES)
    if not cohort_options:
        st.error(
            "No se detectó ninguna columna de fecha, periodo o categoría utilizable como cohorte."
        )
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        cohort_column = st.selectbox(
            "Columna de Cohorte",
            options=cohort_options,
            help="Fecha (se truncará a la granularidad elegida), periodo (ej. '2025-2') o categoría "
            "(ej. Ciudad, Programa) que define a qué cohorte pertenece cada registro.",
        )
        granularity = st.selectbox(
            "Granularidad temporal",
            options=list(Granularity),
            index=list(Granularity).index(Granularity.MES),
            format_func=lambda g: g.value,
            help="Solo aplica cuando la columna de cohorte u observación es una fecha.",
        )
    with c2:
        obs_options = [NONE_LABEL] + _options_for(types, OBSERVATION_COLUMN_TYPES)
        observation_column = st.selectbox(
            "Columna de Observación / Antigüedad",
            options=obs_options,
            help="Fecha o periodo que se compara contra el inicio de la cohorte para calcular la "
            "antigüedad. Si se omite, no habrá evolución temporal (solo comparación estática).",
        )
        entity_options = [NONE_LABEL] + list(df.columns)
        default_entity = next((c for c, t in types.items() if t == ColumnType.IDENTIFICADOR), NONE_LABEL)
        entity_id_column = st.selectbox(
            "Columna de ID de Entidad (opcional)",
            options=entity_options,
            index=entity_options.index(default_entity) if default_entity in entity_options else 0,
            help="Si se indica, la retención se calcula sobre entidades distintas; si no, sobre "
            "conteo de registros.",
        )

    mode_override_label = st.radio(
        "Modo del motor",
        options=["Automático", "Forzar Evento (log de actividad)", "Forzar Snapshot (una fila por entidad)"],
        horizontal=True,
        help="Automático detecta el modo según si el ID de entidad se repite en varias filas.",
    )
    mode_override = None
    if mode_override_label.startswith("Forzar Evento"):
        mode_override = EngineMode.EVENTO
    elif mode_override_label.startswith("Forzar Snapshot"):
        mode_override = EngineMode.SNAPSHOT

    st.divider()
    st.subheader("Métrica y Agregación")
    m1, m2 = st.columns(2)
    with m1:
        metric_options = [NONE_LABEL] + _options_for(types, METRIC_COLUMN_TYPES)
        metric_column = st.selectbox(
            "Columna de Métrica (opcional)",
            options=metric_options,
            help="Columna numérica sobre la que aplicar la función de agregación (suma, promedio...).",
        )
    with m2:
        aggregation = st.selectbox(
            "Función de Agregación",
            options=list(AggregationType),
            format_func=lambda a: a.value,
            disabled=metric_column == NONE_LABEL,
        )

    st.divider()
    st.subheader("Estado (para Retención / Abandono / Conversión precisos)")
    status_options = [NONE_LABEL] + _options_for(types, STATUS_COLUMN_TYPES)
    status_column = st.selectbox(
        "Columna de Estado (opcional)",
        options=status_options,
        help="Ej. Estado = Matriculado/Desertor/Graduado. Permite calcular abandono y conversión "
        "basados en el estado real de cada entidad, no solo en presencia/ausencia.",
    )
    status_mapping = None
    if status_column != NONE_LABEL:
        unique_values = df[status_column].dropna().unique().tolist()
        if len(unique_values) > MAX_STATUS_VALUES:
            st.warning(f"'{status_column}' tiene demasiados valores únicos para mapear ({len(unique_values)}).")
        else:
            st.caption("Asigna cada valor a un bucket de negocio:")
            value_to_bucket = {}
            bucket_options = list(StatusBucket)
            n_cols = min(3, len(unique_values)) or 1
            cols = st.columns(n_cols)
            for i, value in enumerate(unique_values):
                guess = _guess_bucket(value)
                with cols[i % n_cols]:
                    chosen = st.selectbox(
                        str(value),
                        options=bucket_options,
                        index=bucket_options.index(guess),
                        format_func=lambda b: b.value,
                        key=f"status_bucket_{status_column}_{value}",
                    )
                    value_to_bucket[value] = chosen
            status_mapping = StatusMapping(column=status_column, value_to_bucket=value_to_bucket)

    st.divider()
    st.subheader("Filtros Dinámicos")
    with st.expander("Mostrar filtros", expanded=False):
        filter_selections = filters.render_filters(df, types)

    st.divider()
    if st.button("🚀 Generar Análisis", type="primary", use_container_width=True):
        filtered_df = filters.apply_filters(df, filter_selections)
        if filtered_df.empty:
            st.error("Los filtros aplicados no dejan ningún registro. Ajusta la selección.")
        else:
            config = CohortConfig(
                cohort_column=cohort_column,
                granularity=granularity,
                observation_column=None if observation_column == NONE_LABEL else observation_column,
                entity_id_column=None if entity_id_column == NONE_LABEL else entity_id_column,
                status_mapping=status_mapping,
                metric_column=None if metric_column == NONE_LABEL else metric_column,
                aggregation=aggregation,
                engine_mode_override=mode_override,
            )
            try:
                tidy = cohort_engine.compute_cohort_table(filtered_df, types, config)
            except ValueError as exc:
                st.error(f"No se pudo generar el análisis: {exc}")
                tidy = pd.DataFrame()

            if tidy.empty:
                st.error(
                    "No se pudo construir la tabla de cohortes: revisa que las columnas elegidas "
                    "tengan datos válidos y compatibles entre sí."
                )
            else:
                mode = cohort_engine.detect_engine_mode(filtered_df, config)
                counts = metrics.build_count_matrix(tidy, mode)
                retention = metrics.retention_matrix(tidy, mode)
                churn = metrics.churn_matrix(tidy, mode)
                metric_matrix = (
                    metrics.build_metric_matrix(tidy, config.aggregation)
                    if config.metric_column
                    else pd.DataFrame()
                )
                kpis = metrics.executive_kpis(tidy, mode)
                conversion = metrics.conversion_rates(tidy) if config.status_mapping else pd.Series(dtype=float)
                abandono_status = (
                    metrics.abandono_rates(tidy) if config.status_mapping else pd.Series(dtype=float)
                )
                status_summary_df = (
                    metrics.status_summary(tidy) if config.status_mapping else pd.DataFrame()
                )
                st.session_state["result"] = dict(
                    tidy=tidy,
                    mode=mode,
                    counts=counts,
                    retention=retention,
                    churn=churn,
                    metric_matrix=metric_matrix,
                    kpis=kpis,
                    conversion=conversion,
                    abandono_status=abandono_status,
                    status_summary=status_summary_df,
                    config=config,
                )
                st.success(f"Análisis generado: {len(counts.index)} cohortes · modo detectado: {mode.value}.")

result = st.session_state.get("result")

# ---------------------------------------------------------------------------
# Tab 3: Matriz de Cohortes
# ---------------------------------------------------------------------------
with tabs[2]:
    st.header("Matriz de Cohortes")
    if result is None:
        st.info("Configura y genera el análisis en la pestaña ⚙️ Configuración.")
    else:
        view_options = ["Valores absolutos", "% respecto a cohorte inicial (retención)", "% respecto al total"]
        use_metric = result["config"].metric_column is not None
        source_label = st.radio(
            "Fuente de la matriz",
            options=(["Conteo de entidades", "Métrica agregada"] if use_metric else ["Conteo de entidades"]),
            horizontal=True,
        )
        base_matrix = (
            result["metric_matrix"] if source_label == "Métrica agregada" else result["counts"]
        )
        view_label = st.radio("Presentación", options=view_options, horizontal=True)
        view_map = {
            view_options[0]: MatrixView.ABSOLUTO,
            view_options[1]: MatrixView.PCT_COHORTE_INICIAL,
            view_options[2]: MatrixView.PCT_TOTAL,
        }
        display_matrix = metrics.to_percentage(base_matrix, view_map[view_label])
        is_pct = view_map[view_label] != MatrixView.ABSOLUTO

        st.dataframe(
            display_matrix.style.format(lambda v: format_pct(v) if is_pct else format_number(v))
            .background_gradient(cmap="Blues", axis=None),
            use_container_width=True,
        )
        st.plotly_chart(
            visualizations.heatmap_figure(display_matrix, "Heatmap de Cohortes", as_percentage=is_pct),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# Tab 4: Retención y Abandono
# ---------------------------------------------------------------------------
with tabs[3]:
    st.header("Retención y Abandono")
    if result is None:
        st.info("Configura y genera el análisis en la pestaña ⚙️ Configuración.")
    else:
        st.plotly_chart(visualizations.retention_curves_figure(result["retention"]), use_container_width=True)
        st.plotly_chart(visualizations.churn_curve_figure(result["churn"]), use_container_width=True)

        retention = result["retention"]
        if retention.shape[1] >= 2:
            drop = (retention.iloc[:, :-1].values - retention.iloc[:, 1:].values)
            drop_df = pd.DataFrame(drop, index=retention.index, columns=retention.columns[1:])
            if drop_df.size and not pd.isna(drop_df.values).all():
                worst_cell = drop_df.stack().idxmax()
                st.warning(
                    f"📍 Punto crítico de abandono: cohorte **{worst_cell[0]}**, entre la edad "
                    f"{retention.columns[list(retention.columns).index(worst_cell[1]) - 1]} y "
                    f"{worst_cell[1]} (mayor caída de retención observada)."
                )

        if not result["status_summary"].empty:
            st.plotly_chart(
                visualizations.status_distribution_figure(result["status_summary"]), use_container_width=True
            )

# ---------------------------------------------------------------------------
# Tab 5: Dashboard Ejecutivo
# ---------------------------------------------------------------------------
with tabs[4]:
    st.header("Dashboard Ejecutivo")
    if result is None:
        st.info("Configura y genera el análisis en la pestaña ⚙️ Configuración.")
    else:
        kpis = result["kpis"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total de registros", format_number(kpis.get("total_registros")))
        c2.metric("Cohortes activas", kpis.get("cohortes_activas", 0))
        c3.metric("Retención promedio", format_pct(kpis.get("retencion_promedio")))
        c4.metric("Abandono promedio", format_pct(kpis.get("abandono_promedio")))

        c5, c6 = st.columns(2)
        c5.metric("Mejor cohorte", str(kpis.get("mejor_cohorte") or "N/A"))
        c6.metric("Peor cohorte", str(kpis.get("peor_cohorte") or "N/A"))

        st.plotly_chart(
            visualizations.cohort_size_bar_figure(kpis.get("tamano_cohortes", pd.Series(dtype=float))),
            use_container_width=True,
        )

        if not result["conversion"].empty or not result["abandono_status"].empty:
            st.subheader("Conversión y Abandono basados en Estado")
            summary_df = pd.DataFrame(
                {"Conversión": result["conversion"], "Abandono": result["abandono_status"]}
            )
            st.dataframe(summary_df.style.format(format_pct), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 6: Insights
# ---------------------------------------------------------------------------
with tabs[5]:
    st.header("Insights Automáticos")
    if result is None:
        st.info("Configura y genera el análisis en la pestaña ⚙️ Configuración.")
    else:
        found = insights.generate_insights(
            result["retention"], result["kpis"], result["conversion"], result["abandono_status"]
        )
        icon_map = {"exito": "✅", "alerta": "⚠️", "info": "ℹ️"}
        for insight in found:
            st.markdown(f"**{icon_map.get(insight.severidad, 'ℹ️')} {insight.titulo}**")
            st.markdown(insight.mensaje)
            st.markdown("---")

# ---------------------------------------------------------------------------
# Tab 7: Metodología
# ---------------------------------------------------------------------------
with tabs[6]:
    st.header("Metodología: cómo se calcula cada número")

    st.subheader("1. Asignación de cohorte")
    st.markdown(
        "- **Columna de fecha**: se trunca a la granularidad elegida "
        "(día, semana, mes, trimestre, semestre o año).\n"
        "- **Columna de periodo** (ej. `2025-2`) o **numérica categórica** (ej. Semestre): se usa "
        "el valor directamente, con un orden numérico inferido automáticamente.\n"
        "- **Columna categórica** (ej. Ciudad): se usa el valor tal cual, sin orden temporal propio."
    )

    st.subheader("2. Antigüedad (edad de la cohorte)")
    st.latex(r"\text{edad} = \text{ordinal}(\text{observación}) - \text{ordinal}(\text{inicio de la entidad})")
    st.markdown(
        "El inicio de cada **entidad** es el mínimo valor ordinal alcanzado en cualquiera de sus "
        "filas (su primera aparición), no el valor de cada fila individual — así una misma entidad "
        "permanece en su cohorte de origen a lo largo de toda su historia. Si no hay columna de "
        "observación, la edad es 0 para todos los registros (comparación estática, sin evolución)."
    )

    st.subheader("3. Modo del motor")
    st.markdown(
        "- **Evento** (log de actividad, varias filas por entidad): "
        r"$\text{retención}(C, N) = \dfrac{\#\text{entidades distintas con edad} = N}{\#\text{entidades distintas con edad} = 0}$"
        "\n- **Snapshot** (una fila por entidad, antigüedad actual): "
        r"$\text{retención}(C, N) = \dfrac{\#\text{entidades con edad} \geq N}{\#\text{entidades en la cohorte}}$"
    )
    st.markdown(
        "El modo se detecta automáticamente: si el ID de entidad se repite en más de una fila, es "
        "Evento; si cada entidad aparece una sola vez (o no hay ID de entidad), es Snapshot."
    )

    st.subheader("4. Censura por tiempo insuficiente")
    st.markdown(
        "En modo Snapshot, una celda (cohorte, edad=N) se marca como **N/A** — no como 0% — cuando "
        "la cohorte, dado su punto de arranque, todavía no pudo alcanzar calendáricamente esa edad "
        "(por ejemplo, una cohorte de este mes no puede tener 6 meses de antigüedad todavía). Se "
        "calcula comparando N contra la edad máxima posible = edad de la observación más reciente "
        "del dataset menos el inicio de esa cohorte."
    )

    st.subheader("5. Abandono y Conversión")
    st.markdown(
        "- **Abandono (curva)** $= 1 - \\text{retención}$, edad a edad.\n"
        "- **Abandono / Conversión basados en estado** (si se mapea una columna de estado): cada "
        "valor único se asigna a un bucket (Retenido, Convertido, Abandono, Pendiente, Ignorar) y se "
        "calcula el % de entidades de la cohorte en cada bucket."
    )

    st.subheader("6. Insights automáticos")
    st.markdown(
        "Reglas deterministas, no modelos de machine learning: ranking de cohortes por retención "
        "promedio, pendiente de una regresión lineal simple (`numpy.polyfit`) sobre el tamaño y la "
        "retención de cohortes sucesivas para detectar tendencias, y z-score sobre la retención "
        "promedio de cada cohorte para marcar anomalías (|z| > 2)."
    )
