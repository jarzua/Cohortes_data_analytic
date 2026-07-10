"""Motor Universal de Cohortes — punto de entrada de la aplicación Streamlit.

Ejecutar con: `streamlit run app.py` (desde la carpeta `app/`).
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from core import cohort_engine, compatibility, data_loader, filters, insights, metrics, profiling, visualizations
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

SHOW_METHODOLOGY_TAB = True

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


@st.cache_data(show_spinner="Cargando y perfilando el archivo...", max_entries=5)
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
try:
    sheet_name = None
    if data_loader.is_excel(uploaded_file.name):
        sheets = data_loader.list_sheets(io.BytesIO(file_bytes))
        if len(sheets) > 1:
            sheet_name = st.sidebar.selectbox("Hoja de Excel", options=sheets)
        else:
            sheet_name = sheets[0]
    df, types = _load_and_profile(file_bytes, uploaded_file.name, sheet_name)
except Exception as exc:
    # Frontera de E/S: un archivo corrupto, sin motor instalado (ImportError), un zip inválido, etc.
    # puede lanzar excepciones de muchos tipos distintos según el motor (openpyxl/xlrd) — se capturan
    # todas aquí para mostrar un mensaje legible en vez de tumbar el script de Streamlit.
    st.error(f"No se pudo cargar el archivo: {exc}")
    st.stop()

st.sidebar.success(f"{df.shape[0]:,} filas × {df.shape[1]} columnas cargadas.")

TAB_LABELS = [
    "📂 Datos",
    "⚙️ Configuración",
    "🔢 Matriz de Cohortes",
    "📈 Retención y Abandono",
    "📊 Dashboard Ejecutivo",
    "🧠 Insights",
]
if SHOW_METHODOLOGY_TAB:
    TAB_LABELS.append("📐 Metodología")

# Nota: se usa un selector (st.radio) en vez de st.tabs() para elegir la vista activa. La cantidad
# de widgets dentro de "Configuración" cambia dinámicamente entre reruns (mapeo de estado, filtros,
# granularidad condicional...), y eso desestabiliza la reconciliación del árbol de st.tabs() en el
# frontend: tras cambiar una selección, el contenido de todas las vistas queda apilado en una sola
# página en vez de aislado por pestaña. Con renderizado condicional explícito (if/elif) solo se
# ejecuta y dibuja el código de la vista activa, sin depender de que el frontend oculte las demás.
selected_view = st.radio("Vista", options=TAB_LABELS, horizontal=True, label_visibility="collapsed")
st.divider()

# ---------------------------------------------------------------------------
# Vista: Datos
# ---------------------------------------------------------------------------
if selected_view == "📂 Datos":
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
# Vista: Configuración
# ---------------------------------------------------------------------------
if selected_view == "⚙️ Configuración":
    st.header("Configuración del Análisis de Cohortes")

    cohort_options = _options_for(types, COHORT_COLUMN_TYPES)
    if not cohort_options:
        st.error(
            "No se detectó ninguna columna de fecha, periodo o categoría utilizable como cohorte."
        )
        st.stop()

    # La columna de Cohorte se elige primero: de su tipo depende qué columnas tiene sentido ofrecer
    # como Observación (misma escala ordinal) y si la Granularidad temporal aplica o no. Mostrar solo
    # las combinaciones compatibles evita tanto configuraciones sin sentido analítico como mezclas
    # matemáticamente inválidas (fecha vs. valor numérico crudo).
    cohort_column = st.selectbox(
        "Columna de Cohorte",
        options=cohort_options,
        help="Fecha (se truncará a la granularidad elegida), periodo (ej. '2025-2') o categoría "
        "(ej. Ciudad, Programa) que define a qué cohorte pertenece cada registro.",
    )
    cohort_type = types[cohort_column]

    label_options = [NONE_LABEL] + [
        c for c in df.columns if c != cohort_column and types[c] != ColumnType.IDENTIFICADOR
    ]
    cohort_label_column = st.selectbox(
        "Etiqueta de Cohorte para mostrar (opcional)",
        options=label_options,
        help="No afecta el cálculo de antigüedad, solo el texto que se muestra en las filas de la "
        "matriz. Útil cuando la columna de cohorte correcta para el cálculo (ej. 'Semestre Ingreso') "
        "no es la más legible: aquí puedes mapear una columna más descriptiva (ej. 'Periodo de "
        "Ingreso') que sea constante dentro de cada cohorte.",
    )

    c1, c2 = st.columns(2)
    with c1:
        compatible_types = compatibility.compatible_observation_types(cohort_type)
        obs_options = [NONE_LABEL] + _options_for(types, compatible_types)
        observation_column = st.selectbox(
            "Columna de Observación / Antigüedad",
            options=obs_options,
            help="Se compara contra el inicio de la cohorte para calcular la antigüedad. Solo se "
            "muestran columnas compatibles con la escala de la cohorte elegida. Si se omite, no habrá "
            "evolución temporal (solo comparación estática).",
        )
        observation_type = None if observation_column == NONE_LABEL else types[observation_column]
    with c2:
        entity_options = [NONE_LABEL] + list(df.columns)
        default_entity = next((c for c, t in types.items() if t == ColumnType.IDENTIFICADOR), NONE_LABEL)
        entity_id_column = st.selectbox(
            "Columna de ID de Entidad (opcional)",
            options=entity_options,
            index=entity_options.index(default_entity) if default_entity in entity_options else 0,
            help="Si se indica, la retención se calcula sobre entidades distintas; si no, sobre "
            "conteo de registros.",
        )

    if compatibility.granularity_is_relevant(cohort_type, observation_type):
        granularity = st.selectbox(
            "Granularidad temporal",
            options=list(Granularity),
            index=list(Granularity).index(Granularity.MES),
            format_func=lambda g: g.value,
        )
    else:
        granularity = Granularity.MES
        st.caption("⏱️ Granularidad temporal: no aplica (ni la cohorte ni la observación son fechas).")

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
    quality = compatibility.evaluate_selection(
        cohort_column=cohort_column,
        cohort_type=cohort_type,
        observation_column=None if observation_column == NONE_LABEL else observation_column,
        entity_id_column=None if entity_id_column == NONE_LABEL else entity_id_column,
        metric_column=None if metric_column == NONE_LABEL else metric_column,
        metric_type=None if metric_column == NONE_LABEL else types[metric_column],
    )
    st.markdown(f"**{quality.icono} Calidad de la configuración: {quality.resumen}**")
    for razon in quality.razones:
        st.caption(f"· {razon}")

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
                cohort_label_column=None if cohort_label_column == NONE_LABEL else cohort_label_column,
            )
            try:
                tidy = cohort_engine.compute_cohort_table(filtered_df, types, config)
                if tidy.empty:
                    raise ValueError(
                        "No quedó ningún registro válido: revisa que las columnas elegidas tengan "
                        "datos válidos y compatibles entre sí (fechas parseables, antigüedad no "
                        "negativa, etc.)."
                    )
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
                if config.status_mapping:
                    status_summary_df = metrics.status_summary(tidy)
                    conversion = metrics.conversion_rates(status_summary_df)
                    abandono_status = metrics.abandono_rates(status_summary_df)
                else:
                    status_summary_df = pd.DataFrame()
                    conversion = pd.Series(dtype=float)
                    abandono_status = pd.Series(dtype=float)
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
            except ValueError as exc:
                st.error(f"No se pudo generar el análisis: {exc}")

result = st.session_state.get("result")

# ---------------------------------------------------------------------------
# Vista: Matriz de Cohortes
# ---------------------------------------------------------------------------
if selected_view == "🔢 Matriz de Cohortes":
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
            display_matrix.style.format(
                lambda v: format_pct(v, empty="-") if is_pct else format_number(v, empty="-")
            ).background_gradient(cmap="Blues", axis=None),
            use_container_width=True,
        )
        st.plotly_chart(
            visualizations.heatmap_figure(display_matrix, "Heatmap de Cohortes", as_percentage=is_pct),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# Vista: Retención y Abandono
# ---------------------------------------------------------------------------
if selected_view == "📈 Retención y Abandono":
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
# Vista: Dashboard Ejecutivo
# ---------------------------------------------------------------------------
if selected_view == "📊 Dashboard Ejecutivo":
    st.header("Dashboard Ejecutivo")
    if result is None:
        st.info("Configura y genera el análisis en la pestaña ⚙️ Configuración.")
    else:
        kpis = result["kpis"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total de entidades", format_number(kpis.get("total_entidades")))
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
# Vista: Insights
# ---------------------------------------------------------------------------
if selected_view == "🧠 Insights":
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
# Vista: Metodología (oculta temporalmente, ver SHOW_METHODOLOGY_TAB)
# ---------------------------------------------------------------------------
if SHOW_METHODOLOGY_TAB and selected_view == "📐 Metodología":
    st.header("📐 Metodología: cómo leer y configurar el análisis")
    st.caption(
        "Explicación práctica, no solo matemática: qué significa cada número y cómo elegir bien "
        "cada columna en la pestaña ⚙️ Configuración."
    )

    # -----------------------------------------------------------------------
    st.subheader("1. ¿Qué es una matriz de cohortes, en la práctica?")
    st.markdown(
        "Imagina que en enero inscribiste 100 estudiantes. Un mes después, 80 seguían activos. Dos "
        "meses después, 65. Eso ya es una cohorte con su curva de retención: **100 → 80 → 65**.\n\n"
        "La matriz de cohortes hace exactamente eso, pero para **todas** las camadas de estudiantes "
        "(enero, febrero, marzo...) al mismo tiempo, una debajo de la otra, para poder comparar: "
        "¿la cohorte de marzo retiene mejor o peor que la de enero a los 2 meses?"
    )
    st.markdown(
        "|  | Edad 0 | Edad 1 | Edad 2 |\n"
        "|---|---|---|---|\n"
        "| Cohorte Enero | 100 | 80 | 65 |\n"
        "| Cohorte Febrero | 120 | 90 | - |\n"
        "| Cohorte Marzo | 95 | 70 | - |\n"
    )
    st.caption(
        "Los `-` en Febrero y Marzo no son error ni ceros: esas cohortes **todavía no han vivido** "
        "esa antigüedad en el calendario real (ver punto 6)."
    )

    # -----------------------------------------------------------------------
    st.subheader("2. Los roles que configuras, explicados uno por uno")

    st.markdown("**🗂️ Columna de Cohorte** — responde: *¿a qué grupo pertenece cada registro?*")
    st.markdown(
        "- Úsala con una **fecha** (ej. Fecha de Inscripción) cuando quieras agrupar por *cuándo* "
        "llegó cada quien — el caso más común.\n"
        "- Úsala con una **categoría** (ej. Ciudad, Canal de Captación, Programa) cuando quieras "
        "comparar *segmentos* en vez de momentos en el tiempo.\n"
        "- 💡 *Pista para reconocerla en tu archivo*: es la columna con la que responderías "
        "\"¿de qué grupo/mes/canal es esta persona?\"."
    )

    st.markdown("**📈 Columna de Observación / Antigüedad** — responde: *¿cuánto tiempo ha pasado?*")
    st.markdown(
        "- Debe ser otra fecha, u otro contador de periodo, que se pueda **restar** de la Columna "
        "de Cohorte (por eso la app solo te deja elegir columnas de la misma \"familia\" — ver el "
        "punto 3).\n"
        "- Si la omites, no hay evolución: solo comparas los grupos en un instante fijo (sin curva).\n"
        "- 💡 *Pista*: si tu archivo ya trae una columna tipo \"Semestre Actual\" o \"Mes desde "
        "ingreso\", esa es tu columna de observación casi siempre."
    )

    st.markdown("**🆔 Columna de ID de Entidad** — responde: *quién es quién, a través del tiempo*")
    st.markdown(
        "- Documento, cédula, código de estudiante, email — cualquier identificador único y "
        "repetible de la misma persona/cliente.\n"
        "- Sin ella, la retención se cuenta por **filas**, no por personas distintas — sigue "
        "funcionando, pero es menos preciso si una misma persona puede aparecer varias veces.\n"
        "- 💡 *Pista*: la app ya la sugiere sola cuando detecta una columna con valores 100% únicos."
    )

    st.markdown("**🚦 Columna de Estado** — responde: *sigue activo, se graduó o abandonó?*")
    st.markdown(
        "- Ej. \"Matriculado\", \"Desertor\", \"Graduado\". Permite calcular Abandono/Conversión "
        "sobre el **estado real** de cada persona, no solo sobre si aparece o no en los datos.\n"
        "- Tras elegirla, mapeas cada valor único a un bucket (Retenido, Abandono, Convertido, "
        "Pendiente) — la app te sugiere un mapeo automático por palabras clave, revísalo igual.\n"
        "- 💡 *Pista*: si no tienes esta columna, la app sigue midiendo retención por presencia/"
        "ausencia en los datos, sin este nivel de detalle."
    )

    st.markdown("**🔢 Columna de Métrica** — responde: *además de contar personas, quiero sumar algo*")
    st.markdown(
        "- Cualquier columna numérica: ingresos, costos, NPS, edad promedio. Se combina con una "
        "función de agregación (Suma, Promedio, Mediana, Máximo, Mínimo).\n"
        "- Es opcional: sin ella, la matriz simplemente cuenta entidades."
    )

    # -----------------------------------------------------------------------
    st.subheader("3. Glosario: qué significa el \"Tipo detectado\" en la pestaña 📂 Datos")
    st.markdown(
        "La app detecta el tipo de cada columna automáticamente (nunca por el nombre, solo por su "
        "contenido), y de ahí depende qué rol puede cumplir:"
    )
    st.markdown(
        "| Tipo detectado | Qué significa en la práctica | Para qué sirve |\n"
        "|---|---|---|\n"
        "| `fecha` | Fechas reales (día/mes/año) | Cohorte u Observación — el caso clásico |\n"
        "| `periodo_ordinal` | Texto tipo \"2025-2\" (año-semestre/trimestre) | Cohorte u Observación, si ambas son del mismo formato |\n"
        "| `numerico_categorico` | Número con pocos valores repetidos (ej. Semestre 1-10, Estrato) | Cohorte, Observación, Filtro o Métrica |\n"
        "| `numerico_continuo` | Número con muchos valores distintos (ej. Ingresos, Edad) | Métrica o Observación (no Cohorte ni Filtro) |\n"
        "| `categorico` | Texto repetido (ej. Ciudad, Programa) | Cohorte, Estado o Filtro |\n"
        "| `identificador` | Texto único por fila (documento, email) | ID de Entidad — nunca Cohorte ni Métrica |\n"
        "| `booleano` | Verdadero/Falso | Cohorte, Estado o Filtro |\n"
    )
    st.info(
        "**Regla rápida**: si una columna es `identificador`, es tu ID de entidad. Si es `fecha` o "
        "`periodo_ordinal`, es tu mejor candidata a Cohorte u Observación. Si es `numerico_continuo`, "
        "úsala como Métrica, no como Cohorte (números casi todos distintos no agrupan nada)."
    )

    # -----------------------------------------------------------------------
    st.subheader("4. Cómo se calcula la antigüedad, con números reales")
    st.markdown(
        "Ejemplo: una estudiante se inscribió en **Marzo 2025** (su cohorte) y hoy está en el "
        "periodo **Junio 2025** (su observación). Con granularidad Mes:"
    )
    st.latex(r"\text{antigüedad} = \text{ordinal(Junio)} - \text{ordinal(Marzo)} = 3 \text{ meses}")
    st.markdown(
        "El \"inicio\" siempre se toma como la primera aparición de **esa misma entidad** (no de la "
        "fila), así que si más adelante esa estudiante tiene otro registro en Julio, su antigüedad "
        "en esa fila es 4, no 0 — sigue perteneciendo a su cohorte de Marzo."
    )

    # -----------------------------------------------------------------------
    st.subheader("5. ¿Evento o Snapshot? Pregúntate: ¿mis filas son personas o son eventos?")
    st.markdown(
        "- **Snapshot** (detectado automáticamente si cada entidad aparece **una sola vez**): tu "
        "archivo es una foto actual — una fila por estudiante con su antigüedad de hoy. Es el caso "
        "más común en reportes de matrícula/CRM exportados directamente.\n"
        "- **Evento** (si el ID de entidad se repite en varias filas): tu archivo es un historial — "
        "cada fila es una actividad (login, compra, pago) en un momento distinto de la misma persona.\n\n"
        "No necesitas elegirlo tú: la app lo detecta sola. Solo fuerza el modo manualmente si sabes "
        "que tus datos son ambiguos (por ejemplo, pocas personas con eventos repetidos por error)."
    )

    # -----------------------------------------------------------------------
    st.subheader("6. Por qué algunas celdas muestran \"-\" en vez de 0%")
    st.markdown(
        "Una cohorte de **este mes** no puede tener \"6 meses de antigüedad\" todavía — no es que "
        "hayan abandonado, es que ese futuro **aún no ha ocurrido** en el calendario. La app "
        "distingue esto automáticamente: `-` significa \"todavía no aplica\", `0%` significa "
        "\"ya pudo pasar, y no pasó\". Confundir ambas cosas es el error más común al leer una "
        "matriz de cohortes a mano — aquí no hace falta, la app ya lo separa por ti."
    )

    # -----------------------------------------------------------------------
    st.subheader("7. Retención, Abandono y Conversión — la diferencia práctica")
    st.markdown(
        "- **Retención**: de los que empezaron, ¿qué % sigue ahí en la edad N? Es la métrica base "
        "de toda la matriz.\n"
        "- **Abandono** = 1 − Retención, edad por edad — la misma información, leída al revés.\n"
        "- **Conversión / Abandono basados en Estado**: en vez de solo mirar si la persona *aparece* "
        "en los datos, mira su **estado real** más reciente (Matriculado, Desertor, Graduado...) — "
        "más preciso si tienes esa columna, porque alguien puede seguir \"apareciendo\" en los datos "
        "sin realmente seguir activo."
    )

    # -----------------------------------------------------------------------
    st.subheader("8. Guía rápida: combinaciones típicas que funcionan")
    st.markdown(
        "- **Cohortes de calendario** (la más habitual): Cohorte = una fecha (ej. Fecha de "
        "Inscripción) + Observación = otra fecha (ej. Fecha de Matrícula), con Granularidad = Mes "
        "o Semestre.\n"
        "- **Cohortes por avance ya calculado**: Cohorte = un contador (ej. Semestre de Ingreso) + "
        "Observación = otro contador de la misma escala (ej. Semestre Actual) — sin granularidad, "
        "la resta ya está en unidades correctas.\n"
        "- **Segmentación pura**: Cohorte = una categoría (ej. Ciudad, Canal de Captación) sin "
        "Observación — compara segmentos en un instante, sin curva de tiempo.\n"
    )
    st.warning(
        "⚠️ No mezcles una fecha con un contador ya calculado (ej. Fecha de Inscripción como "
        "Cohorte + Semestre Actual como Observación): son escalas distintas y la app lo rechaza "
        "con un mensaje claro en vez de dar un resultado sin sentido."
    )

    st.subheader("9. Insights automáticos")
    st.markdown(
        "Reglas deterministas, no modelos de machine learning: ranking de cohortes por retención "
        "promedio, pendiente de una regresión lineal simple (`numpy.polyfit`) sobre el tamaño y la "
        "retención de cohortes sucesivas para detectar tendencias, y z-score sobre la retención "
        "promedio de cada cohorte para marcar anomalías (|z| > 2)."
    )
