# Motor Universal de Cohortes

Aplicación Streamlit para análisis de cohortes con **cualquier** archivo Excel o CSV: detecta
automáticamente tipos de columna, permite definir cohortes por fecha, periodo o categoría, calcula
antigüedad, retención, abandono y conversión, y genera matrices, heatmaps, curvas y un dashboard
ejecutivo, sin configuración específica a un dominio.

## Cómo ejecutar

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

Carga tu archivo desde el panel lateral. La pestaña **📐 Metodología** dentro de la propia app
explica en detalle cada fórmula usada (asignación de cohorte, cálculo de antigüedad, retención en
modo evento/snapshot, censura por tiempo insuficiente, abandono y conversión).

## Arquitectura

```
app/
├── app.py                # Streamlit entrypoint — orquesta las 7 pestañas y el estado de sesión
├── requirements.txt
├── core/
│   ├── data_loader.py    # Carga csv/xlsx: detección de hoja, encabezado real y codificación
│   ├── profiling.py      # Inferencia de tipos de columna, resumen de dataset, nulos, stats
│   ├── cohort_engine.py  # Asignación de cohorte + cálculo de antigüedad (modo evento/snapshot)
│   ├── metrics.py         # Matrices cohorte×edad: conteo, métrica, retención, abandono, conversión
│   ├── filters.py         # Filtros dinámicos combinables (multiselect, rango de fecha, rango numérico)
│   ├── visualizations.py  # Figuras Plotly: heatmap, curvas de retención/abandono, KPIs
│   └── insights.py        # Observaciones automáticas basadas en reglas (ranking, tendencia, anomalías)
└── utils/
    ├── types.py           # Enums y dataclasses compartidos (ColumnType, Granularity, CohortConfig...)
    └── formatting.py      # Formato de %, números y división segura
```

### Flujo de procesamiento

1. **Carga** (`data_loader`): detecta csv/xlsx, hoja, y la fila de encabezado real (soporta archivos
   con filas de título/banner antes del encabezado, comunes en exportes reales).
2. **Perfilado** (`profiling`): clasifica cada columna en fecha, periodo ordinal (ej. `2025-2`),
   numérica continua, numérico-categórica, categórica o identificador — por estructura, no por
   nombre, para que el motor sea genérico.
3. **Configuración** (UI): el usuario elige columna de cohorte, columna de observación/antigüedad
   (opcional), granularidad, ID de entidad (opcional), columna de estado + mapeo a buckets de
   negocio (opcional), métrica + agregación, y filtros dinámicos.
4. **Motor de cohortes** (`cohort_engine`): produce una tabla larga `[entity_id, cohort_key, age,
   status_bucket?, metric_value?]`, detectando automáticamente si el dataset es un log de eventos
   (varias filas por entidad) o un snapshot (una fila por entidad con antigüedad actual).
5. **Métricas** (`metrics`): matrices cohorte×edad de conteo, métrica, retención (%), abandono (%) y
   conversión, con censura de celdas donde una cohorte reciente aún no pudo alcanzar esa edad.
6. **Visualización** (`visualizations`) e **Insights** (`insights`): heatmap, curvas, KPIs y
   observaciones automáticas explicables (ranking, tendencia por regresión lineal, z-score).

### Notas de diseño

- Ninguna lógica depende de nombres de columna específicos: todo se basa en el tipo estructural
  detectado (fecha, numérica, categórica, identificador, periodo ordinal), por lo que funciona igual
  con datos de estudiantes, clientes, suscriptores, etc.
- Columnas categóricas se convierten a `category` dtype tras el perfilado para memoria y velocidad
  de `groupby` en datasets grandes; todos los cómputos son vectorizados (sin loops fila-a-fila).
