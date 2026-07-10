"""Carga genérica de archivos Excel/CSV: detección de hoja, encabezado real y codificación.

Los exportes reales de Excel suelen traer filas de título/banner antes del encabezado real
(p.ej. una fila con el nombre del reporte fusionada en varias columnas). Este módulo detecta esa
fila automáticamente en vez de asumir `header=0`, para que el motor funcione con cualquier archivo.
"""

from __future__ import annotations

import csv
from typing import BinaryIO, Optional, Union

import pandas as pd

FileLike = Union[str, BinaryIO]

MAX_HEADER_SCAN_ROWS = 15
_PLAUSIBLE_SEPARATORS = ",;\t|:"


def is_excel(filename: str) -> bool:
    """True si el nombre de archivo corresponde a un Excel soportado."""
    return filename.lower().endswith((".xlsx", ".xls"))


def is_csv(filename: str) -> bool:
    """True si el nombre de archivo corresponde a un CSV."""
    return filename.lower().endswith(".csv")


def list_sheets(file: FileLike) -> list[str]:
    """Devuelve los nombres de las hojas disponibles en un archivo Excel."""
    if hasattr(file, "seek"):
        file.seek(0)
    xls = pd.ExcelFile(file)
    return xls.sheet_names


def _looks_numeric(value: str) -> bool:
    try:
        float(value.replace(",", "").replace("%", "").replace("$", ""))
        return True
    except ValueError:
        return False


def _detect_header_row(raw: pd.DataFrame) -> int:
    """Heurística que localiza la fila de encabezado real entre las primeras filas de una hoja.

    Puntúa cada fila candidata por: proporción de celdas no nulas, proporción de valores de texto
    únicos (un encabezado no repite nombres de columna) y penaliza filas con muchos valores
    numéricos (más probable que sean datos, no nombres de columna). Se queda con la de mayor score.
    """
    n_cols = raw.shape[1] or 1
    best_row = 0
    best_score = float("-inf")
    scan_rows = min(MAX_HEADER_SCAN_ROWS, len(raw))
    for i in range(scan_rows):
        row = raw.iloc[i]
        non_null = int(row.notna().sum())
        if non_null == 0:
            continue
        values = [str(v).strip() for v in row if pd.notna(v)]
        unique_ratio = len(set(values)) / len(values) if values else 0.0
        fill_ratio = non_null / n_cols
        numeric_ratio = sum(_looks_numeric(v) for v in values) / len(values) if values else 1.0
        score = fill_ratio * 0.5 + unique_ratio * 0.4 - numeric_ratio * 0.3
        if score > best_score:
            best_score = score
            best_row = i
    return best_row


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Descarta filas/columnas completamente vacías y normaliza nombres de columna."""
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    empty_unnamed = df.columns.str.startswith("Unnamed:") & df.isna().all()
    df = df.loc[:, ~empty_unnamed]
    return df.reset_index(drop=True)


def load_excel(file: FileLike, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Carga una hoja de Excel detectando automáticamente la fila de encabezado real."""
    if hasattr(file, "seek"):
        file.seek(0)
    xls = pd.ExcelFile(file)
    sheet = sheet_name or xls.sheet_names[0]
    raw = xls.parse(sheet, header=None, nrows=MAX_HEADER_SCAN_ROWS)
    header_row = _detect_header_row(raw)
    df = xls.parse(sheet, header=header_row)
    return _clean_dataframe(df)


def _read_raw_bytes(file: FileLike) -> bytes:
    """Lee el contenido completo del archivo como bytes, sin consumir el stream para lecturas
    posteriores (deja el cursor en 0 si `file` es un objeto tipo archivo)."""
    if hasattr(file, "read"):
        file.seek(0)
        data = file.read()
        file.seek(0)
        return data
    with open(file, "rb") as f:
        return f.read()


def _detect_separator(sample_text: str) -> str:
    """Detecta el separador de un CSV con `csv.Sniffer`, restringido a delimitadores plausibles
    (`,` `;` tab `|` `:`).

    Sin esta restricción, un CSV de una sola columna (sin ningún delimitador real que detectar,
    ej. una lista de IDs o nombres) hace que `csv.Sniffer` elija un carácter cualquiera del propio
    contenido como si fuera el separador — partiendo los datos en columnas falsas de forma
    silenciosa. Si ningún delimitador plausible aparece de forma consistente, se usa ',' por
    defecto, que no tiene efecto si nunca aparece en los datos (columna única preservada tal cual).
    """
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=_PLAUSIBLE_SEPARATORS)
        return dialect.delimiter
    except csv.Error:
        return ","


def load_csv(file: FileLike) -> pd.DataFrame:
    """Carga un CSV probando codificaciones comunes, detectando el separador de forma acotada (ver
    `_detect_separator`) y la fila de encabezado real (un CSV exportado desde Excel puede conservar
    las mismas filas de título/banner que un .xlsx, así que no se puede asumir `header=0`)."""
    raw_bytes = _read_raw_bytes(file)
    if not raw_bytes.strip():
        raise ValueError("El archivo CSV está vacío.")

    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            sample = raw_bytes[:8192].decode(encoding)
            sep = _detect_separator(sample)

            if hasattr(file, "seek"):
                file.seek(0)
            raw = pd.read_csv(file, sep=sep, encoding=encoding, header=None, nrows=MAX_HEADER_SCAN_ROWS)
            header_row = _detect_header_row(raw)

            if hasattr(file, "seek"):
                file.seek(0)
            df = pd.read_csv(file, sep=sep, encoding=encoding, header=header_row)
            if df.empty or df.shape[1] == 0:
                raise pd.errors.EmptyDataError("El CSV no contiene columnas legibles.")
            return _clean_dataframe(df)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            last_error = exc
            continue
    raise ValueError(f"No se pudo leer el CSV con ninguna codificación soportada: {last_error}")


def load_dataset(file: FileLike, filename: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Punto de entrada único de carga: detecta csv/xlsx y delega en el loader correspondiente.

    Lanza `ValueError` con un mensaje legible si el formato no es soportado o el archivo está vacío.
    """
    if is_excel(filename):
        df = load_excel(file, sheet_name=sheet_name)
    elif is_csv(filename):
        df = load_csv(file)
    else:
        raise ValueError(f"Formato de archivo no soportado: '{filename}'. Usa .csv, .xlsx o .xls.")

    if df.empty or df.shape[1] == 0:
        raise ValueError("El archivo se cargó pero no contiene datos legibles.")
    return df
