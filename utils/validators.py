"""Validación de DataFrames y archivos del pipeline."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def cargar_archivo(ruta: str | Path) -> Optional[pd.DataFrame]:
    """Carga CSV o Excel y retorna DataFrame, o None si falla."""
    p = Path(ruta)
    try:
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p, low_memory=False)
        elif p.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(p)
        else:
            logger.error(f"Formato no soportado: {p.suffix}")
            return None
    except Exception as e:
        logger.error(f"Error cargando {ruta}: {e}")
        return None


def validar_no_vacio(df: Optional[pd.DataFrame], nombre: str) -> Tuple[bool, List[str]]:
    if df is None or len(df) == 0:
        return False, [f"El dataset '{nombre}' está vacío o no pudo cargarse."]
    return True, []


def validar_columnas(df: pd.DataFrame, requeridas: List[str], nombre: str) -> Tuple[bool, List[str]]:
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        return False, [f"[{nombre}] Columnas faltantes: {faltantes}"]
    return True, []
