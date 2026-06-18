"""
Paso 7 — Distribuir Datasets por Carrera / Programa
Basado en: 7_Distribuir_DF's.ipynb

Entradas: DF_DEMOGRAFICOS_2_RELLENADO.csv
          DF_ACADEMICO_3_FINAL.csv
          DF_DESERTORES.csv
Salida  : outputs/por_carrera/
              demograficos/   DF_{PROGRAMA}_DEMOGRAFICOS.csv
              academicos/     DF_{PROGRAMA}_ACADEMICOS.csv
              desertores/     DF_{PROGRAMA}_DESERTORES.csv
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def _safe(nombre: str) -> str:
    """Convierte nombre de programa en string seguro para nombre de archivo."""
    return (
        nombre
        .replace("/", "_").replace(":", "_").replace(" ", "_")
        .replace("(", "").replace(")", "")
    )


def _distribuir(df: pd.DataFrame, directorio: Path, sufijo: str) -> Dict[str, int]:
    directorio.mkdir(parents=True, exist_ok=True)
    resumen: Dict[str, int] = {}
    for prog in df["PROGRAMA"].unique():
        sub = df[df["PROGRAMA"] == prog].copy()
        archivo = directorio / f"DF_{_safe(str(prog))}_{sufijo}.csv"
        sub.to_csv(archivo, index=False)
        resumen[str(prog)] = len(sub)
    return resumen


def ejecutar(
    df_demograficos: pd.DataFrame,
    df_academico: pd.DataFrame,
    df_desertores: pd.DataFrame,
    output_base: Path,
) -> Tuple[dict, dict]:
    """
    Genera un CSV por programa para cada tipo de dataset.

    Args:
        df_demograficos : DF_DEMOGRAFICOS_2_RELLENADO
        df_academico    : DF_ACADEMICO_3_FINAL
        df_desertores   : DF_DESERTORES
        output_base     : Directorio raíz de salida (outputs/)

    Returns:
        (archivos_generados, stats)
    """
    stats: dict = {"paso": 7}
    logger.info("[Paso 7] Inicio")

    carpeta = output_base / "por_carrera"

    res_demo = _distribuir(df_demograficos, carpeta / "demograficos", "DEMOGRAFICOS")
    res_acad = _distribuir(df_academico,    carpeta / "academicos",   "ACADEMICOS")
    res_des  = _distribuir(df_desertores,   carpeta / "desertores",   "DESERTORES")

    stats.update({
        "programas_demograficos": len(res_demo),
        "programas_academicos":   len(res_acad),
        "programas_desertores":   len(res_des),
        "total_archivos": len(res_demo) + len(res_acad) + len(res_des),
    })
    archivos = {"demograficos": res_demo, "academicos": res_acad, "desertores": res_des}
    logger.info(f"[Paso 7] Fin — {stats['total_archivos']} archivos generados")
    return archivos, stats
