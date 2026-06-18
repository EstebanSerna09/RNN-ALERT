"""
Paso 4 — Asignar Semestre a Materias
Basado en: 4_Asignar_Semestre_a_materias_y_retencion.ipynb

Entrada : outputs/intermedios/DF_ACADEMICOS_1_LIMPIO.csv  (salida del Paso 2)
          data/referencia/DF_PLANES_ESTUDIO_UNIMAYOR_FINAL.csv  (incluido en el proyecto)
Salida  : outputs/intermedios/DF_ACADEMICO_2_RELLENADO.csv

Lógica  : Merge entre el dataset académico limpio y el plan de estudios
          para asignar el número de semestre a cada materia cursada.
"""
from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def ejecutar(
    df_acad_limpio: pd.DataFrame,
    df_planes: pd.DataFrame,
) -> Tuple[pd.DataFrame, dict]:
    """
    Asigna el semestre a cada fila del dataset académico.

    Args:
        df_acad_limpio : Resultado del Paso 2 (DF_ACADEMICOS_1_LIMPIO).
        df_planes      : Plan de estudios (DF_PLANES_ESTUDIO_UNIMAYOR_FINAL).
                         Debe contener columnas: PROGRAMA, NOMBRE MODULO, SEMESTRE.

    Returns:
        (df_con_semestre, stats)
    """
    stats: dict = {"paso": 4}
    total_ini = len(df_acad_limpio)
    logger.info(f"[Paso 4] Inicio — {total_ini:,} filas")

    df = df_acad_limpio.copy()

    # Limpiar espacios en plan de estudios
    df_p = df_planes.copy()
    for col in df_p.columns:
        if df_p[col].dtype == "object":
            df_p[col] = df_p[col].str.strip()

    # Detectar nombre de la columna de módulos en el plan
    col_modulo = "NOMBRE MODULO" if "NOMBRE MODULO" in df_p.columns else "NOMBRE_MODULO"
    if col_modulo not in df_p.columns:
        # intentar cualquier columna que contenga "MODULO"
        candidatos = [c for c in df_p.columns if "MODULO" in c.upper()]
        col_modulo = candidatos[0] if candidatos else df_p.columns[1]

    df_p = df_p.rename(columns={col_modulo: "COMPONENTE_MODULO"})[
        ["PROGRAMA", "COMPONENTE_MODULO", "SEMESTRE"]
    ]
    # Eliminar duplicados en el plan para evitar multiplicación de filas
    df_p = df_p.drop_duplicates(subset=["PROGRAMA", "COMPONENTE_MODULO"])

    # Merge left: conserva todas las filas del académico
    df_con_semestre = pd.merge(
        df, df_p, on=["PROGRAMA", "COMPONENTE_MODULO"], how="left"
    )

    asignados = int(df_con_semestre["SEMESTRE"].notna().sum())
    sin_asig  = int(df_con_semestre["SEMESTRE"].isna().sum())

    stats.update({
        "total_inicial":        total_ini,
        "total_final":          len(df_con_semestre),
        "semestres_asignados":  asignados,
        "semestres_sin_asignar": sin_asig,
        "cobertura_pct": round(asignados / len(df_con_semestre) * 100, 2) if len(df_con_semestre) else 0,
    })
    logger.info(
        f"[Paso 4] Fin — {asignados:,}/{len(df_con_semestre):,} materias con semestre "
        f"({stats['cobertura_pct']}%)"
    )
    return df_con_semestre, stats
