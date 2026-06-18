"""
Paso 5 — Calcular Estadísticas Académicas por Semestre
Basado en: 5_Rellenar_DF_ACADEMICO.ipynb

Entradas: outputs/intermedios/DF_ACADEMICO_2_RELLENADO.csv  (salida Paso 4)
          outputs/intermedios/DF_DEMOGRAFICOS_2_RELLENADO.csv  (salida Paso 3)
Salida  : outputs/intermedios/DF_ACADEMICO_3_FINAL.csv

Lógica  : Agrega las notas por estudiante-programa-semestre y calcula:
            - PROMEDIO_ACADEMICO
            - MATERIAS_CURSADAS / APROBADAS / REPROBADAS
            - NOTA_MAXIMA / NOTA_MINIMA
            - TASA_APROBACION
            - ACUMULACION_MATERIAS_CURSADAS
          Luego une TIEMPO_RETENCION_EST desde el dataset demográfico rellenado.

NOTA: Este paso es el puente entre la rama demográfica (pasos 1→3) y
      la rama académica (pasos 2→4). Consolida ambos flujos.
"""
from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

NOTA_APROBACION = 3.0


def ejecutar(
    df_acad_rellenado: pd.DataFrame,
    df_demograficos_rellenado: pd.DataFrame,
) -> Tuple[pd.DataFrame, dict]:
    """
    Construye el DataFrame académico final con estadísticas por semestre.

    Args:
        df_acad_rellenado           : Resultado del Paso 4 (con columna SEMESTRE).
        df_demograficos_rellenado   : Resultado del Paso 3 (con TIEMPO_RETENCION_EST).

    Returns:
        (df_academico_final, stats)
    """
    stats: dict = {"paso": 5}
    logger.info("[Paso 5] Inicio")

    df = df_acad_rellenado.copy()

    # ── 1. Promedio por estudiante-programa-semestre ──────────────────────────
    avg = (
        df.groupby(["ID_EST", "PROGRAMA", "SEMESTRE"])
        .agg(PROMEDIO_ACADEMICO=("NOTA_DEFINITIVA", "mean"))
        .reset_index()
        .rename(columns={"SEMESTRE": "NUMERO_SEMESTRE"})
    )

    # ── 2. Estadísticas brutas por estudiante-semestre ────────────────────────
    raw = (
        df.groupby(["ID_EST", "SEMESTRE"])
        .agg(
            MATERIAS_CURSADAS  = ("NOTA_DEFINITIVA", "count"),
            MATERIAS_APROBADAS = ("NOTA_DEFINITIVA", lambda x: (x >= NOTA_APROBACION).sum()),
            MATERIAS_REPROBADAS= ("NOTA_DEFINITIVA", lambda x: (x <  NOTA_APROBACION).sum()),
            NOTA_MAXIMA        = ("NOTA_DEFINITIVA", "max"),
            NOTA_MINIMA        = ("NOTA_DEFINITIVA", "min"),
        )
        .reset_index()
        .rename(columns={"SEMESTRE": "NUMERO_SEMESTRE"})
    )

    # ── 3. Combinar ───────────────────────────────────────────────────────────
    final = avg.merge(raw, on=["ID_EST", "NUMERO_SEMESTRE"], how="left")

    # ── 4. Tasa de aprobación ─────────────────────────────────────────────────
    final["TASA_APROBACION"] = (
        (final["MATERIAS_APROBADAS"] / final["MATERIAS_CURSADAS"]) * 100
    ).fillna(0).round(2)

    # ── 5. Acumulación de materias cursadas ───────────────────────────────────
    final = final.sort_values(by=["ID_EST", "NUMERO_SEMESTRE"])
    final["ACUMULACION_MATERIAS_CURSADAS"] = (
        final.groupby(["ID_EST", "PROGRAMA"])["MATERIAS_CURSADAS"].cumsum()
    )

    # ── 6. Unir TIEMPO_RETENCION_EST desde demográficos ───────────────────────
    retencion = (
        df_demograficos_rellenado[["ID_EST", "PROGRAMA", "TIEMPO_RETENCION_EST"]]
        .drop_duplicates()
    )
    final = pd.merge(final, retencion, on=["ID_EST", "PROGRAMA"], how="left")

    stats.update({
        "total_final":        len(final),
        "estudiantes_unicos": int(final["ID_EST"].nunique()),
        "semestres_unicos":   int(final["NUMERO_SEMESTRE"].nunique()),
        "programas_unicos":   int(final["PROGRAMA"].nunique()),
    })
    logger.info(
        f"[Paso 5] Fin — {stats['total_final']:,} filas | "
        f"{stats['estudiantes_unicos']:,} estudiantes | "
        f"{stats['programas_unicos']} programas"
    )
    return final, stats
