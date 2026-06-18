"""
Paso 6 — Calcular Desertores
Basado en: 6_Calcular_Desertores.ipynb

Entradas: outputs/intermedios/DF_ACADEMICO_3_FINAL.csv   (salida Paso 5)
          outputs/intermedios/DF_DEMOGRAFICOS_2_RELLENADO.csv  (salida Paso 3)
Salida  : outputs/intermedios/DF_DESERTORES.csv

Clasificación:
  • DESERTOR = 0 → tiene PERIODO_GRADO  (graduado)
  • DESERTOR = 1 → ≥ 3 semestres sin actividad académica y sin grado
  • DESERTOR = 0 → activo (< 3 semestres sin actividad)

Referencia temporal: 2026-1
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PERIODO_REF = "2026-1"


def _periodo_num(periodo: object) -> float:
    """Convierte 'YYYY-S' a entero secuencial de semestres."""
    if periodo is None or (isinstance(periodo, float) and np.isnan(periodo)):
        return np.nan
    s = str(periodo).strip()
    if s in ("", "nan", "None", "<NA>"):
        return np.nan
    try:
        partes = s.split("-")
        if len(partes) != 2:
            return np.nan
        anio, sem = int(partes[0]), int(partes[1])
        return float(anio * 2 + (sem - 1))
    except Exception:
        return np.nan


def ejecutar(
    df_academico_final: pd.DataFrame,
    df_demografico_rellenado: pd.DataFrame,
) -> Tuple[pd.DataFrame, dict]:
    """
    Clasifica si cada estudiante es desertor.

    Args:
        df_academico_final       : Resultado del Paso 5 (con NUMERO_SEMESTRE).
        df_demografico_rellenado : Resultado del Paso 3 (con PERIDO_INGRESO, PERIODO_GRADO).

    Returns:
        (df_desertores, stats)
    """
    stats: dict = {"paso": 6}
    logger.info("[Paso 6] Inicio")

    ref = _periodo_num(PERIODO_REF)

    # 1. Último semestre registrado por (ID_EST, PROGRAMA)
    ultimo = (
        df_academico_final.groupby(["ID_EST", "PROGRAMA"])["NUMERO_SEMESTRE"]
        .max()
        .reset_index()
        .rename(columns={"NUMERO_SEMESTRE": "ULTIMO_SEMESTRE_REGISTRADO"})
    )

    # 2. Base demográfica (un registro por estudiante-programa)
    base = (
        df_demografico_rellenado[["ID_EST", "PROGRAMA", "PERIDO_INGRESO", "PERIODO_GRADO"]]
        .drop_duplicates()
        .copy()
    )

    # 3. Unir último semestre
    des = pd.merge(base, ultimo, on=["ID_EST", "PROGRAMA"], how="left")
    des["ULTIMO_SEMESTRE_REGISTRADO"] = des["ULTIMO_SEMESTRE_REGISTRADO"].fillna(0)

    # 4. Semestres transcurridos desde ingreso hasta 2026-1
    des["PERIODO_INGRESO_NUM"]    = des["PERIDO_INGRESO"].apply(_periodo_num)
    des["SEMESTRES_DESDE_INGRESO"] = ref - des["PERIODO_INGRESO_NUM"] + 1

    # 5. Semestres sin actividad
    des["SEMESTRES_SIN_ACTIVIDAD"] = (
        des["SEMESTRES_DESDE_INGRESO"] - des["ULTIMO_SEMESTRE_REGISTRADO"]
    ).clip(lower=0)
    # Graduados: 0 semestres sin actividad
    des.loc[des["PERIODO_GRADO"].notna(), "SEMESTRES_SIN_ACTIVIDAD"] = 0

    # 6. Clasificación
    def _clasificar(row: pd.Series) -> int:
        if pd.notna(row["PERIODO_GRADO"]):
            return 0
        return 1 if row["SEMESTRES_SIN_ACTIVIDAD"] >= 3 else 0

    des["DESERTOR"] = des.apply(_clasificar, axis=1)

    # 7. Dataset final
    df_des = des[
        ["ID_EST", "PROGRAMA", "ULTIMO_SEMESTRE_REGISTRADO",
         "SEMESTRES_SIN_ACTIVIDAD", "DESERTOR"]
    ].copy()

    stats.update({
        "total_estudiantes": len(df_des),
        "desertores":        int(df_des["DESERTOR"].sum()),
        "no_desertores":     int((df_des["DESERTOR"] == 0).sum()),
        "tasa_desercion_pct": round(
            df_des["DESERTOR"].sum() / len(df_des) * 100, 2
        ) if len(df_des) else 0,
    })
    logger.info(
        f"[Paso 6] Fin — desertores: {stats['desertores']} | "
        f"tasa: {stats['tasa_desercion_pct']}%"
    )
    return df_des, stats
