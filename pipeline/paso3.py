"""
Paso 3 — Rellenar TIPO_INSTITUCION (imputación probabilística)
Basado en: 3_Rellenar_Tipo_Institucion.ipynb

Entrada : DF_DEMOGRAFICOS_1_LIMPIO.csv  (salida del Paso 1)
Salida  : outputs/intermedios/DF_DEMOGRAFICOS_2_RELLENADO.csv

Lógica  : Los registros con TIPO_INSTITUCION == 2 (desconocido) se imputan
          usando probabilidades derivadas del año de ingreso, estrato,
          comunidad negra, pueblo indígena y rango de edad, tal como fue
          calculado manualmente en el notebook original.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BINS_EDAD  = [0, 18, 28, 60, np.inf]
LABELS_EDAD = ["Adolescentes", "Jóvenes", "Adultos", "Mayores"]

# Probabilidades por año de ingreso (extraídas del notebook 3)
# Cada entrada: estrato→p_privado, p_negra_si/no, p_indigena_si/no,
#               edad→p_privado, función forzar_cero
PROBS: dict = {
    2018: {
        "estrato":       {1: 0.05, 2: 0.16, 3: 0.1818},
        "p_negra_si":    0.0,    "p_negra_no":    0.1406,
        "p_indigena_si": 0.0,    "p_indigena_no": 0.1475,
        "edad":          {"Adolescentes": 0.0, "Jóvenes": 0.1379, "Adultos": 0.1667, "Mayores": 0.0},
        "forzar_cero":   lambda r: r["COMUNIDAD_NEGRA"] == 1 or r["PUEBLO_INDIGENA"] == 1 or str(r.get("RANGOS_EDAD","")) == "Adolescentes",
    },
    2019: {
        "estrato":       {1: 0.1091, 2: 0.1515, 3: 0.1538, 4: 0.50},
        "p_negra_si":    0.0,    "p_negra_no":    0.1414,
        "p_indigena_si": 0.3333, "p_indigena_no": 0.1237,
        "edad":          {"Adolescentes": 0.50, "Jóvenes": 0.1205, "Adultos": 0.0714, "Mayores": 0.0},
        "forzar_cero":   lambda r: r["COMUNIDAD_NEGRA"] == 1,
    },
    2020: {
        "estrato":       {1: 0.1277, 2: 0.40, 3: 0.1538, 4: 0.0},
        "p_negra_si":    0.0,    "p_negra_no":    0.20,
        "p_indigena_si": 0.1111, "p_indigena_no": 0.20,
        "edad":          {"Adolescentes": 0.1667, "Jóvenes": 0.20, "Adultos": 0.1667, "Mayores": 0.0},
        "forzar_cero":   lambda r: r["COMUNIDAD_NEGRA"] == 1,
    },
    2021: {
        "estrato":       {1: 0.094, 2: 0.1585, 3: 0.1818},
        "p_negra_si":    0.0,    "p_negra_no":    0.1284,
        "p_indigena_si": 0.0,    "p_indigena_no": 0.1359,
        "edad":          {"Adolescentes": 0.0769, "Jóvenes": 0.1429, "Adultos": 0.0606, "Mayores": 0.0},
        "forzar_cero":   lambda r: r["COMUNIDAD_NEGRA"] == 1,
    },
    2022: {
        "estrato":       {1: 0.1367, 2: 0.2546, 3: 0.380, 4: 1.0},
        "p_negra_si":    0.125,  "p_negra_no":    0.2115,
        "p_indigena_si": 0.160,  "p_indigena_no": 0.2153,
        "edad":          {"Adolescentes": 0.3019, "Jóvenes": 0.1875, "Adultos": 0.2553, "Mayores": 0.0},
        "forzar_cero":   lambda r: False,
    },
    2023: {
        "estrato":       {1: 0.1367, 2: 0.2546, 3: 0.380, 4: 1.0},
        "p_negra_si":    0.125,  "p_negra_no":    0.2115,
        "p_indigena_si": 0.160,  "p_indigena_no": 0.2153,
        "edad":          {"Adolescentes": 0.3019, "Jóvenes": 0.1875, "Adultos": 0.2553, "Mayores": 0.0},
        "forzar_cero":   lambda r: False,
    },
}


def _imputar_fila(row: pd.Series, p: dict) -> int:
    if row["TIPO_INSTITUCION"] != 2:
        return int(row["TIPO_INSTITUCION"])

    pe  = p["estrato"].get(row.get("ESTRATO"), 0.0)
    pn  = p["p_negra_si"]    if row["COMUNIDAD_NEGRA"]  == 1 else p["p_negra_no"]
    pi  = p["p_indigena_si"] if row["PUEBLO_INDIGENA"]  == 1 else p["p_indigena_no"]
    pe2 = p["edad"].get(str(row.get("RANGOS_EDAD", "")), 0.0)

    prob = max(0.0, min(1.0, (pe + pn + pi + pe2) / 4))
    if p["forzar_cero"](row):
        prob = 0.0

    return int(np.random.choice([1, 0], p=[prob, 1 - prob]))


def _imputar_anio(df: pd.DataFrame, anio: int) -> pd.DataFrame:
    if anio not in PROBS:
        return df
    p = PROBS[anio]

    # Detectar filas del año con TIPO_INSTITUCION==2
    # Forzar str antes de split para evitar errores con floats/NaN
    try:
        anios_col = (
            df["PERIDO_INGRESO"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.split("-")
            .str[0]
        )
        mask_anio = pd.to_numeric(anios_col, errors="coerce").fillna(-1).astype(int) == anio
    except Exception:
        return df
    mask = mask_anio & (df["TIPO_INSTITUCION"] == 2)
    if mask.sum() == 0:
        return df

    sub = df[mask].copy()
    sub["RANGOS_EDAD"] = pd.cut(sub["EDAD_INGRESO"], bins=BINS_EDAD, labels=LABELS_EDAD,
                                right=False, include_lowest=True)
    sub["TIPO_INSTITUCION"] = sub.apply(lambda r: _imputar_fila(r, p), axis=1)

    mapa = sub.set_index("ID_EST")["TIPO_INSTITUCION"].to_dict()
    df["TIPO_INSTITUCION"] = df.apply(
        lambda r: mapa.get(r["ID_EST"], r["TIPO_INSTITUCION"]), axis=1
    )
    return df


def ejecutar(df_demo_limpio: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Imputa TIPO_INSTITUCION en el dataset demográfico limpio.

    Args:
        df_demo_limpio: Resultado del Paso 1.

    Returns:
        (df_rellenado, stats)
    """
    stats: dict = {"paso": 3}
    df = df_demo_limpio.copy()
    nulos_ini = int((df["TIPO_INSTITUCION"] == 2).sum())
    logger.info(f"[Paso 3] Inicio — TIPO_INSTITUCION==2: {nulos_ini}")

    # Obtener años disponibles en el dataset
    anios = []
    try:
        anios = (
            df["PERIDO_INGRESO"]
            .fillna("").astype(str).str.strip()
            .str.split("-").str[0]
            .pipe(lambda s: pd.to_numeric(s, errors="coerce"))
            .dropna().astype(int)
            .unique().tolist()
        )
    except Exception:
        pass

    for anio in sorted(anios):
        df = _imputar_anio(df, anio)
        logger.info(f"  Año {anio} imputado.")

    nulos_fin = int((df["TIPO_INSTITUCION"] == 2).sum())
    stats.update({
        "nulos_tipo_institucion_inicial": nulos_ini,
        "nulos_tipo_institucion_final":   nulos_fin,
        "imputados": nulos_ini - nulos_fin,
        "total_final": len(df),
    })
    logger.info(f"[Paso 3] Fin — imputados: {stats['imputados']} | restantes: {nulos_fin}")
    return df, stats
