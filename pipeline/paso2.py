"""
Paso 2 — Formalizar Dataset Académico
Basado en: 2_Formalizar_dataset_academicos.ipynb

Entrada : DATASET_DATOS_ACADEMICOS.csv  (sube el usuario)
Salida  : outputs/intermedios/DF_ACADEMICOS_1_LIMPIO.csv
"""
from __future__ import annotations

import logging
import unicodedata
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

PROGRAMAS_EXCLUIR = [
    "ESPECIALIZACION EN ADMINISTRACION DE LA INFORMACION Y BASES DE DATOS",
    "ESPECIALIZACION EN GERENCIA FINANCIERA",
    "ESPECIALIZACION EN ALTA GERENCIA",
    "ESPECIALIZACION EN DISENO DE AMBIENTES",
    "MAESTRIA EN ADMINISTRACION DE NEGOCIOS MBA",
    "INGENIERIA MULTIMEDIA",
    "LICENCIATURA EN ESPANOL E INGLES",
    "LICENCIATURA EN MUSICA",
]


def limpiar_texto(texto: object) -> object:
    if pd.isna(texto):
        return texto
    texto = unicodedata.normalize("NFD", str(texto))
    return texto.encode("ascii", "ignore").decode("utf-8")


def get_integer_part(s: object) -> object:
    if pd.isna(s):
        return None
    s = str(s).strip()
    if s in ["", "nan", "none"]:
        return None
    try:
        return int(float(s.replace(",", "")))
    except ValueError:
        return None


def ejecutar(df_academico: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Formaliza el dataset académico (materias + notas detalladas).

    Args:
        df_academico: DataFrame cargado desde DATASET_DATOS_ACADEMICOS.csv

    Returns:
        (df_limpio, stats)
    """
    stats: dict = {"paso": 2}
    total_ini = len(df_academico)
    logger.info(f"[Paso 2] Inicio — {total_ini:,} filas")

    df = df_academico.copy()

    # 1. Limpiar nombres de columnas
    df.columns = [limpiar_texto(col) for col in df.columns]

    # 2. Limpiar valores texto
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(limpiar_texto)

    # 3. Normalizar ID_EST
    if "ID_EST" in df.columns:
        df["ID_EST"] = df["ID_EST"].astype(str).apply(get_integer_part)
        df["ID_EST"] = pd.to_numeric(df["ID_EST"], errors="coerce").fillna(0).astype(int)

    # 4. Excluir programas
    antes = len(df)
    df = df[~df["PROGRAMA"].isin(PROGRAMAS_EXCLUIR)].copy()
    stats["filas_excluidas_programa"] = antes - len(df)

    # 5. Eliminar duplicados exactos
    mask_dup = df.duplicated()
    stats["duplicados_eliminados"] = int(mask_dup.sum())
    df_limpio = df[~mask_dup].reset_index(drop=True)

    stats.update({
        "total_inicial":   total_ini,
        "total_final":     len(df_limpio),
        "filas_eliminadas": total_ini - len(df_limpio),
    })
    logger.info(f"[Paso 2] Fin — {len(df_limpio):,} filas | {stats['duplicados_eliminados']} dups")
    return df_limpio, stats
