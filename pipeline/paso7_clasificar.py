"""
pipeline/paso7_clasificar.py
=============================
Paso 7 — Clasificación de Estudiantes Entrenables y No Clasificables
RNN-ALERT · UNIMAYOR · 2026

Este paso separa los estudiantes con trayectoria suficiente para entrenamiento
respecto a los estudiantes aptos para predicción temprana.

Regla principal:
  • Clasificables / entrenables: estudiantes con 4 o más semestres registrados.
  • No clasificables para predicción temprana: estudiantes con menos de 4
    semestres, pero que NO cumplen condición de deserción.
  • Excluidos por deserción temprana/antigua: estudiantes con menos de 4
    semestres y 3 o más semestres sin actividad, según DF_DESERTORES.

La condición de predicción temprana evita que estudiantes antiguos con poca
historia académica queden como "no clasificables" cuando en realidad ya deben
ser tratados como desertores.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

SEMESTRES_MINIMOS = 4
SEMESTRES_INACTIVIDAD_DESERCION = 3


def _normalizar_id(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza ID_EST sin alterar el objeto original."""
    out = df.copy()
    if "ID_EST" in out.columns:
        out["ID_EST"] = pd.to_numeric(out["ID_EST"], errors="coerce").fillna(-1).astype(int)
    return out


def _filtrar_por_pares(df: pd.DataFrame, pares: set[tuple]) -> pd.DataFrame:
    """Filtra por (ID_EST, PROGRAMA) si existen ambas columnas; si no, por ID_EST."""
    if df is None or df.empty:
        return pd.DataFrame(columns=getattr(df, "columns", []))
    base = _normalizar_id(df)
    if "ID_EST" in base.columns and "PROGRAMA" in base.columns:
        tmp = base.copy()
        tmp["_PAIR"] = list(zip(tmp["ID_EST"], tmp["PROGRAMA"].astype(str)))
        res = tmp[tmp["_PAIR"].isin(pares)].drop(columns=["_PAIR"]).copy()
        return res
    ids = {i for i, _ in pares}
    return base[base["ID_EST"].isin(ids)].copy()


def ejecutar(
    df_acad: pd.DataFrame,
    df_demo: pd.DataFrame,
    df_desertores: pd.DataFrame,
    ruta_auditoria: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Separa estudiantes clasificables y no clasificables para predicción temprana.

    Returns:
        df_clasif_acad    : Académico solo con estudiantes clasificables.
        df_clasif_demo    : Demográfico solo con estudiantes clasificables.
        df_no_clas_acad   : Académico de estudiantes no clasificables válidos para predicción temprana.
        df_no_clas_demo   : Demográfico de estudiantes no clasificables válidos para predicción temprana.
        stats             : Estadísticas del proceso.
    """
    stats: dict = {"paso": "7_clasificar"}
    logger.info(
        "[Paso7] Clasificando estudiantes "
        f"(entrenables >= {SEMESTRES_MINIMOS} semestres; "
        f"predicción temprana sin {SEMESTRES_INACTIVIDAD_DESERCION}+ semestres inactivos)"
    )

    df_acad_n = _normalizar_id(df_acad)
    df_demo_n = _normalizar_id(df_demo)
    df_des_n = _normalizar_id(df_desertores) if df_desertores is not None else pd.DataFrame()

    # 1. Máximo de semestres registrados por estudiante-programa.
    sem_max = (
        df_acad_n.groupby(["ID_EST", "PROGRAMA"])["NUMERO_SEMESTRE"]
        .max()
        .reset_index()
        .rename(columns={"NUMERO_SEMESTRE": "MAX_SEM"})
    )
    sem_max["PROGRAMA"] = sem_max["PROGRAMA"].astype(str)

    # 2. Incorporar estado de deserción calculado en Paso 6.
    columnas_des = [c for c in [
        "ID_EST", "PROGRAMA", "ULTIMO_SEMESTRE_REGISTRADO",
        "SEMESTRES_SIN_ACTIVIDAD", "DESERTOR"
    ] if c in df_des_n.columns]

    if {"ID_EST", "PROGRAMA", "DESERTOR"}.issubset(df_des_n.columns):
        des_ref = df_des_n[columnas_des].drop_duplicates(subset=["ID_EST", "PROGRAMA"]).copy()
        des_ref["PROGRAMA"] = des_ref["PROGRAMA"].astype(str)
        sem_max = sem_max.merge(des_ref, on=["ID_EST", "PROGRAMA"], how="left")
    else:
        sem_max["DESERTOR"] = 0
        sem_max["SEMESTRES_SIN_ACTIVIDAD"] = np.nan

    sem_max["DESERTOR"] = pd.to_numeric(sem_max.get("DESERTOR", 0), errors="coerce").fillna(0).astype(int)
    if "SEMESTRES_SIN_ACTIVIDAD" not in sem_max.columns:
        sem_max["SEMESTRES_SIN_ACTIVIDAD"] = np.nan
    sem_max["SEMESTRES_SIN_ACTIVIDAD"] = pd.to_numeric(sem_max["SEMESTRES_SIN_ACTIVIDAD"], errors="coerce")

    # 3. Reglas de clasificación.
    sem_max["CLASIFICABLE"] = sem_max["MAX_SEM"] >= SEMESTRES_MINIMOS
    sem_max["NO_CLASIFICABLE_PREDICCION"] = (
        (sem_max["MAX_SEM"] < SEMESTRES_MINIMOS)
        & (sem_max["DESERTOR"] == 0)
        & (
            sem_max["SEMESTRES_SIN_ACTIVIDAD"].isna()
            | (sem_max["SEMESTRES_SIN_ACTIVIDAD"] < SEMESTRES_INACTIVIDAD_DESERCION)
        )
    )
    sem_max["EXCLUIDO_DESERTOR_BAJA_TRAYECTORIA"] = (
        (sem_max["MAX_SEM"] < SEMESTRES_MINIMOS)
        & (~sem_max["NO_CLASIFICABLE_PREDICCION"])
    )

    ids_clasif = set(zip(
        sem_max.loc[sem_max["CLASIFICABLE"], "ID_EST"],
        sem_max.loc[sem_max["CLASIFICABLE"], "PROGRAMA"],
    ))
    ids_no_clas = set(zip(
        sem_max.loc[sem_max["NO_CLASIFICABLE_PREDICCION"], "ID_EST"],
        sem_max.loc[sem_max["NO_CLASIFICABLE_PREDICCION"], "PROGRAMA"],
    ))
    ids_excluidos = set(zip(
        sem_max.loc[sem_max["EXCLUIDO_DESERTOR_BAJA_TRAYECTORIA"], "ID_EST"],
        sem_max.loc[sem_max["EXCLUIDO_DESERTOR_BAJA_TRAYECTORIA"], "PROGRAMA"],
    ))

    logger.info(
        f"[Paso7] Clasificables: {len(ids_clasif):,} | "
        f"No clasificables predicción: {len(ids_no_clas):,} | "
        f"Excluidos por deserción/baja trayectoria: {len(ids_excluidos):,}"
    )

    # 4. Separar datasets académicos y demográficos por par ID_EST-PROGRAMA.
    df_clasif_acad = _filtrar_por_pares(df_acad_n, ids_clasif)
    df_no_clas_acad = _filtrar_por_pares(df_acad_n, ids_no_clas)
    df_clasif_demo = _filtrar_por_pares(df_demo_n, ids_clasif)
    df_no_clas_demo = _filtrar_por_pares(df_demo_n, ids_no_clas)

    # 5. Estadísticas.
    stats.update({
        "total_pares_id_programa": int(len(sem_max)),
        "clasificables": int(len(ids_clasif)),
        "no_clasificables": int(len(ids_no_clas)),
        "excluidos_desertor_baja_trayectoria": int(len(ids_excluidos)),
        "umbral_semestres": SEMESTRES_MINIMOS,
        "umbral_semestres_sin_actividad_desercion": SEMESTRES_INACTIVIDAD_DESERCION,
        "filas_acad_clasif": int(len(df_clasif_acad)),
        "filas_acad_no_clas": int(len(df_no_clas_acad)),
        "estudiantes_demo_clasif": int(df_clasif_demo["ID_EST"].nunique()) if "ID_EST" in df_clasif_demo.columns else int(len(df_clasif_demo)),
        "estudiantes_demo_no_clas": int(df_no_clas_demo["ID_EST"].nunique()) if "ID_EST" in df_no_clas_demo.columns else int(len(df_no_clas_demo)),
    })

    # 6. Auditoría.
    if ruta_auditoria is not None:
        p = Path(ruta_auditoria)
        p.mkdir(parents=True, exist_ok=True)
        sem_max_export = sem_max.copy()
        sem_max_export["ESTADO"] = np.select(
            [
                sem_max_export["CLASIFICABLE"],
                sem_max_export["NO_CLASIFICABLE_PREDICCION"],
                sem_max_export["EXCLUIDO_DESERTOR_BAJA_TRAYECTORIA"],
            ],
            [
                "CLASIFICABLE_ENTRENAMIENTO",
                "NO_CLASIFICABLE_PREDICCION_TEMPRANA",
                "EXCLUIDO_DESERTOR_BAJA_TRAYECTORIA",
            ],
            default="SIN_CLASIFICAR",
        )
        sem_max_export.to_csv(p / "resumen_estudiantes_no_clasificables.csv", index=False)
        sem_max_export[sem_max_export["EXCLUIDO_DESERTOR_BAJA_TRAYECTORIA"]].to_csv(
            p / "estudiantes_excluidos_baja_trayectoria_desertores.csv", index=False
        )
        logger.info(f"[Paso7] Auditoría guardada en: {p}")

    logger.info(
        f"[Paso7] Completado — Entrenamiento: {len(ids_clasif):,} | "
        f"Predicción temprana: {len(ids_no_clas):,}"
    )
    return df_clasif_acad, df_clasif_demo, df_no_clas_acad, df_no_clas_demo, stats
