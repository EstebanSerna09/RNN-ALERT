"""
Paso 1 — Formalizar Dataset Demográfico
Basado en: 1_Formalizar_dataset_demograficos.ipynb

Entrada : DATASET_DATOS_DEMOGRAFICOS.csv  (sube el usuario)
          data/referencia/DATASET_MUNICIPIOS_RURALES_COLOMBIA.csv  (incluido en el proyecto)
Salida  : outputs/intermedios/DF_DEMOGRAFICOS_1_LIMPIO.csv
"""
from __future__ import annotations

import logging
import unicodedata
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROGRAMAS_EXCLUIR = [
    "ESPECIALIZACION EN ADMINISTRACION DE LA INFORMACION Y BASES DE DATOS",
    "ESPECIALIZACION EN GERENCIA FINANCIERA",
    "ESPECIALIZACION EN ALTA GERENCIA",
    "ESPECIALIZACION EN DISENO DE AMBIENTES",
    "ESPECIALIZACION EN FORMULACION Y EVALUACION DE PROYECTOS",
    "MAESTRIA EN ADMINISTRACION DE NEGOCIOS MBA",
    "INGENIERIA MULTIMEDIA",
    "LICENCIATURA EN ESPANOL E INGLES",
    "LICENCIATURA EN MUSICA",
]


# ── helpers 100% seguros con cualquier tipo ────────────────────────────────────

def _safe_str(x: object) -> str:
    """Convierte cualquier valor a string limpio. Nunca falla."""
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except (TypeError, ValueError):
        pass
    return str(x).strip()


def _quitar_tildes(texto: str) -> str:
    """Elimina tildes de un string ya limpio."""
    t = unicodedata.normalize("NFD", texto)
    return t.encode("ascii", "ignore").decode("utf-8")


def _normalizar_col_texto(serie: pd.Series) -> pd.Series:
    """
    Convierte una columna a string limpio sin tildes, de forma vectorizada.
    Es la única función que toca las columnas de texto — garantiza que
    el resultado siempre sea dtype object con strings o NaN.
    """
    return (
        serie
        .fillna("")                   # NaN → ""
        .astype(str)                  # float/int/object → str
        .str.strip()                  # quitar espacios
        .apply(_quitar_tildes)        # quitar tildes
        .replace("nan", "")           # "nan" literal → ""
        .replace("None", "")          # "None" literal → ""
    )


def calcular_periodo_grado(fecha: object) -> object:
    """Convierte fecha de grado a string 'YYYY-S'."""
    try:
        if pd.isna(fecha):
            return pd.NA
    except (TypeError, ValueError):
        return pd.NA
    mes  = fecha.month
    anio = fecha.year
    if   mes in [1, 2, 3, 4]:  return f"{anio - 1}-2"
    elif mes in [5, 6]:         return f"{anio}-1"
    elif mes in [7, 8, 9, 10]:  return f"{anio}-1"
    else:                       return f"{anio}-2"


def calcular_tiempo_retencion(row: pd.Series) -> object:
    """Semestres de retención entre ingreso y grado (o 2026-1 si no graduado)."""
    pg = _safe_str(row.get("PERIODO_GRADO", ""))
    pi = _safe_str(row.get("PERIDO_INGRESO", ""))
    if not pi or pi in ("nan", "None"):
        return pd.NA
    if not pg or pg in ("nan", "None"):
        gy, gp = 2026, 1
    else:
        try:
            gy, gp = map(int, pg.split("-"))
        except Exception:
            return pd.NA
    try:
        iy, ip = map(int, pi.split("-"))
        return (gy * 2 + gp) - (iy * 2 + ip) + 1
    except Exception:
        return pd.NA


# ── función principal ──────────────────────────────────────────────────────────

def ejecutar(df_demo: pd.DataFrame, df_rural: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Formaliza el dataset demográfico.

    Args:
        df_demo  : DataFrame cargado desde DATASET_DATOS_DEMOGRAFICOS.csv
        df_rural : DataFrame cargado desde data/referencia/DATASET_MUNICIPIOS_RURALES_COLOMBIA.csv

    Returns:
        (df_limpio, stats)
    """
    stats: dict = {"paso": 1}
    total_ini = len(df_demo)
    logger.info(f"[Paso 1] Inicio — {total_ini:,} filas")

    df = df_demo.copy()

    # ── PASO A: Normalizar TODAS las columnas objeto a string limpio sin tildes ──
    # Esto elimina de raíz el error 'float has no attribute strip'
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            df[col] = _normalizar_col_texto(df[col])
            
    # Columnas numéricas que pueden haber quedado como object por valores mixtos:
    for col in ["ESTRATO", "ID_EST"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("  Texto normalizado (ASCII, sin tildes).")

    # ── PASO B: Tipos numéricos y fechas ───────────────────────────────────────
    df["FECHA_NACIMIENTO"] = pd.to_datetime(df["FECHA_NACIMIENTO"], errors="coerce")
    df["AÑO_INGRESO"]      = pd.to_numeric(df["AÑO_INGRESO"],      errors="coerce")
    df["EDAD_INGRESO"]     = df["AÑO_INGRESO"] - df["FECHA_NACIMIENTO"].dt.year

    # ── PASO C: Situación laboral → 0/1 ───────────────────────────────────────
    no_trabaja = {"desempleado", "estudiante", "incapacitado para trabajar", "", "nan"}
    df["SITUACION_LABORAL"] = (
        df["SITUACION_LABORAL"].str.lower()
        .apply(lambda x: 0 if x in no_trabaja else 1)
    )

    # ── PASO D: Fecha registro info laboral ────────────────────────────────────
    df["FECHA_REG_INF_LABORAL"] = (
        df["FECHA_REG_INF_LABORAL"].str.split(" ").str[0].replace("", np.nan)
    )
    df["FECHA_REG_INF_LABORAL"] = pd.to_datetime(
        df["FECHA_REG_INF_LABORAL"], errors="coerce", dayfirst=True, format="mixed"
    )

    # ── PASO E: Comunidad negra / pueblo indígena → 0/1 ───────────────────────
    df["COMUNIDAD_NEGRA"] = (
        df["COMUNIDAD_NEGRA"].str.lower()
        .apply(lambda x: 0 if x in {"no aplica", "", "nan"} else 1)
    )
    df["PUEBLO_INDIGENA"] = (
        df["PUEBLO_INDIGENA"].str.lower()
        .apply(lambda x: 0 if x in {"no aplica", "no informa", "", "nan"} else 1)
    )
    df.drop(columns=["GRUPO_ETNICO"], inplace=True, errors="ignore")

    # ── PASO F: Discapacidad → categoría numérica ──────────────────────────────
    def _cat_disc(x: str) -> int:
        x = x.lower()
        if x in ("", "no aplica", "nan"):   return 0
        if "fisica"      in x:              return 1
        if "multiple"    in x:              return 2
        if "psicosocial" in x:              return 3
        if "sensorial"   in x or "baja vision" in x: return 4
        return 5

    df["DISCAPACIDAD"] = df["DISCAPACIDAD"].apply(_cat_disc)

    def _cat_disc(x) -> int:
        x = _safe_str(x).lower()
        if x in ("", "no aplica", "nan", "none"):       return 0
        if "fisica" in x:                               return 1
        if "multiple" in x:                             return 2
        if "psicosocial" in x:                          return 3
        if "sensorial" in x or "baja vision" in x:      return 4
        return 5

    df["DISCAPACIDAD"] = df["DISCAPACIDAD"].apply(_cat_disc)

    # ── PASO G: Procedencia Popayán → 0/1 ─────────────────────────────────────
    df["PROCEDENCIA"] = (
        df["MUNICIPIO_RESIDENCIA"].str.lower()
        .apply(lambda x: 1 if x == "popayan" else 0)
    )

    # ── PASO H: Municipio rural ────────────────────────────────────────────────
    if "MUNICIPIOS" in df_rural.columns:
        dr = df_rural.copy()
        dr["MUNICIPIOS"] = _normalizar_col_texto(dr["MUNICIPIOS"]).str.lower()
        dr = dr.dropna(subset=["MUNICIPIOS"]).drop_duplicates(subset=["MUNICIPIOS"])
        rural_map = dict(zip(dr["MUNICIPIOS"], dr["RURAL"]))
        df["MUNICIPIO_PROCEDENCIA_RURAL"] = (
            df["MUNICIPIO_RESIDENCIA"].str.lower().map(rural_map).fillna(0).astype(int)
        )
    else:
        df["MUNICIPIO_PROCEDENCIA_RURAL"] = 0

    # ── PASO I: Tipo institución → 0/1/2 ──────────────────────────────────────
    def _tipo_inst(x: str) -> int:
        x = x.lower()
        if x == "privado":                                                    return 1
        if x in ("oficial nacional", "oficial departamental", "oficial municipal"): return 0
        return 2

    df["TIPO_INSTITUCION"] = df["TIPO_INSTITUCION"].apply(_tipo_inst)

    # ── PASO J: Excluir programas ──────────────────────────────────────────────
    antes = len(df)
    df = df[~df["PROGRAMA"].isin(PROGRAMAS_EXCLUIR)].copy()
    stats["filas_excluidas_programa"] = antes - len(df)

    # ── PASO K: Fecha y periodo de grado ──────────────────────────────────────
    df["FECHA_GRADO"]   = pd.to_datetime(df["FECHA_GRADO"], errors="coerce")
    df["PERIODO_GRADO"] = df["FECHA_GRADO"].apply(calcular_periodo_grado)

    # ── PASO L: Periodo de ingreso (YYYY-S) ────────────────────────────────────
    # AÑO_INGRESO es float (ej 2018.0) → convertir a int string
    anio_s = df["AÑO_INGRESO"].where(
        df["AÑO_INGRESO"].notna(), other=np.nan
    ).dropna().astype(int).astype(str)
    anio_col = df["AÑO_INGRESO"].map(
        lambda v: "" if (v is None or (isinstance(v, float) and np.isnan(v)))
                  else str(int(v))
    )
    # PERIDO_INGRESO puede ser int (1, 2) si la columna original era numérica
    per_col = df["PERIDO_INGRESO"].fillna("").astype(str).str.strip()

    combinado = anio_col + "-" + per_col

    def _validar_periodo(x: str) -> object:
        """Retorna el string si es YYYY-S válido, de lo contrario pd.NA."""
        if not x or x in ("-", "--"):
            return pd.NA
        partes = x.split("-")
        if len(partes) != 2:
            return pd.NA
        try:
            int(partes[0]); int(partes[1])
        except (ValueError, TypeError):
            return pd.NA
        if int(partes[0]) < 2000 or int(partes[1]) not in (1, 2):
            return pd.NA
        return x

    df["PERIDO_INGRESO"] = combinado.apply(_validar_periodo)

    # ── PASO M: Tiempo de retención ────────────────────────────────────────────
    df["TIEMPO_RETENCION_EST"] = df.apply(calcular_tiempo_retencion, axis=1)

    # ── PASO N: Eliminar columnas innecesarias ─────────────────────────────────
    df.drop(
        columns=["MUNICIPIO_RESIDENCIA", "CARGO", "AÑO_INGRESO",
                 "FECHA_GRADO", "TITULO_ESTUDIO"],
        inplace=True, errors="ignore"
    )

    # ── PASO O: Ordenar y detectar duplicados ─────────────────────────────────
    df = df.sort_values(
        by=["ID_EST", "FECHA_REG_INF_LABORAL"], ascending=[True, True]
    ).reset_index(drop=True)

    cols_dup = [c for c in df.columns if c not in ["ELIMINADO", "FECHA_REG_INF_LABORAL"]]
    df_dup   = df[cols_dup].copy()
    # Normalizar a string para comparación
    for col in df_dup.columns:
        if df_dup[col].dtype == object:
            df_dup[col] = df_dup[col].fillna("").astype(str).str.strip()

    mask_dup = df_dup.duplicated(keep="last")
    df["ELIMINADO"] = mask_dup
    stats["duplicados_eliminados"] = int(mask_dup.sum())

    df_limpio = df[~mask_dup].drop(columns=["ELIMINADO"]).reset_index(drop=True)

    stats.update({
        "total_inicial":   total_ini,
        "total_final":     len(df_limpio),
        "filas_eliminadas": total_ini - len(df_limpio),
    })
    logger.info(
        f"[Paso 1] Fin — {len(df_limpio):,} filas | "
        f"dups eliminados: {stats['duplicados_eliminados']}"
    )
    return df_limpio, stats
