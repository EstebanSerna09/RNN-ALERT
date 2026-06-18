"""
modelos/comparacion_modelos.py
================================
Lógica de comparación de modelos separada en dos bloques:

  BLOQUE 1 — Clasificación de deserción
    Incluye: RNN (prob_desercion), SVM Clas, RF Clas, Reg. Logística
    Métricas: ROC-AUC, PR-AUC, Recall, Precision, F1, Accuracy, Umbral

  BLOQUE 2 — Regresión de nota
    Incluye: RNN (pred_nota), SVM Reg, RF Reg, Reg. Lineal
    Métricas: MAE, MSE, RMSE, R²

Permite filtrado por programa académico o vista global.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Nombres legibles por clave de modelo ML ────────────────────────────────────
_NOMBRE_CLAVE = {
    "svm_clas":   "SVM Classifier",
    "rf_clas":    "Random Forest Classifier",
    "rl_clas":    "Regresión Logística",
    "svm_reg":    "SVM Regressor",
    "rf_reg":     "Random Forest Regressor",
    "lineal_reg": "Regresión Lineal",
}

# Columnas de cada bloque
COLS_CLAS = ["Modelo", "Programa", "ROC-AUC", "PR-AUC", "Recall", "Precision", "F1", "Accuracy", "Umbral"]
COLS_REG  = ["Modelo", "Programa", "MAE", "MSE", "RMSE", "R²"]


def _safe(v, decimales=4):
    """Formatea un número o devuelve '—' si no está disponible."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return round(float(v), decimales)




def _normalizar_programa(valor: Any) -> str:
    """Normaliza un programa para comparación interna."""
    return str(valor).strip().upper().replace(" ", "_")


def _programa_legible(valor: Any) -> str:
    """Convierte un programa seguro a nombre legible para la UI."""
    txt = str(valor).strip()
    if not txt or txt.lower() in ("none", "nan"):
        return "Global"
    return txt.replace("_", " ").title()


def _extraer_programa_de_nombre(nombre_modelo: str) -> Optional[str]:
    """
    Extrae el programa desde nombres tipo:
    'RNN Multitarea (Administracion De Empresas)'.
    """
    if "(" in nombre_modelo and ")" in nombre_modelo:
        try:
            return nombre_modelo.split("(", 1)[1].rsplit(")", 1)[0].strip()
        except Exception:
            return None
    return None


def _obtener_programa_modelo(nombre_modelo: str, info: Dict, met: Dict) -> str:
    """
    Obtiene el programa del modelo de forma robusta.

    Prioridad:
    1. info['programa']
    2. info['datos_rnn']['programa']
    3. metricas['programa']
    4. texto entre paréntesis en el nombre del modelo
    5. 'Global'

    Esto evita que las RNN entrenadas por programa aparezcan como 'Global'
    en Comparación de Modelos y Evaluación General cuando fueron registradas
    sin metadato explícito de programa.
    """
    candidatos = [
        info.get("programa"),
        (info.get("datos_rnn") or {}).get("programa"),
        met.get("programa"),
        _extraer_programa_de_nombre(nombre_modelo),
    ]
    for c in candidatos:
        if c is not None and str(c).strip() and str(c).strip().lower() not in ("global", "none", "nan"):
            return str(c).strip()
    return "Global"

# ══════════════════════════════════════════════════════════════════════════════
# EXTRACCIÓN DE FILAS POR TIPO DE MODELO
# ══════════════════════════════════════════════════════════════════════════════

def _fila_clas(nombre_modelo: str, programa: str, met: Dict) -> Dict:
    return {
        "Modelo":    nombre_modelo,
        "Programa":  programa,
        "ROC-AUC":   _safe(met.get("roc_auc")),
        "PR-AUC":    _safe(met.get("pr_auc")),
        "Recall":    _safe(met.get("recall_opt")),
        "Precision": _safe(met.get("precision_opt")),
        "F1":        _safe(met.get("f1_opt")),
        "Accuracy":  _safe(met.get("accuracy_opt")),
        "Umbral":    _safe(met.get("umbral_optimo")),
    }


def _fila_reg(nombre_modelo: str, programa: str, met: Dict) -> Dict:
    return {
        "Modelo":   nombre_modelo,
        "Programa": programa,
        "MAE":      _safe(met.get("mae")),
        "MSE":      _safe(met.get("mse")),
        "RMSE":     _safe(met.get("rmse")),
        "R²":       _safe(met.get("r2")),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CONSTRUCCIÓN DE TABLAS DE COMPARACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def construir_tablas_comparacion(
    modelos: Dict[str, Dict],
    programa_filtro: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Genera dos DataFrames de comparación desde el dict de modelos registrados.

    Args:
        modelos         : st.session_state["modelos_entrenados"]
        programa_filtro : si no es None, filtra solo ese programa (safe_name)
                          (comparación ignora mayúsculas/espacios)

    Returns:
        (df_clasificacion, df_regresion)
    """
    filas_clas: List[Dict] = []
    filas_reg:  List[Dict] = []

    for nombre, info in modelos.items():
        tipo  = info.get("tipo", "")
        met   = info.get("metricas", {})
        prog  = _obtener_programa_modelo(nombre, info, met)

        # Normalizar programa para filtro
        prog_norm = _normalizar_programa(prog)
        if programa_filtro:
            filtro_norm = _normalizar_programa(programa_filtro)
            if prog_norm != filtro_norm:
                continue

        # ── RNN ──────────────────────────────────────────────────────────
        if tipo == "rnn_multi":
            prog_leg = _programa_legible(prog)
            # Bloque clasificación: usa métricas del modelo RNN directamente
            if met.get("roc_auc") is not None:
                filas_clas.append(_fila_clas(f"RNN Multitarea", prog_leg, met))
            # Bloque regresión: usa mae/r2 del modelo RNN
            if met.get("mae") is not None:
                filas_reg.append(_fila_reg(f"RNN Multitarea", prog_leg, met))

        # ── ML por programa ───────────────────────────────────────────────
        elif tipo == "ml_por_programa":
            tarea = info.get("tarea", "")
            clave = info.get("clave", "")
            nombre_leg = _NOMBRE_CLAVE.get(clave, nombre)
            prog_leg   = _programa_legible(prog)

            if tarea == "clasificacion_desercion":
                filas_clas.append(_fila_clas(nombre_leg, prog_leg, met))
            elif tarea == "regresion_nota":
                filas_reg.append(_fila_reg(nombre_leg, prog_leg, met))

        # ── ML global (antiguo — sin programa) ───────────────────────────
        elif tipo == "ml":
            if programa_filtro:
                continue  # Modelos globales no se muestran cuando hay filtro de programa
            # Estos tienen métricas de clasificación únicamente
            if met.get("roc_auc") is not None:
                filas_clas.append(_fila_clas(nombre, "Global", met))

    df_clas = pd.DataFrame(filas_clas, columns=COLS_CLAS) if filas_clas else pd.DataFrame(columns=COLS_CLAS)
    df_reg  = pd.DataFrame(filas_reg,  columns=COLS_REG)  if filas_reg  else pd.DataFrame(columns=COLS_REG)
    return df_clas, df_reg


# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN GLOBAL
# ══════════════════════════════════════════════════════════════════════════════

def resumen_global(modelos: Dict[str, Dict]) -> Dict:
    """
    Genera un resumen con el mejor modelo de clasificación y regresión
    por programa y de forma global.

    Returns:
        dict con claves:
          "mejor_clas_global", "mejor_reg_global",
          "mejores_clas_por_prog", "mejores_reg_por_prog",
          "df_clas", "df_reg"
    """
    df_clas, df_reg = construir_tablas_comparacion(modelos)

    # Mejor clasificación global (por ROC-AUC numérico)
    mejor_clas_global = None
    mejor_reg_global  = None

    if not df_clas.empty:
        df_c = df_clas.copy()
        df_c["_auc"] = pd.to_numeric(df_c["ROC-AUC"], errors="coerce")
        idx = df_c["_auc"].idxmax()
        if pd.notna(idx):
            mejor_clas_global = df_c.loc[idx, ["Modelo", "Programa", "ROC-AUC", "Recall", "F1"]].to_dict()

    if not df_reg.empty:
        df_r = df_reg.copy()
        df_r["_r2"] = pd.to_numeric(df_r["R²"], errors="coerce")
        idx = df_r["_r2"].idxmax()
        if pd.notna(idx):
            mejor_reg_global = df_r.loc[idx, ["Modelo", "Programa", "MAE", "RMSE", "R²"]].to_dict()

    # Mejores por programa
    mejores_clas_prog: Dict[str, Dict] = {}
    mejores_reg_prog:  Dict[str, Dict] = {}

    if not df_clas.empty:
        df_c = df_clas.copy()
        df_c["_auc"] = pd.to_numeric(df_c["ROC-AUC"], errors="coerce")
        for prog, sub in df_c.groupby("Programa"):
            idx = sub["_auc"].idxmax()
            if pd.notna(idx):
                mejores_clas_prog[prog] = sub.loc[idx].to_dict()

    if not df_reg.empty:
        df_r = df_reg.copy()
        df_r["_r2"] = pd.to_numeric(df_r["R²"], errors="coerce")
        for prog, sub in df_r.groupby("Programa"):
            idx = sub["_r2"].idxmax()
            if pd.notna(idx):
                mejores_reg_prog[prog] = sub.loc[idx].to_dict()

    return {
        "mejor_clas_global":    mejor_clas_global,
        "mejor_reg_global":     mejor_reg_global,
        "mejores_clas_por_prog": mejores_clas_prog,
        "mejores_reg_por_prog":  mejores_reg_prog,
        "df_clas":              df_clas,
        "df_reg":               df_reg,
    }


# ══════════════════════════════════════════════════════════════════════════════
# LISTA DE PROGRAMAS DISPONIBLES EN LOS MODELOS
# ══════════════════════════════════════════════════════════════════════════════

def programas_en_modelos(modelos: Dict[str, Dict]) -> List[str]:
    """Devuelve lista ordenada de programas únicos con al menos un modelo."""
    progs = set()
    for nombre, info in modelos.items():
        met = info.get("metricas", {})
        p = _obtener_programa_modelo(nombre, info, met)
        if p and _normalizar_programa(p) != "GLOBAL":
            progs.add(_programa_legible(p))
    return sorted(progs)
