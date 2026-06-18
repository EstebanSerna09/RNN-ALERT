"""
modelos/modelos_ml_por_programa.py
====================================
Entrenamiento de modelos clásicos de ML separados por programa académico
y por tipo de tarea (clasificación de deserción / regresión de nota).

Estructura de guardado:
  outputs/modelos_ml/por_programa/<SAFE_PROG>/
      svm_socio_clas.pkl
      rf_socio_clas.pkl
      rl_socio_clas.pkl
      svm_acad_reg.pkl
      rf_acad_reg.pkl
      reg_lineal_acad_reg.pkl
      config_socio.pkl
      config_acad.pkl
      metricas.json

Modelos de clasificación (target=DESERTOR, features sociodemográficas):
  - SVM Classifier
  - Random Forest Classifier
  - Regresión Logística

Modelos de regresión (target=PROMEDIO_ACADEMICO, features académicas):
  - SVM Regressor
  - Random Forest Regressor
  - Regresión Lineal
"""
from __future__ import annotations

import json
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, mean_absolute_error, mean_squared_error,
    precision_recall_curve, precision_score, r2_score,
    recall_score, roc_auc_score, roc_curve, auc as sk_auc,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)

# ── Nombres de archivo dentro de cada carpeta de programa ─────────────────────
_ARCHIVOS = {
    "svm_clas":     "svm_socio_clas.pkl",
    "rf_clas":      "rf_socio_clas.pkl",
    "rl_clas":      "rl_socio_clas.pkl",      # Regresión Logística
    "svm_reg":      "svm_acad_reg.pkl",
    "rf_reg":       "rf_acad_reg.pkl",
    "lineal_reg":   "reg_lineal_acad_reg.pkl",
    "config_socio": "config_socio.pkl",
    "config_acad":  "config_acad.pkl",
    "metricas":     "metricas.json",
}

# ── Features esperadas ────────────────────────────────────────────────────────
FEATURES_SOCIO = [
    "ESTRATO", "SITUACION_LABORAL", "COMUNIDAD_NEGRA", "PUEBLO_INDIGENA",
    "DISCAPACIDAD", "PROCEDENCIA", "MUNICIPIO_PROCEDENCIA_RURAL",
    "TIPO_INSTITUCION", "EDAD_INGRESO", "TIEMPO_RETENCION_EST",
]
FEATURES_ACAD = [
    "PROMEDIO_ACADEMICO", "MATERIAS_CURSADAS", "MATERIAS_APROBADAS",
    "MATERIAS_REPROBADAS", "NOTA_MAXIMA", "NOTA_MINIMA",
    "TASA_APROBACION", "ACUMULACION_MATERIAS_CURSADAS",
]
TARGET_CLAS = "DESERTOR"
TARGET_REG  = "PROMEDIO_ACADEMICO"   # promedio del último semestre


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS INTERNOS
# ══════════════════════════════════════════════════════════════════════════════

def _dir_prog(outputs_dir: Path, safe_prog: str) -> Path:
    d = outputs_dir / "modelos_ml" / "por_programa" / safe_prog
    d.mkdir(parents=True, exist_ok=True)
    return d


def _class_weights(y: np.ndarray) -> Dict[int, float]:
    clases = np.unique(y)
    w = compute_class_weight("balanced", classes=clases, y=y)
    return dict(zip(clases.tolist(), w.tolist()))


def _metricas_clasificacion(y_true: np.ndarray, y_proba: np.ndarray) -> Dict:
    """Métricas completas de clasificación con umbral óptimo por F1."""
    umbrales = np.arange(0.05, 0.95, 0.01)
    f1s = [f1_score(y_true, (y_proba >= u).astype(int), zero_division=0) for u in umbrales]
    u_opt = float(umbrales[int(np.argmax(f1s))])
    y_opt = (y_proba >= u_opt).astype(int)
    y_05  = (y_proba >= 0.5).astype(int)

    prec_c, rec_c, _ = precision_recall_curve(y_true, y_proba)
    fpr, tpr, _      = roc_curve(y_true, y_proba)

    return {
        "tarea":         "clasificacion_desercion",
        "roc_auc":       float(roc_auc_score(y_true, y_proba)),
        "pr_auc":        float(sk_auc(rec_c, prec_c)),
        "recall_opt":    float(recall_score(y_true, y_opt, zero_division=0)),
        "precision_opt": float(precision_score(y_true, y_opt, zero_division=0)),
        "f1_opt":        float(f1_score(y_true, y_opt, zero_division=0)),
        "accuracy_opt":  float(accuracy_score(y_true, y_opt)),
        "recall_05":     float(recall_score(y_true, y_05, zero_division=0)),
        "precision_05":  float(precision_score(y_true, y_05, zero_division=0)),
        "f1_05":         float(f1_score(y_true, y_05, zero_division=0)),
        "umbral_optimo": u_opt,
        "cm":            confusion_matrix(y_true, y_opt).tolist(),
        "reporte":       classification_report(
                             y_true, y_opt,
                             target_names=["No desertor", "Desertor"], digits=4),
        "fpr":           fpr.tolist(),
        "tpr":           tpr.tolist(),
        "prec_curve":    prec_c.tolist(),
        "rec_curve":     rec_c.tolist(),
    }


def _metricas_regresion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Métricas completas de regresión."""
    mae  = float(mean_absolute_error(y_true, y_pred))
    mse  = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2   = float(r2_score(y_true, y_pred))
    return {
        "tarea": "regresion_nota",
        "mae":   mae,
        "mse":   mse,
        "rmse":  rmse,
        "r2":    r2,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PREPARACIÓN DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

def preparar_datos_sociodemograficos(
    df_demo: pd.DataFrame,
    df_des:  pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Construye X sociodemográfico e y (DESERTOR) para un programa.

    Returns:
        (X, y, cols_usadas)
    """
    cols = ["ID_EST"] + [c for c in FEATURES_SOCIO if c in df_demo.columns]
    demo_u = df_demo[cols].drop_duplicates("ID_EST")

    etiq = df_des[["ID_EST", TARGET_CLAS]].drop_duplicates("ID_EST")
    merged = demo_u.merge(etiq, on="ID_EST", how="inner")

    feature_cols = [c for c in FEATURES_SOCIO if c in merged.columns]
    X = merged[feature_cols].fillna(0).values.astype(np.float32)
    y = merged[TARGET_CLAS].values.astype(np.int32)

    logger.info(f"[ML-Socio] X={X.shape}, desertores={y.sum()} ({y.mean()*100:.1f}%)")
    return X, y, feature_cols


def preparar_datos_academicos(
    df_acad: pd.DataFrame,
    df_des:  pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Construye X académico (promedio por estudiante) e y (PROMEDIO_ACADEMICO
    del último semestre) para un programa.

    Returns:
        (X, y, cols_usadas)
    """
    # Target: promedio del último semestre disponible de cada estudiante
    ultimo = (
        df_acad.sort_values("NUMERO_SEMESTRE")
               .groupby("ID_EST")
               .last()
               .reset_index()
    )
    feature_cols = [c for c in FEATURES_ACAD if c in df_acad.columns]
    # Features: promedio de todos los semestres
    agg = df_acad.groupby("ID_EST")[feature_cols].mean().reset_index()

    # Unir con target del último semestre
    target_col = TARGET_REG if TARGET_REG in ultimo.columns else feature_cols[0]
    merged = agg.merge(ultimo[["ID_EST", target_col]], on="ID_EST", how="inner",
                       suffixes=("", "_target"))
    y_col = target_col + "_target" if target_col + "_target" in merged.columns else target_col

    X = merged[feature_cols].fillna(0).values.astype(np.float32)
    y = merged[y_col].fillna(0).values.astype(np.float32)

    # Eliminar filas con y inválido
    mask = np.isfinite(y)
    X, y = X[mask], y[mask]

    logger.info(f"[ML-Acad] X={X.shape}, y_mean={y.mean():.3f}")
    return X, y, feature_cols


# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO — CLASIFICACIÓN SOCIODEMOGRÁFICA
# ══════════════════════════════════════════════════════════════════════════════

def _entrenar_bloque_clasificacion(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    scaler: StandardScaler,
    seed: int = 42,
) -> Dict[str, Tuple[Any, Dict]]:
    """
    Entrena SVM Classifier, RF Classifier y Regresión Logística.
    Returns: {nombre_clave: (modelo, metricas)}
    """
    cw = _class_weights(y_tr)
    X_tr_s = scaler.transform(X_tr)
    X_va_s = scaler.transform(X_va)

    resultados = {}

    # SVM Classifier
    try:
        svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                  class_weight=cw, random_state=seed)
        svm.fit(X_tr_s, y_tr)
        proba = svm.predict_proba(X_va_s)[:, 1]
        resultados["svm_clas"] = (svm, _metricas_clasificacion(y_va, proba))
        logger.info(f"  [SVM-Clas] AUC={resultados['svm_clas'][1]['roc_auc']:.4f}")
    except Exception as e:
        logger.error(f"  [SVM-Clas] Error: {e}")
        resultados["svm_clas"] = (None, {"error": str(e)})

    # Random Forest Classifier
    try:
        rf = RandomForestClassifier(n_estimators=200, max_depth=None,
                                    min_samples_leaf=4, class_weight=cw,
                                    random_state=seed, n_jobs=-1)
        rf.fit(X_tr_s, y_tr)
        proba = rf.predict_proba(X_va_s)[:, 1]
        resultados["rf_clas"] = (rf, _metricas_clasificacion(y_va, proba))
        logger.info(f"  [RF-Clas] AUC={resultados['rf_clas'][1]['roc_auc']:.4f}")
    except Exception as e:
        logger.error(f"  [RF-Clas] Error: {e}")
        resultados["rf_clas"] = (None, {"error": str(e)})

    # Regresión Logística
    try:
        rl = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                                solver="lbfgs", random_state=seed)
        rl.fit(X_tr_s, y_tr)
        proba = rl.predict_proba(X_va_s)[:, 1]
        resultados["rl_clas"] = (rl, _metricas_clasificacion(y_va, proba))
        logger.info(f"  [RL-Clas] AUC={resultados['rl_clas'][1]['roc_auc']:.4f}")
    except Exception as e:
        logger.error(f"  [RL-Clas] Error: {e}")
        resultados["rl_clas"] = (None, {"error": str(e)})

    return resultados


# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO — REGRESIÓN ACADÉMICA
# ══════════════════════════════════════════════════════════════════════════════

def _entrenar_bloque_regresion(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    scaler: StandardScaler,
    seed: int = 42,
) -> Dict[str, Tuple[Any, Dict]]:
    """
    Entrena SVM Regressor, RF Regressor y Regresión Lineal.
    Returns: {nombre_clave: (modelo, metricas)}
    """
    X_tr_s = scaler.transform(X_tr)
    X_va_s = scaler.transform(X_va)
    resultados = {}

    # SVM Regressor
    try:
        svm = SVR(kernel="rbf", C=1.0, gamma="scale", epsilon=0.1)
        svm.fit(X_tr_s, y_tr)
        y_pred = svm.predict(X_va_s)
        resultados["svm_reg"] = (svm, _metricas_regresion(y_va, y_pred))
        logger.info(f"  [SVM-Reg] MAE={resultados['svm_reg'][1]['mae']:.4f} R2={resultados['svm_reg'][1]['r2']:.4f}")
    except Exception as e:
        logger.error(f"  [SVM-Reg] Error: {e}")
        resultados["svm_reg"] = (None, {"error": str(e)})

    # Random Forest Regressor
    try:
        rf = RandomForestRegressor(n_estimators=200, max_depth=None,
                                   min_samples_leaf=4, random_state=seed, n_jobs=-1)
        rf.fit(X_tr_s, y_tr)
        y_pred = rf.predict(X_va_s)
        resultados["rf_reg"] = (rf, _metricas_regresion(y_va, y_pred))
        logger.info(f"  [RF-Reg] MAE={resultados['rf_reg'][1]['mae']:.4f} R2={resultados['rf_reg'][1]['r2']:.4f}")
    except Exception as e:
        logger.error(f"  [RF-Reg] Error: {e}")
        resultados["rf_reg"] = (None, {"error": str(e)})

    # Regresión Lineal
    try:
        lin = LinearRegression()
        lin.fit(X_tr_s, y_tr)
        y_pred = lin.predict(X_va_s)
        resultados["lineal_reg"] = (lin, _metricas_regresion(y_va, y_pred))
        logger.info(f"  [Lin-Reg] MAE={resultados['lineal_reg'][1]['mae']:.4f} R2={resultados['lineal_reg'][1]['r2']:.4f}")
    except Exception as e:
        logger.error(f"  [Lin-Reg] Error: {e}")
        resultados["lineal_reg"] = (None, {"error": str(e)})

    return resultados


# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO COMPLETO POR PROGRAMA
# ══════════════════════════════════════════════════════════════════════════════

def entrenar_programa_ml(
    safe_prog:   str,
    outputs_dir: Path,
    seed:        int = 42,
    test_size:   float = 0.15,
    val_size:    float = 0.15,
) -> Dict[str, Any]:
    """
    Entrena los 6 modelos clásicos para un programa académico.

    Args:
        safe_prog   : nombre seguro del programa (ej. "INGENIERIA_INFORMATICA")
        outputs_dir : directorio raíz de outputs
        seed        : semilla de reproducibilidad
        test_size   : proporción de test
        val_size    : proporción de validación (del entrenamiento)

    Returns:
        dict con claves "ok", "programa", "modelos", "metricas_global", "error"
    """
    t0 = time.time()
    logger.info(f"[ML-Prog] ══ Entrenando: {safe_prog}")

    # ── Rutas de datos ────────────────────────────────────────────────────
    dir_car = outputs_dir / "por_carrera"
    ruta_demo = dir_car / "demograficos" / f"DF_{safe_prog}_DEMOGRAFICOS.csv"
    ruta_acad = dir_car / "academicos"   / f"DF_{safe_prog}_ACADEMICOS.csv"
    ruta_des  = dir_car / "desertores"   / f"DF_{safe_prog}_DESERTORES.csv"

    for ruta in [ruta_demo, ruta_acad, ruta_des]:
        if not ruta.exists():
            msg = f"Archivo no encontrado: {ruta}"
            logger.error(f"[ML-Prog] {msg}")
            return {"ok": False, "programa": safe_prog, "error": msg,
                    "modelos": {}, "metricas_global": {}}
    try:
        df_demo = pd.read_csv(ruta_demo)
        df_acad = pd.read_csv(ruta_acad)
        df_des  = pd.read_csv(ruta_des)
    except Exception as e:
        return {"ok": False, "programa": safe_prog, "error": str(e),
                "modelos": {}, "metricas_global": {}}

    # ── Datos sociodemográficos ───────────────────────────────────────────
    try:
        X_s, y_s, cols_s = preparar_datos_sociodemograficos(df_demo, df_des)
        if len(np.unique(y_s)) < 2:
            raise ValueError("Solo hay una clase en y_socio — no se puede clasificar.")
        X_s_tr, X_s_te, y_s_tr, y_s_te = train_test_split(
            X_s, y_s, test_size=test_size, stratify=y_s, random_state=seed)
        X_s_tr, X_s_va, y_s_tr, y_s_va = train_test_split(
            X_s_tr, y_s_tr, test_size=val_size/(1-test_size),
            stratify=y_s_tr, random_state=seed)
        scaler_socio = StandardScaler().fit(X_s_tr)
    except Exception as e:
        logger.error(f"[ML-Prog] Error preparando datos socio: {e}")
        X_s_va = y_s_va = X_s_tr = y_s_tr = None
        cols_s = []; scaler_socio = StandardScaler()

    # ── Datos académicos ─────────────────────────────────────────────────
    try:
        X_a, y_a, cols_a = preparar_datos_academicos(df_acad, df_des)
        X_a_tr, X_a_te, y_a_tr, y_a_te = train_test_split(
            X_a, y_a, test_size=test_size, random_state=seed)
        X_a_tr, X_a_va, y_a_tr, y_a_va = train_test_split(
            X_a_tr, y_a_tr, test_size=val_size/(1-test_size), random_state=seed)
        scaler_acad = StandardScaler().fit(X_a_tr)
    except Exception as e:
        logger.error(f"[ML-Prog] Error preparando datos acad: {e}")
        X_a_va = y_a_va = X_a_tr = y_a_tr = None
        cols_a = []; scaler_acad = StandardScaler()

    # ── Entrenamiento ────────────────────────────────────────────────────
    res_clas = {}
    if X_s_tr is not None:
        res_clas = _entrenar_bloque_clasificacion(
            X_s_tr, y_s_tr, X_s_va, y_s_va, scaler_socio, seed)

    res_reg = {}
    if X_a_tr is not None:
        res_reg = _entrenar_bloque_regresion(
            X_a_tr, y_a_tr, X_a_va, y_a_va, scaler_acad, seed)

    # ── Persistencia ─────────────────────────────────────────────────────
    dir_p = _dir_prog(outputs_dir, safe_prog)

    config_socio = {
        "features":      cols_s,
        "target":        TARGET_CLAS,
        "programa":      safe_prog,
        "scaler":        scaler_socio,
        "timestamp":     datetime.now().isoformat(),
    }
    config_acad = {
        "features":      cols_a,
        "target":        TARGET_REG,
        "programa":      safe_prog,
        "scaler":        scaler_acad,
        "timestamp":     datetime.now().isoformat(),
    }

    with open(dir_p / _ARCHIVOS["config_socio"], "wb") as f:
        pickle.dump(config_socio, f)
    with open(dir_p / _ARCHIVOS["config_acad"], "wb") as f:
        pickle.dump(config_acad, f)

    # Guardar modelos individuales
    modelos_guardados = {}
    for clave, (modelo, met) in {**res_clas, **res_reg}.items():
        if modelo is not None:
            ruta_pkl = dir_p / _ARCHIVOS[clave]
            with open(ruta_pkl, "wb") as f:
                pickle.dump({"modelo": modelo}, f)
            modelos_guardados[clave] = {"modelo": modelo, "metricas": met}

    # Guardar métricas globales en JSON
    metricas_json: Dict = {"programa": safe_prog, "timestamp": datetime.now().isoformat()}
    _guardar_metricas_seguras = lambda m: {
        k: v for k, v in m.items()
        if k not in ("fpr", "tpr", "prec_curve", "rec_curve", "cm", "reporte")
    }
    for clave, (_, met) in {**res_clas, **res_reg}.items():
        metricas_json[clave] = _guardar_metricas_seguras(met) if "error" not in met else met
    # Guardar también cm y reporte por separado (son útiles en UI)
    for clave, (_, met) in res_clas.items():
        if "cm" in met:
            metricas_json[f"{clave}_cm"] = met["cm"]

    with open(dir_p / _ARCHIVOS["metricas"], "w", encoding="utf-8") as f:
        json.dump(metricas_json, f, ensure_ascii=False, indent=2)

    duracion = round(time.time() - t0, 1)
    logger.info(f"[ML-Prog] ✔ {safe_prog} completado en {duracion}s — "
                f"{len(modelos_guardados)}/6 modelos guardados")

    return {
        "ok":             len(modelos_guardados) > 0,
        "programa":       safe_prog,
        "modelos":        modelos_guardados,
        "config_socio":   config_socio,
        "config_acad":    config_acad,
        "metricas_global": metricas_json,
        "duracion_seg":   duracion,
        "error":          "",
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO MASIVO
# ══════════════════════════════════════════════════════════════════════════════

def entrenar_todos_los_programas_ml(
    outputs_dir:         Path,
    programas_safe:      List[str],
    seed:                int = 42,
    callback_progreso    = None,
) -> Dict[str, Any]:
    """
    Entrena los 6 modelos clásicos para todos los programas dados.

    Args:
        outputs_dir     : directorio raíz de outputs
        programas_safe  : lista de nombres seguros de programas
        seed            : semilla
        callback_progreso: función opcional (idx, total, safe_prog)

    Returns:
        dict con "resultados", "resumen", "df_metricas"
    """
    total = len(programas_safe)
    resultados = []
    ok_list, err_list = [], []

    for idx, safe_prog in enumerate(programas_safe, 1):
        if callback_progreso:
            try:
                callback_progreso(idx, total, safe_prog)
            except Exception:
                pass
        res = entrenar_programa_ml(safe_prog, outputs_dir, seed=seed)
        resultados.append(res)
        if res["ok"]:
            ok_list.append(safe_prog)
        else:
            err_list.append(safe_prog)

    if callback_progreso:
        try:
            callback_progreso(total, total, "COMPLETADO")
        except Exception:
            pass

    # Construir DataFrame de métricas
    filas = []
    for res in resultados:
        fila_base = {"programa": res["programa"], "estado": "✅ OK" if res["ok"] else "❌ Error"}
        m = res.get("metricas_global", {})
        for clave in ["svm_clas", "rf_clas", "rl_clas"]:
            met = m.get(clave, {})
            filas.append({**fila_base,
                "tipo":    "clasificacion_desercion",
                "modelo":  clave,
                "roc_auc": met.get("roc_auc", None),
                "recall":  met.get("recall_opt", None),
                "f1":      met.get("f1_opt", None),
                "precision": met.get("precision_opt", None),
                "mae": None, "rmse": None, "r2": None,
            })
        for clave in ["svm_reg", "rf_reg", "lineal_reg"]:
            met = m.get(clave, {})
            filas.append({**fila_base,
                "tipo":    "regresion_nota",
                "modelo":  clave,
                "mae":     met.get("mae", None),
                "rmse":    met.get("rmse", None),
                "r2":      met.get("r2", None),
                "roc_auc": None, "recall": None, "f1": None, "precision": None,
            })

    df_metricas = pd.DataFrame(filas)

    resumen = {
        "total":     total,
        "ok":        len(ok_list),
        "errores":   len(err_list),
        "timestamp": datetime.now().isoformat(),
        "programas_ok":    ok_list,
        "programas_error": err_list,
    }
    return {"resultados": resultados, "resumen": resumen, "df_metricas": df_metricas}


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DESDE DISCO
# ══════════════════════════════════════════════════════════════════════════════

_NOMBRE_LEGIBLE = {
    "svm_clas":   "SVM Classifier (Sociodemográfico)",
    "rf_clas":    "Random Forest Classifier (Sociodemográfico)",
    "rl_clas":    "Regresión Logística (Sociodemográfico)",
    "svm_reg":    "SVM Regressor (Académico)",
    "rf_reg":     "Random Forest Regressor (Académico)",
    "lineal_reg": "Regresión Lineal (Académico)",
}
_TIPO_TAREA = {
    "svm_clas":   "clasificacion_desercion",
    "rf_clas":    "clasificacion_desercion",
    "rl_clas":    "clasificacion_desercion",
    "svm_reg":    "regresion_nota",
    "rf_reg":     "regresion_nota",
    "lineal_reg": "regresion_nota",
}


def cargar_modelos_ml_desde_disco(outputs_dir: Path) -> Dict[str, Dict]:
    """
    Escanea outputs/modelos_ml/por_programa/ y carga todos los modelos
    clásicos que tengan artefactos completos.

    Returns:
        dict {nombre_modelo_completo: info_dict}
        donde info_dict tiene: modelo, tipo, tarea, metricas, config, programa
    """
    dir_base = outputs_dir / "modelos_ml" / "por_programa"
    if not dir_base.exists():
        return {}

    resultado = {}
    for dir_prog in sorted(dir_base.iterdir()):
        if not dir_prog.is_dir():
            continue
        safe_prog = dir_prog.name

        # Cargar configs
        ruta_cs = dir_prog / _ARCHIVOS["config_socio"]
        ruta_ca = dir_prog / _ARCHIVOS["config_acad"]
        config_socio = None
        config_acad  = None
        if ruta_cs.exists():
            try:
                with open(ruta_cs, "rb") as f:
                    config_socio = pickle.load(f)
            except Exception as e:
                logger.warning(f"[ML-Carga] config_socio de {safe_prog}: {e}")
        if ruta_ca.exists():
            try:
                with open(ruta_ca, "rb") as f:
                    config_acad = pickle.load(f)
            except Exception as e:
                logger.warning(f"[ML-Carga] config_acad de {safe_prog}: {e}")

        # Cargar métricas JSON
        ruta_met = dir_prog / _ARCHIVOS["metricas"]
        metricas_json = {}
        if ruta_met.exists():
            try:
                with open(ruta_met, encoding="utf-8") as f:
                    metricas_json = json.load(f)
            except Exception as e:
                logger.warning(f"[ML-Carga] metricas de {safe_prog}: {e}")

        # Cargar cada modelo pkl
        for clave, archivo in _ARCHIVOS.items():
            if clave in ("config_socio", "config_acad", "metricas"):
                continue
            ruta_pkl = dir_prog / archivo
            if not ruta_pkl.exists():
                continue
            try:
                with open(ruta_pkl, "rb") as f:
                    datos = pickle.load(f)
                modelo = datos.get("modelo")
                if modelo is None:
                    continue

                nom_leg = _NOMBRE_LEGIBLE.get(clave, clave)
                nombre_completo = f"{nom_leg} — {safe_prog}"
                tarea = _TIPO_TAREA.get(clave, "desconocido")
                cfg = config_socio if "clas" in clave or "rl" in clave else config_acad

                resultado[nombre_completo] = {
                    "modelo":        modelo,
                    "scaler":        cfg.get("scaler") if cfg else None,
                    "tipo":          "ml_por_programa",
                    "tarea":         tarea,
                    "clave":         clave,
                    "programa":      safe_prog,
                    "features":      (cfg.get("features") if cfg else []) or [],
                    "metricas":      metricas_json.get(clave, {}),
                    "config":        cfg or {},
                }
                logger.info(f"[ML-Carga] ✔ {nombre_completo}")
            except Exception as e:
                logger.warning(f"[ML-Carga] Error cargando {clave} de {safe_prog}: {e}")

    logger.info(f"[ML-Carga] Total modelos ML clásicos cargados: {len(resultado)}")
    return resultado


def auto_cargar_modelos_ml_en_session(outputs_dir: Path) -> int:
    """
    Carga automáticamente modelos ML clásicos desde disco y los registra
    en st.session_state["modelos_entrenados"].

    Returns: número de modelos nuevos registrados.
    """
    import streamlit as st

    if st.session_state.get("_modelos_ml_cargados_disco", False):
        return 0

    modelos_disco = cargar_modelos_ml_desde_disco(outputs_dir)
    modelos_actual = st.session_state.get("modelos_entrenados", {})
    registrados = 0

    for nombre, info in modelos_disco.items():
        if nombre not in modelos_actual:
            modelos_actual[nombre] = info
            registrados += 1

    st.session_state["modelos_entrenados"] = modelos_actual
    st.session_state["_modelos_ml_cargados_disco"] = True

    if registrados > 0:
        logger.info(f"[ML-AutoCarga] {registrados} modelo(s) ML clásicos restaurados desde disco.")
    return registrados


# ══════════════════════════════════════════════════════════════════════════════
# PREDICCIÓN INDIVIDUAL
# ══════════════════════════════════════════════════════════════════════════════

def predecir_individual_ml(
    info_modelo: Dict,
    datos_socio: Dict,
    datos_acad_df: pd.DataFrame,
) -> Dict:
    """
    Realiza predicción individual usando un modelo clásico por programa.

    Args:
        info_modelo   : dict con modelo, scaler, features, tarea
        datos_socio   : dict de variables sociodemográficas del estudiante
        datos_acad_df : DataFrame con historial académico del estudiante

    Returns:
        dict con "prediccion", "probabilidad"/"nota", "tarea"
    """
    modelo  = info_modelo["modelo"]
    scaler  = info_modelo.get("scaler")
    features = info_modelo.get("features", [])
    tarea    = info_modelo.get("tarea", "")

    if tarea == "clasificacion_desercion":
        # Vector sociodemográfico
        X = np.array([[float(datos_socio.get(f, 0)) for f in features]], dtype=np.float32)
        if scaler:
            X = scaler.transform(X)
        if hasattr(modelo, "predict_proba"):
            prob = float(modelo.predict_proba(X)[0, 1])
            pred = int(prob >= 0.5)
        else:
            pred = int(modelo.predict(X)[0])
            prob = float(pred)
        return {"prediccion": pred, "probabilidad": prob, "tarea": tarea}

    elif tarea == "regresion_nota":
        # Promedio de features académicas
        feat_pres = [f for f in features if f in datos_acad_df.columns]
        X = datos_acad_df[feat_pres].mean().reindex(features, fill_value=0).values.reshape(1, -1).astype(np.float32)
        if scaler:
            X = scaler.transform(X)
        nota = float(modelo.predict(X)[0])
        return {"prediccion": nota, "nota": nota, "tarea": tarea}

    return {"error": f"Tarea desconocida: {tarea}"}
