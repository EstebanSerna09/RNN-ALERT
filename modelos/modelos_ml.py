"""
modelos/modelos_ml.py
=====================
Modelos de Machine Learning clásicos para RNN-ALERT.
Incluye: Random Forest, SVM y Regresión Logística.

Todos operan sobre representaciones ESTÁTICAS del estudiante:
se promedia cada feature académica sobre todos sus semestres
y se concatenan con las variables sociodemográficas.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)

# ── Features esperados ─────────────────────────────────────────────────────────
FEATURES_ACAD = [
    "PROMEDIO_ACADEMICO", "MATERIAS_CURSADAS", "MATERIAS_APROBADAS",
    "MATERIAS_REPROBADAS", "NOTA_MAXIMA", "NOTA_MINIMA",
    "TASA_APROBACION", "ACUMULACION_MATERIAS_CURSADAS",
]
FEATURES_DEMO = [
    "ESTRATO", "SITUACION_LABORAL", "COMUNIDAD_NEGRA", "PUEBLO_INDIGENA",
    "DISCAPACIDAD", "PROCEDENCIA", "MUNICIPIO_PROCEDENCIA_RURAL",
    "TIPO_INSTITUCION", "EDAD_INGRESO", "TIEMPO_RETENCION_EST",
]
FEATURES_TODOS = FEATURES_ACAD + FEATURES_DEMO


def preparar_features_estaticas(
    df_acad: pd.DataFrame,
    df_demo: pd.DataFrame,
    df_des:  pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construye la matriz de features estática (N, F) para modelos clásicos.
    Agrega features académicas por estudiante (promedio de todos los semestres)
    y une con features demográficas.

    Returns: (X, y)
    """
    # Promedio de features académicas por (ID_EST, PROGRAMA)
    cols_acad = [c for c in FEATURES_ACAD if c in df_acad.columns]
    agg_acad = (
        df_acad.groupby(["ID_EST", "PROGRAMA"])[cols_acad]
        .mean()
        .reset_index()
    )

    # Features demográficas (un registro por estudiante-programa)
    cols_demo = ["ID_EST", "PROGRAMA"] + [c for c in FEATURES_DEMO if c in df_demo.columns]
    demo_uniq = df_demo[cols_demo].drop_duplicates(subset=["ID_EST", "PROGRAMA"])

    # Etiquetas
    etiq = df_des[["ID_EST", "PROGRAMA", "DESERTOR"]].drop_duplicates(subset=["ID_EST", "PROGRAMA"])

    # Merge
    merged = agg_acad.merge(demo_uniq, on=["ID_EST", "PROGRAMA"], how="inner")
    merged = merged.merge(etiq, on=["ID_EST", "PROGRAMA"], how="inner")

    feature_cols = [c for c in FEATURES_TODOS if c in merged.columns]
    X = merged[feature_cols].fillna(0).values.astype(np.float32)
    y = merged["DESERTOR"].values.astype(np.int32)

    logger.info(f"Features estáticas: {X.shape} | Desertores: {y.sum()} ({y.mean()*100:.1f}%)")
    return X, y


def _class_weights(y: np.ndarray) -> Dict[int, float]:
    classes = np.unique(y)
    w = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes.tolist(), w.tolist()))


def calcular_metricas(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    umbral: float = 0.5,
    k: int = 100,
) -> Dict[str, Any]:
    """Calcula el conjunto completo de métricas para un modelo."""
    y_pred = (y_proba >= umbral).astype(int)
    prec_c, rec_c, _ = precision_recall_curve(y_true, y_proba)

    # Umbral óptimo por F1
    umbrales = np.arange(0.05, 0.95, 0.01)
    f1s = [f1_score(y_true, (y_proba >= u).astype(int), zero_division=0) for u in umbrales]
    u_opt = float(umbrales[int(np.argmax(f1s))])
    y_opt = (y_proba >= u_opt).astype(int)

    # Precision@K
    k_real = min(k, len(y_true))
    top_k  = np.argsort(y_proba)[::-1][:k_real]
    p_at_k = float(y_true[top_k].sum() / k_real)

    return {
        "roc_auc":       float(roc_auc_score(y_true, y_proba)),
        "pr_auc":        float(auc(rec_c, prec_c)),
        "recall_05":     float(recall_score(y_true, y_pred, zero_division=0)),
        "precision_05":  float(precision_score(y_true, y_pred, zero_division=0)),
        "f1_05":         float(f1_score(y_true, y_pred, zero_division=0)),
        "recall_opt":    float(recall_score(y_true, y_opt, zero_division=0)),
        "precision_opt": float(precision_score(y_true, y_opt, zero_division=0)),
        "f1_opt":        float(f1_score(y_true, y_opt, zero_division=0)),
        "umbral_optimo": u_opt,
        f"precision@{k}": p_at_k,
        "cm":            confusion_matrix(y_true, y_opt).tolist(),
        "reporte":       classification_report(y_true, y_opt,
                            target_names=["No desertor", "Desertor"], digits=4),
        "fpr":           roc_curve(y_true, y_proba)[0].tolist(),
        "tpr":           roc_curve(y_true, y_proba)[1].tolist(),
        "prec_curve":    prec_c.tolist(),
        "rec_curve":     rec_c.tolist(),
        "y_proba":       y_proba.tolist(),
        "y_true":        y_true.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RANDOM FOREST
# ═══════════════════════════════════════════════════════════════════════════════

def entrenar_random_forest(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    n_estimators: int   = 300,
    max_depth:    Optional[int] = None,
    min_samples_leaf: int = 4,
    seed: int = 42,
) -> Tuple[RandomForestClassifier, StandardScaler, Dict]:
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_va = scaler.transform(X_val)
    cw   = _class_weights(y_train)

    modelo = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=cw,
        random_state=seed,
        n_jobs=-1,
    )
    modelo.fit(X_tr, y_train)
    proba_val = modelo.predict_proba(X_va)[:, 1]
    metricas  = calcular_metricas(y_val, proba_val)
    logger.info(f"RF val_auc={metricas['roc_auc']:.4f} recall={metricas['recall_opt']:.4f}")
    return modelo, scaler, metricas


# ═══════════════════════════════════════════════════════════════════════════════
# SVM
# ═══════════════════════════════════════════════════════════════════════════════

def entrenar_svm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    kernel: str   = "rbf",
    C:      float = 1.0,
    gamma:  str   = "scale",
    seed:   int   = 42,
) -> Tuple[SVC, StandardScaler, Dict]:
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_train)
    X_va   = scaler.transform(X_val)
    cw     = _class_weights(y_train)

    modelo = SVC(
        kernel=kernel, C=C, gamma=gamma,
        probability=True,
        class_weight=cw,
        random_state=seed,
    )
    modelo.fit(X_tr, y_train)
    proba_val = modelo.predict_proba(X_va)[:, 1]
    metricas  = calcular_metricas(y_val, proba_val)
    logger.info(f"SVM val_auc={metricas['roc_auc']:.4f} recall={metricas['recall_opt']:.4f}")
    return modelo, scaler, metricas


# ═══════════════════════════════════════════════════════════════════════════════
# REGRESIÓN LOGÍSTICA
# ═══════════════════════════════════════════════════════════════════════════════

def entrenar_regresion_logistica(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    C:       float = 1.0,
    max_iter: int  = 1000,
    seed:     int  = 42,
) -> Tuple[LogisticRegression, StandardScaler, Dict]:
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_train)
    X_va   = scaler.transform(X_val)

    modelo = LogisticRegression(
        C=C, max_iter=max_iter,
        class_weight="balanced",
        solver="lbfgs", random_state=seed,
    )
    modelo.fit(X_tr, y_train)
    proba_val = modelo.predict_proba(X_va)[:, 1]
    metricas  = calcular_metricas(y_val, proba_val)
    logger.info(f"LR val_auc={metricas['roc_auc']:.4f} recall={metricas['recall_opt']:.4f}")
    return modelo, scaler, metricas


# ═══════════════════════════════════════════════════════════════════════════════
# GUARDAR / CARGAR modelos clásicos
# ═══════════════════════════════════════════════════════════════════════════════

def guardar_modelo_ml(modelo: Any, scaler: StandardScaler, nombre: str, directorio: Path) -> Path:
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / f"{nombre}.pkl"
    with open(ruta, "wb") as f:
        pickle.dump({"modelo": modelo, "scaler": scaler}, f)
    logger.info(f"Modelo guardado: {ruta}")
    return ruta


def cargar_modelo_ml(ruta: Path) -> Tuple[Any, StandardScaler]:
    with open(ruta, "rb") as f:
        d = pickle.load(f)
    return d["modelo"], d["scaler"]
