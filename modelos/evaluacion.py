"""
evaluacion.py
=============
Evaluación, métricas y visualizaciones para RNN-ALERT.

Métricas implementadas (en orden de prioridad para EWS):
  1. Recall         — detectar el máximo de desertores reales
  2. F1-Score       — balance precisión/recall
  3. ROC-AUC        — discriminación general del modelo
  4. PR-AUC         — más informativo que ROC bajo desbalance fuerte
  5. Precision      — calidad de las alertas generadas
  6. Precision@K    — precisión en los K estudiantes con mayor riesgo

Justificación de Recall como métrica primaria:
  En un EWS (Early Warning System), un falso negativo (desertor no
  detectado) tiene un costo social mucho mayor que un falso positivo
  (estudiante activo marcado como riesgo). Institucionalmente, es
  preferible intervenir con estudiantes que no necesitaban ayuda que
  omitir a quienes sí la necesitaban. Esto se formaliza como:

    Recall = TP / (TP + FN)

  donde minimizar FN es el objetivo central del sistema de alertas.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
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
from tensorflow import keras

logger = logging.getLogger(__name__)


# ── Búsqueda de umbral óptimo ─────────────────────────────────────────────────

def encontrar_umbral_optimo(
    y_true:    np.ndarray,
    y_proba:   np.ndarray,
    metrica:   str = "f1",
) -> Tuple[float, float]:
    """
    Encuentra el umbral de clasificación que maximiza la métrica objetivo.

    Justificación:
    El umbral por defecto de 0.5 es subóptimo para datos desbalanceados.
    Con 25% de desertores, la distribución de probabilidades tiende a
    valores más bajos, por lo que el umbral óptimo suele ser menor (~0.30-0.40).

    Parámetros
    ----------
    metrica : "f1" | "recall" | "precision_recall"
        • "f1": maximiza el F1-score (balance recall/precision)
        • "recall": maximiza recall (prioriza detección, acepta más FP)
        • "youden": maximiza el índice J de Youden = Sensitivity + Specificity - 1
    """
    umbrales = np.arange(0.05, 0.95, 0.01)
    mejores: Dict[str, float] = {"umbral": 0.5, "valor": 0.0}

    for u in umbrales:
        y_pred = (y_proba >= u).astype(int)
        if metrica == "f1":
            val = f1_score(y_true, y_pred, zero_division=0)
        elif metrica == "recall":
            val = recall_score(y_true, y_pred, zero_division=0)
        elif metrica == "youden":
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            val  = sens + spec - 1
        else:
            val = f1_score(y_true, y_pred, zero_division=0)

        if val > mejores["valor"]:
            mejores["valor"]   = val
            mejores["umbral"]  = u

    logger.info(
        f"Umbral óptimo ({metrica}): {mejores['umbral']:.2f} → {mejores['valor']:.4f}"
    )
    return mejores["umbral"], mejores["valor"]


# ── Precision@K ──────────────────────────────────────────────────────────────

def precision_at_k(
    y_true:  np.ndarray,
    y_proba: np.ndarray,
    k:       int = 100,
) -> float:
    """
    Precision@K: fracción de desertores reales entre los K estudiantes
    con mayor probabilidad predicha.

    Justificación:
    En la práctica institucional, los recursos de intervención son limitados.
    Un departamento de bienestar puede atender K estudiantes por semestre.
    P@K mide la eficiencia del sistema: ¿qué fracción de esas K alertas
    corresponde a desertores reales?

    P@K = |{desertores en top-K}| / K
    """
    k = min(k, len(y_true))
    top_k_idx = np.argsort(y_proba)[::-1][:k]
    return float(y_true[top_k_idx].sum() / k)


# ── Evaluación completa ────────────────────────────────────────────────────────

def evaluar_modelo(
    modelo:      keras.Model,
    X_test:      np.ndarray,
    y_test:      np.ndarray,
    nombre:      str  = "modelo",
    umbral:      float = 0.5,
    k_precision: int   = 100,
    ruta_salida: str   = "outputs",
) -> Dict[str, float]:
    """
    Evaluación completa del modelo sobre el conjunto de prueba.

    Retorna un diccionario con todas las métricas calculadas y genera
    las visualizaciones (matriz de confusión, curva ROC, curva PR,
    histograma de probabilidades).
    """
    Path(ruta_salida).mkdir(parents=True, exist_ok=True)

    # ── Predicciones ─────────────────────────────────────────────────────
    y_proba = modelo.predict(X_test, verbose=0).flatten()
    y_pred  = (y_proba >= umbral).astype(int)

    # ── Umbral óptimo por F1 ─────────────────────────────────────────────
    umbral_opt, f1_opt = encontrar_umbral_optimo(y_test, y_proba, "f1")
    y_pred_opt = (y_proba >= umbral_opt).astype(int)

    # ── Métricas base ────────────────────────────────────────────────────
    metricas = {
        "roc_auc":          roc_auc_score(y_test, y_proba),
        "recall_05":        recall_score(y_test, y_pred,     zero_division=0),
        "precision_05":     precision_score(y_test, y_pred,  zero_division=0),
        "f1_05":            f1_score(y_test, y_pred,         zero_division=0),
        "recall_opt":       recall_score(y_test, y_pred_opt, zero_division=0),
        "precision_opt":    precision_score(y_test, y_pred_opt, zero_division=0),
        "f1_opt":           f1_opt,
        "umbral_optimo":    umbral_opt,
        f"precision@{k_precision}": precision_at_k(y_test, y_proba, k_precision),
    }

    # PR-AUC
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
    metricas["pr_auc"] = auc(rec_curve, prec_curve)

    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUACIÓN — {nombre.upper()}")
    logger.info(f"{'='*60}")
    for k, v in metricas.items():
        logger.info(f"  {k:25s}: {v:.4f}")

    print(f"\n{'='*60}")
    print(f"EVALUACIÓN — {nombre.upper()}")
    print(f"{'='*60}")
    for k, v in metricas.items():
        print(f"  {k:25s}: {v:.4f}")
    print(f"\nReporte de clasificación (umbral={umbral_opt:.2f}):")
    print(classification_report(
        y_test, y_pred_opt,
        target_names=["No desertor", "Desertor"],
        digits=4
    ))

    # ── Visualizaciones ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"Evaluación del Modelo — {nombre}", fontsize=14, fontweight="bold")

    # 1. Matriz de confusión (umbral óptimo)
    cm  = confusion_matrix(y_test, y_pred_opt)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0],
        xticklabels=["No desertor", "Desertor"],
        yticklabels=["No desertor", "Desertor"]
    )
    axes[0, 0].set_title(f"Matriz de Confusión (umbral={umbral_opt:.2f})")
    axes[0, 0].set_ylabel("Etiqueta Real")
    axes[0, 0].set_xlabel("Etiqueta Predicha")

    # Anotaciones de TP/TN/FP/FN
    tn, fp, fn, tp = cm.ravel()
    axes[0, 0].text(
        0.5, -0.18,
        f"TP={tp} | TN={tn} | FP={fp} | FN={fn}  "
        f"Recall={tp/(tp+fn):.3f}  Precision={tp/(tp+fp):.3f}" if (tp+fp) > 0 else "",
        ha="center", transform=axes[0, 0].transAxes, fontsize=9, color="darkblue"
    )

    # 2. Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[0, 1].plot(fpr, tpr, color="steelblue", lw=2,
                    label=f"ROC AUC = {metricas['roc_auc']:.4f}")
    axes[0, 1].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Aleatorio")
    axes[0, 1].fill_between(fpr, tpr, alpha=0.10, color="steelblue")
    axes[0, 1].set_title("Curva ROC")
    axes[0, 1].set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
    axes[0, 1].set_ylabel("Tasa de Verdaderos Positivos (Recall)")
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(alpha=0.3)

    # 3. Curva Precision-Recall
    axes[1, 0].plot(rec_curve, prec_curve, color="darkorange", lw=2,
                    label=f"PR AUC = {metricas['pr_auc']:.4f}")
    baseline = y_test.sum() / len(y_test)
    axes[1, 0].axhline(y=baseline, color="gray", linestyle="--", lw=1,
                       label=f"Línea base = {baseline:.3f}")
    axes[1, 0].fill_between(rec_curve, prec_curve, alpha=0.10, color="darkorange")
    axes[1, 0].set_title("Curva Precision-Recall")
    axes[1, 0].set_xlabel("Recall")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # 4. Distribución de probabilidades predichas
    mask_pos = y_test == 1
    mask_neg = y_test == 0
    axes[1, 1].hist(y_proba[mask_neg], bins=40, alpha=0.6, color="royalblue",
                    label="No desertor", density=True)
    axes[1, 1].hist(y_proba[mask_pos], bins=40, alpha=0.6, color="tomato",
                    label="Desertor", density=True)
    axes[1, 1].axvline(x=umbral_opt, color="black", linestyle="--", lw=1.5,
                       label=f"Umbral óptimo = {umbral_opt:.2f}")
    axes[1, 1].set_title("Distribución de Probabilidades Predichas")
    axes[1, 1].set_xlabel("P(Deserción)")
    axes[1, 1].set_ylabel("Densidad")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    ruta_fig = Path(ruta_salida) / f"evaluacion_{nombre.lower().replace(' ', '_')}.png"
    fig.savefig(ruta_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figura guardada: {ruta_fig}")

    return metricas


# ── Curvas de aprendizaje ──────────────────────────────────────────────────────

def graficar_historia(
    historia:    dict,
    nombre:      str = "modelo",
    ruta_salida: str = "outputs",
) -> None:
    """Grafica las curvas de pérdida y AUC durante el entrenamiento."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Curvas de Entrenamiento — {nombre}", fontsize=13, fontweight="bold")

    epochs = range(1, len(historia["loss"]) + 1)

    # Pérdida
    axes[0].plot(epochs, historia["loss"],     label="Train Loss",  color="steelblue", lw=2)
    axes[0].plot(epochs, historia["val_loss"], label="Val Loss",    color="tomato",    lw=2, linestyle="--")
    axes[0].set_title("Función de Pérdida")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # AUC
    axes[1].plot(epochs, historia["auc"],     label="Train AUC",  color="steelblue", lw=2)
    axes[1].plot(epochs, historia["val_auc"], label="Val AUC",    color="tomato",    lw=2, linestyle="--")
    axes[1].set_title("ROC-AUC por Época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("AUC")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    ruta_fig = Path(ruta_salida) / f"historia_{nombre.lower().replace(' ', '_')}.png"
    fig.savefig(ruta_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Historia guardada: {ruta_fig}")


# ── Comparación de modelos ────────────────────────────────────────────────────

def comparar_modelos(
    metricas_lstm: Dict[str, float],
    metricas_gru:  Dict[str, float],
    ruta_salida:   str = "outputs",
) -> None:
    """Genera tabla comparativa y gráfico de barras entre LSTM y GRU."""
    metricas_clave = ["roc_auc", "pr_auc", "recall_opt", "precision_opt",
                      "f1_opt", "precision@100"]

    nombres_legibles = {
        "roc_auc":        "ROC AUC",
        "pr_auc":         "PR AUC",
        "recall_opt":     "Recall (umbral ópt.)",
        "precision_opt":  "Precision (umbral ópt.)",
        "f1_opt":         "F1-Score (umbral ópt.)",
        "precision@100":  "Precision@100",
    }

    print(f"\n{'='*65}")
    print(f"{'COMPARACIÓN LSTM vs GRU+ATTENTION':^65}")
    print(f"{'='*65}")
    print(f"{'Métrica':<30} {'LSTM':>12} {'GRU+Attn':>12} {'Δ (GRU-LSTM)':>12}")
    print(f"{'-'*65}")
    for m in metricas_clave:
        v_lstm = metricas_lstm.get(m, 0.0)
        v_gru  = metricas_gru.get(m, 0.0)
        delta  = v_gru - v_lstm
        signo  = "+" if delta >= 0 else ""
        nombre = nombres_legibles.get(m, m)
        print(f"{nombre:<30} {v_lstm:>12.4f} {v_gru:>12.4f} {signo}{delta:>11.4f}")
    print(f"{'='*65}")

    # Gráfico de barras comparativo
    fig, ax = plt.subplots(figsize=(11, 6))
    x  = np.arange(len(metricas_clave))
    w  = 0.35
    vals_lstm = [metricas_lstm.get(m, 0) for m in metricas_clave]
    vals_gru  = [metricas_gru.get(m, 0)  for m in metricas_clave]

    bars_lstm = ax.bar(x - w/2, vals_lstm, w, label="LSTM Bidireccional",
                       color="steelblue", alpha=0.85)
    bars_gru  = ax.bar(x + w/2, vals_gru,  w, label="GRU + Attention",
                       color="darkorange", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([nombres_legibles.get(m, m) for m in metricas_clave],
                       rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Valor de la métrica")
    ax.set_title("Comparación de Modelos — RNN-ALERT (UNIMAYOR)", fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in bars_lstm:
        ax.annotate(f"{bar.get_height():.3f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
    for bar in bars_gru:
        ax.annotate(f"{bar.get_height():.3f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)

    plt.tight_layout()
    ruta_fig = Path(ruta_salida) / "comparacion_lstm_gru.png"
    fig.savefig(ruta_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Comparación guardada: {ruta_fig}")
