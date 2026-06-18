"""
modelos/modelo_rnn.py
=====================
Modelo RNN Multimodal Multitarea — RNN-ALERT · UNIMAYOR · 2026

Arquitectura exacta extraída de RNN_Model.ipynb:

Rama 1 — Historia Académica (secuencial):
  Input(max_semestres, num_features_acad)
  → Masking(mask_value=0)
  → LSTM(64, return_sequences=False)
  → Dropout(0.2)

Rama 2 — Variables Sociodemográficas (estáticas):
  Input(num_features_socio,)
  → Dense(32, relu)

Fusión:
  Concatenate()
  → Dense(64, relu)
  → Dropout(0.2)

Salida 1 — Clasificación:
  Dense(1, sigmoid, name='prob_desercion')

Salida 2 — Regresión:
  Dense(1, linear, name='pred_nota')

Optimizador: Adam
Loss:
  prob_desercion → binary_crossentropy
  pred_nota      → mse
Métricas:
  prob_desercion → accuracy, AUC
  pred_nota      → mae

─────────────────────────────────────────────────────────────────────────────
OPTIMIZACIONES CONTROLADAS (2026-06-14) — tesis RNN-ALERT
─────────────────────────────────────────────────────────────────────────────
  stratify_split    : bool  — split estratificado por clase desertor
  loss_weight_des   : float — peso de prob_desercion en compile()
  usar_sample_weight: bool  — pesos balanceados por clase en fit()

CORRECCIÓN 2026-06-14 — Bug "inconsistent number of samples":
  Causa: df_socio y df_label podían tener filas duplicadas por ID_EST
  (múltiples registros por programa o semestre). Al extraer
    y_desertor = df_label["DESERTOR"].values   → N+k filas
    X_estatico = df_socio_f.values             → N+m filas
  pero indices = np.arange(num_estudiantes)    → N filas
  → stratify=y_desertor fallaba con ValueError.

  Solución: construir y_desertor y X_estatico desde ids_comunes usando
  diccionarios indexados por ID_EST, igual que X_secuencial. Así los
  tres arrays tienen exactamente len(ids_comunes) filas en el mismo orden,
  garantizado independientemente de duplicados en los DataFrames.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc, classification_report, confusion_matrix,
    f1_score, mean_absolute_error, precision_recall_curve,
    precision_score, r2_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)

# ── Columnas del notebook ──────────────────────────────────────────────────────
COLUMNAS_EXCLUIR_ACAD  = ["ID_EST", "NUMERO_SEMESTRE", "PROGRAMA"]
COLUMNAS_EXCLUIR_SOCIO = ["ID_EST", "PROGRAMA"]

MAX_SEMESTRES_GLOBAL = 12


# ══════════════════════════════════════════════════════════════════════════════
# PREPARACIÓN DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

def preparar_datos_rnn(
    df_socio:       pd.DataFrame,
    df_acad:        pd.DataFrame,
    df_label:       pd.DataFrame,
    test_size:      float = 0.20,
    seed:           int   = 42,
    stratify_split: bool  = True,
) -> Dict[str, Any]:
    """
    Prepara los tensores de entrada para el modelo RNN multimodal.

    IMPORTANTE — Garantía de alineamiento:
        Todos los arrays (X_secuencial, X_estatico, y_desertor, y_nota_futura)
        se construyen indexando desde ids_comunes usando diccionarios. Esto
        garantiza que shape[0] == len(ids_comunes) para todos, independien-
        temente de filas duplicadas en los DataFrames de entrada.

    Parámetros:
        stratify_split : Si True, aplica stratify en train_test_split usando
                         y_desertor_final — el vector construido desde
                         ids_comunes, que tiene exactamente num_estudiantes
                         elementos. Incluye fallback automático con try/except.
    """
    # ── 1. Alineación de IDs ──────────────────────────────────────────────────
    ids_comunes = sorted(
        set(df_socio["ID_EST"])
        .intersection(set(df_acad["ID_EST"]))
        .intersection(set(df_label["ID_EST"]))
    )
    num_estudiantes = len(ids_comunes)
    logger.info(f"[RNN] Estudiantes alineados: {num_estudiantes}")

    if num_estudiantes == 0:
        raise ValueError("[RNN] No hay estudiantes comunes entre los tres datasets.")

    # ── 2. Diagnóstico de semestres ───────────────────────────────────────────
    df_acad_filt = df_acad[df_acad["ID_EST"].isin(ids_comunes)]
    max_sem_real = int(df_acad_filt["NUMERO_SEMESTRE"].max())
    sem_counts   = df_acad_filt.groupby("ID_EST")["NUMERO_SEMESTRE"].count()
    n_truncados  = int((sem_counts > MAX_SEMESTRES_GLOBAL).sum())
    logger.info(f"[RNN] Máx semestres: {max_sem_real} | Techo: {MAX_SEMESTRES_GLOBAL}")
    if n_truncados > 0:
        logger.warning(f"[RNN] {n_truncados} estudiante(s) con semestres > techo, serán recortados.")

    # ── 3. Construir diccionarios indexados por ID_EST ────────────────────────
    # CORRECCIÓN: no extraer .values directamente de df filtrado (puede tener
    # duplicados). En cambio, construir un dict {id_est: valor/fila} tomando
    # solo la PRIMERA ocurrencia por ID_EST, y luego indexar por ids_comunes.

    # 3a. Etiqueta de deserción: primera ocurrencia por ID_EST
    df_label_dedup = (
        df_label[df_label["ID_EST"].isin(ids_comunes)]
        .drop_duplicates(subset=["ID_EST"], keep="first")
        .set_index("ID_EST")
    )
    y_desertor_final = np.array(
        [int(df_label_dedup.loc[id_est, "DESERTOR"]) for id_est in ids_comunes],
        dtype="int32"
    )

    # 3b. Variables sociodemográficas: primera ocurrencia por ID_EST
    cols_socio = [c for c in df_socio.columns if c not in COLUMNAS_EXCLUIR_SOCIO]
    df_socio_dedup = (
        df_socio[df_socio["ID_EST"].isin(ids_comunes)]
        .drop_duplicates(subset=["ID_EST"], keep="first")
        .set_index("ID_EST")
    )
    # Convertir a numérico y rellenar nulos
    for col in cols_socio:
        if col in df_socio_dedup.columns:
            df_socio_dedup[col] = pd.to_numeric(df_socio_dedup[col], errors="coerce").fillna(0)

    # Extraer features_socio en el orden exacto de ids_comunes
    cols_socio_presentes = [c for c in cols_socio if c in df_socio_dedup.columns]
    num_features_socio   = len(cols_socio_presentes)
    X_estatico = np.zeros((num_estudiantes, num_features_socio), dtype="float32")
    for i, id_est in enumerate(ids_comunes):
        if id_est in df_socio_dedup.index:
            X_estatico[i] = df_socio_dedup.loc[id_est, cols_socio_presentes].values.astype("float32")

    # ── 4. Tensor 3D académico → X_secuencial ────────────────────────────────
    max_semestres      = MAX_SEMESTRES_GLOBAL
    features_acad_cols = [c for c in df_acad.columns if c not in COLUMNAS_EXCLUIR_ACAD]
    num_features_acad  = len(features_acad_cols)

    X_secuencial  = np.zeros((num_estudiantes, max_semestres, num_features_acad), dtype="float32")
    y_nota_futura = np.zeros(num_estudiantes, dtype="float32")
    id_to_idx     = {id_est: idx for idx, id_est in enumerate(ids_comunes)}

    df_acad_ord = df_acad_filt.sort_values(["ID_EST", "NUMERO_SEMESTRE"])

    for id_est, group in df_acad_ord.groupby("ID_EST"):
        if id_est not in id_to_idx:
            continue
        idx          = id_to_idx[id_est]
        group_sorted = group.sort_values("NUMERO_SEMESTRE")
        seq_data     = group_sorted[features_acad_cols].values.astype("float32")
        n_sem_real   = len(seq_data)

        if n_sem_real > max_semestres:
            logger.warning(
                f"[RNN] Estudiante {id_est}: {n_sem_real} semestres > techo "
                f"{max_semestres}. Recortando a los últimos {max_semestres}."
            )
            seq_data   = seq_data[-max_semestres:]
            n_sem_real = max_semestres

        n_sem = min(n_sem_real, max_semestres)
        X_secuencial[idx, :n_sem, :] = seq_data[:n_sem]

        if "PROMEDIO_ACADEMICO" in group_sorted.columns:
            y_nota_futura[idx] = float(group_sorted["PROMEDIO_ACADEMICO"].iloc[-1])

    # ── 5. Validación explícita de alineamiento ───────────────────────────────
    # Garantía: todos los arrays deben tener exactamente num_estudiantes filas.
    assert X_secuencial.shape[0] == num_estudiantes, \
        f"[RNN] X_secuencial tiene {X_secuencial.shape[0]} filas, esperado {num_estudiantes}"
    assert X_estatico.shape[0] == num_estudiantes, \
        f"[RNN] X_estatico tiene {X_estatico.shape[0]} filas, esperado {num_estudiantes}"
    assert len(y_desertor_final) == num_estudiantes, \
        f"[RNN] y_desertor tiene {len(y_desertor_final)} elementos, esperado {num_estudiantes}"
    assert len(y_nota_futura) == num_estudiantes, \
        f"[RNN] y_nota_futura tiene {len(y_nota_futura)} elementos, esperado {num_estudiantes}"

    logger.info(
        f"[RNN] Alineamiento OK — X_sec: {X_secuencial.shape} | "
        f"X_est: {X_estatico.shape} | y_des: {y_desertor_final.shape} | "
        f"y_nota: {y_nota_futura.shape} | "
        f"Desertores: {y_desertor_final.sum()} / {num_estudiantes} "
        f"({100*y_desertor_final.mean():.1f}%)"
    )

    # ── 6. Split train/test estratificado ─────────────────────────────────────
    # stratify usa y_desertor_final — garantizadamente del mismo tamaño
    # que indices. Fallback a split aleatorio si alguna clase tiene <2 muestras
    # o si train_test_split lanza ValueError por clases insuficientes.
    indices = np.arange(num_estudiantes)

    uso_stratify = False
    if stratify_split:
        try:
            idx_train, idx_test = train_test_split(
                indices,
                test_size=test_size,
                random_state=seed,
                stratify=y_desertor_final,
            )
            uso_stratify = True
            logger.info("[RNN] Split ESTRATIFICADO por y_desertor aplicado correctamente.")
        except ValueError as e:
            idx_train, idx_test = train_test_split(
                indices, test_size=test_size, random_state=seed
            )
            logger.warning(
                f"[RNN] stratify falló ({e}). "
                f"Usando split aleatorio como fallback para este programa."
            )
    else:
        idx_train, idx_test = train_test_split(
            indices, test_size=test_size, random_state=seed
        )
        logger.info("[RNN] Split SIN estratificación (configuración original).")

    logger.info(
        f"[RNN] Split → Train: {len(idx_train)} | Test: {len(idx_test)} | "
        f"stratify={'sí' if uso_stratify else 'no (fallback)'} | "
        f"Desertores train: {y_desertor_final[idx_train].sum()} | "
        f"Desertores test: {y_desertor_final[idx_test].sum()}"
    )

    return {
        "X_sec_train":         X_secuencial[idx_train],
        "X_sec_test":          X_secuencial[idx_test],
        "X_est_train":         X_estatico[idx_train],
        "X_est_test":          X_estatico[idx_test],
        "y_des_train":         y_desertor_final[idx_train],
        "y_des_test":          y_desertor_final[idx_test],
        "y_not_train":         y_nota_futura[idx_train],
        "y_not_test":          y_nota_futura[idx_test],
        "max_semestres":       max_semestres,
        "num_features_acad":   num_features_acad,
        "num_features_socio":  num_features_socio,
        "features_acad_cols":  features_acad_cols,
        "features_socio_cols": cols_socio_presentes,
        "estudiantes_comunes": ids_comunes,
        "df_socio":            df_socio,
        "df_acad":             df_acad,
        "_uso_stratify":       uso_stratify,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CONSTRUCCIÓN DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════

def construir_modelo_rnn(
    max_semestres:      int,
    num_features_acad:  int,
    num_features_socio: int,
    lstm_units:         int   = 64,
    dense_socio_units:  int   = 32,
    dense_fusion_units: int   = 64,
    dropout_rate:       float = 0.2,
    learning_rate:      float = 0.001,
    loss_weight_des:    float = 1.0,
) -> Any:
    """
    Construye el modelo multimodal multitarea.

    loss_weight_des: peso de prob_desercion en compile().
        1.0 = comportamiento original.
        >1.0 = prioriza la salida de deserción sobre la de nota.
        Justificación: Cipolla et al. (2018) — Uncertainty Weighting.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import Concatenate, Dense, Dropout, Input, LSTM, Masking
        from tensorflow.keras.models import Model
    except ImportError:
        raise ImportError("TensorFlow no está instalado.")

    input_acad  = Input(shape=(max_semestres, num_features_acad), name="historia_academica")
    masked_acad = Masking(mask_value=0.0)(input_acad)
    lstm_out    = LSTM(lstm_units, return_sequences=False, activation="tanh")(masked_acad)
    lstm_out    = Dropout(dropout_rate)(lstm_out)

    input_socio = Input(shape=(num_features_socio,), name="sociodemografico")
    dense_socio = Dense(dense_socio_units, activation="relu")(input_socio)

    merged        = Concatenate()([lstm_out, dense_socio])
    x             = Dense(dense_fusion_units, activation="relu")(merged)
    x             = Dropout(dropout_rate)(x)

    out_desercion = Dense(1, activation="sigmoid", name="prob_desercion")(x)
    out_nota      = Dense(1, activation="linear",  name="pred_nota")(x)

    modelo = Model(
        inputs=[input_acad, input_socio],
        outputs=[out_desercion, out_nota],
        name="RNN_ALERT_Multimodal",
    )

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "prob_desercion": "binary_crossentropy",
            "pred_nota":      "mse",
        },
        loss_weights={
            "prob_desercion": loss_weight_des,
            "pred_nota":      1.0,
        },
        metrics={
            "prob_desercion": ["accuracy", tf.keras.metrics.AUC(name="auc")],
            "pred_nota":      ["mae"],
        },
    )

    logger.info(
        f"[RNN] Modelo construido. Parámetros: {modelo.count_params():,} | "
        f"loss_weight_desercion={loss_weight_des}"
    )
    return modelo


# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def entrenar_modelo_rnn(
    datos:              Dict[str, Any],
    epochs:             int   = 150,
    batch_size:         int   = 32,
    patience:           int   = 15,
    ruta_salida:        Path  = Path("outputs/modelos"),
    seed:               int   = 42,
    loss_weight_des:    float = 1.0,
    usar_sample_weight: bool  = False,
) -> Tuple[Any, Dict]:
    """
    Entrena el modelo RNN multimodal multitarea.

    usar_sample_weight: Si True, compensa el desbalance de clases
        con pesos por muestra normalizados (mean=1) para preservar
        la magnitud total del gradiente.
        Justificación: He & Garcia (2009) — Learning from Imbalanced Data.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    except ImportError:
        raise ImportError("TensorFlow no está instalado.")

    tf.random.set_seed(seed)
    ruta_salida.mkdir(parents=True, exist_ok=True)
    ruta_ckpt = str(ruta_salida / "mejor_modelo_rnn.keras")

    modelo = construir_modelo_rnn(
        max_semestres=datos["max_semestres"],
        num_features_acad=datos["num_features_acad"],
        num_features_socio=datos["num_features_socio"],
        loss_weight_des=loss_weight_des,
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=ruta_ckpt,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    fit_kwargs: Dict[str, Any] = {}
    if usar_sample_weight:
        y_train = datos["y_des_train"]
        clases  = np.unique(y_train)
        pesos   = compute_class_weight("balanced", classes=clases, y=y_train)
        cw      = dict(zip(clases.tolist(), pesos.tolist()))
        sw      = np.array([cw[int(y)] for y in y_train], dtype="float32")
        sw      = sw / sw.mean()   # normalizar: mean=1 preserva magnitud del gradiente
        fit_kwargs["sample_weight"] = sw
        logger.info(
            f"[RNN] sample_weight activado. class_weights={cw} | "
            f"sw.min={sw.min():.3f} sw.max={sw.max():.3f}"
        )
    else:
        logger.info("[RNN] sample_weight desactivado (pesos uniformes).")

    historia = modelo.fit(
        x=[datos["X_sec_train"], datos["X_est_train"]],
        y={
            "prob_desercion": datos["y_des_train"],
            "pred_nota":      datos["y_not_train"],
        },
        validation_data=(
            [datos["X_sec_test"], datos["X_est_test"]],
            {
                "prob_desercion": datos["y_des_test"],
                "pred_nota":      datos["y_not_test"],
            },
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
        **fit_kwargs,
    )

    logger.info(f"[RNN] Entrenamiento finalizado. Modelo guardado en: {ruta_ckpt}")
    return modelo, historia.history


# ══════════════════════════════════════════════════════════════════════════════
# EVALUACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def evaluar_modelo_rnn(
    modelo:      Any,
    datos:       Dict[str, Any],
    umbral:      float = 0.5,
    k_precision: int   = 100,
) -> Dict[str, Any]:
    """Calcula métricas completas de clasificación y regresión."""
    preds       = modelo.predict([datos["X_sec_test"], datos["X_est_test"]], verbose=0)
    y_prob_des  = preds[0].flatten()
    y_pred_nota = preds[1].flatten()
    y_true_des  = datos["y_des_test"]
    y_true_nota = datos["y_not_test"]

    y_pred_cls = (y_prob_des >= umbral).astype(int)

    umbrales = np.arange(0.05, 0.95, 0.01)
    f1s      = [f1_score(y_true_des, (y_prob_des >= u).astype(int), zero_division=0) for u in umbrales]
    u_opt    = float(umbrales[int(np.argmax(f1s))])
    y_opt    = (y_prob_des >= u_opt).astype(int)

    fpr, tpr, _      = roc_curve(y_true_des, y_prob_des)
    prec_c, rec_c, _ = precision_recall_curve(y_true_des, y_prob_des)

    k_real = min(k_precision, len(y_true_des))
    top_k  = np.argsort(y_prob_des)[::-1][:k_real]
    p_at_k = float(y_true_des[top_k].sum() / k_real)

    mae  = mean_absolute_error(y_true_nota, y_pred_nota)
    r2   = r2_score(y_true_nota, y_pred_nota)
    rmse = float(np.sqrt(np.mean((y_true_nota - y_pred_nota) ** 2)))

    metricas = {
        "roc_auc":       float(roc_auc_score(y_true_des, y_prob_des)),
        "pr_auc":        float(auc(rec_c, prec_c)),
        "recall_opt":    float(recall_score(y_true_des, y_opt,      zero_division=0)),
        "precision_opt": float(precision_score(y_true_des, y_opt,   zero_division=0)),
        "f1_opt":        float(f1_score(y_true_des, y_opt,          zero_division=0)),
        "recall_05":     float(recall_score(y_true_des, y_pred_cls,  zero_division=0)),
        "precision_05":  float(precision_score(y_true_des, y_pred_cls, zero_division=0)),
        "f1_05":         float(f1_score(y_true_des, y_pred_cls,     zero_division=0)),
        "umbral_optimo": u_opt,
        f"precision@{k_precision}": p_at_k,
        "cm":            confusion_matrix(y_true_des, y_opt).tolist(),
        "reporte":       classification_report(
                             y_true_des, y_opt,
                             target_names=["No desertor", "Desertor"],
                             digits=4,
                         ),
        "fpr":           fpr.tolist(),
        "tpr":           tpr.tolist(),
        "prec_curve":    prec_c.tolist(),
        "rec_curve":     rec_c.tolist(),
        "y_proba":       y_prob_des.tolist(),
        "y_true":        y_true_des.tolist(),
        "mae":           float(mae),
        "rmse":          rmse,
        "r2":            float(r2),
        "y_pred_nota":   y_pred_nota.tolist(),
        "y_true_nota":   y_true_nota.tolist(),
    }

    logger.info(
        f"[RNN] Evaluación — AUC: {metricas['roc_auc']:.4f} | "
        f"Recall: {metricas['recall_opt']:.4f} | F1: {metricas['f1_opt']:.4f} | "
        f"MAE: {mae:.4f} | R²: {r2:.4f}"
    )
    return metricas


def obtener_metricas(modelo: Any, datos: Dict[str, Any]) -> Dict[str, Any]:
    """Alias directo para evaluar_modelo_rnn."""
    return evaluar_modelo_rnn(modelo, datos)


# ══════════════════════════════════════════════════════════════════════════════
# GUARDAR / CARGAR
# ══════════════════════════════════════════════════════════════════════════════

def guardar_modelo(
    modelo:       Any,
    datos_config: Dict[str, Any],
    ruta_salida:  Path = Path("outputs/modelos"),
) -> Path:
    ruta_salida.mkdir(parents=True, exist_ok=True)
    ruta_keras = ruta_salida / "modelo_rnn_final.keras"
    ruta_cfg   = ruta_salida / "rnn_config.pkl"

    modelo.save(str(ruta_keras))
    config = {
        "max_semestres":      datos_config["max_semestres"],
        "num_features_acad":  datos_config["num_features_acad"],
        "num_features_socio": datos_config["num_features_socio"],
        "features_acad_cols": datos_config["features_acad_cols"],
        "features_socio_cols": datos_config.get("features_socio_cols", []),
    }
    with open(ruta_cfg, "wb") as f:
        pickle.dump(config, f)

    logger.info(f"[RNN] Modelo guardado: {ruta_keras}")
    return ruta_keras


def cargar_modelo(ruta_salida: Path = Path("outputs/modelos")) -> Tuple[Any, Dict]:
    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        raise ImportError("TensorFlow no está instalado.")

    ruta_keras = ruta_salida / "modelo_rnn_final.keras"
    ruta_cfg   = ruta_salida / "rnn_config.pkl"

    if not ruta_keras.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {ruta_keras}")

    modelo = load_model(str(ruta_keras))
    config = {}
    if ruta_cfg.exists():
        with open(ruta_cfg, "rb") as f:
            config = pickle.load(f)

    logger.info(f"[RNN] Modelo cargado desde: {ruta_keras}")
    return modelo, config


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCIA INDIVIDUAL
# ══════════════════════════════════════════════════════════════════════════════

def predecir_estudiante(
    modelo:              Any,
    datos_socio_dict:    Dict[str, Any],
    historia_acad_df:    pd.DataFrame,
    max_semestres:       int,
    features_acad_cols:  List[str],
    columnas_excl_socio: List[str] = None,
    features_socio_cols: List[str] = None,
) -> Tuple[float, float, int]:
    """
    Predice probabilidad de deserción y nota proyectada para UN estudiante.
    Interfaz idéntica a la versión original — sin cambios de firma.
    """
    if columnas_excl_socio is None:
        columnas_excl_socio = COLUMNAS_EXCLUIR_SOCIO

    if features_socio_cols:
        # Construir el vector sociodemográfico en el mismo orden usado al entrenar.
        # Esto es clave para predicción manual y masiva.
        fila_socio = {col: datos_socio_dict.get(col, 0) for col in features_socio_cols}
        df_s = pd.DataFrame([fila_socio], columns=features_socio_cols)
    else:
        # Compatibilidad con artefactos antiguos: conserva el comportamiento previo.
        df_s = pd.DataFrame([datos_socio_dict])
        df_s = df_s.drop(columns=columnas_excl_socio, errors="ignore")

    for col in df_s.columns:
        df_s[col] = pd.to_numeric(df_s[col], errors="coerce")
    df_s  = df_s.fillna(0)
    X_est = df_s.values.astype("float32")

    X_seq        = np.zeros((1, max_semestres, len(features_acad_cols)), dtype="float32")
    historia_ord = historia_acad_df.sort_values("NUMERO_SEMESTRE")

    cols_presentes = [c for c in features_acad_cols if c in historia_ord.columns]
    if len(cols_presentes) < len(features_acad_cols):
        logger.warning(
            f"[RNN-Inferencia] Faltan {len(features_acad_cols)-len(cols_presentes)} "
            f"columnas. Se rellenarán con 0."
        )

    seq_data   = historia_ord[cols_presentes].values.astype("float32")
    n_sem_real = len(seq_data)

    if n_sem_real > max_semestres:
        seq_data   = seq_data[-max_semestres:]
        n_sem_real = max_semestres

    n_sem = min(n_sem_real, max_semestres)

    for fi, col in enumerate(features_acad_cols):
        if col in cols_presentes:
            ci = cols_presentes.index(col)
            X_seq[0, :n_sem, fi] = seq_data[:n_sem, ci]

    proximo_semestre = n_sem + 1

    res             = modelo.predict([X_seq, X_est], verbose=0)
    prob_desercion  = float(res[0][0][0]) * 100.0
    nota_proyectada = float(res[1][0][0])

    logger.info(
        f"[RNN-Inferencia] P(deserción)={prob_desercion:.1f}% | "
        f"Nota proyectada={nota_proyectada:.2f} | Semestre={proximo_semestre}"
    )
    return prob_desercion, nota_proyectada, proximo_semestre
