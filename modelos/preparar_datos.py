"""
preparar_datos.py
=================
Ingeniería de datos para RNN/LSTM/GRU — RNN-ALERT
Institución Universitaria Colegio Mayor del Cauca (UNIMAYOR)

Responsabilidades:
  1. Construir secuencias temporales semestre a semestre por estudiante.
  2. Aplicar padding + masking para longitudes variables.
  3. Normalizar y escalar features.
  4. Calcular class_weight para el desbalance 75/25 (~1:3.7).
  5. Generar splits estratificados reproducibles.

Datos de entrada esperados (salida del pipeline de limpieza):
  - DF_ACADEMICO_3_FINAL.csv        → features por semestre
  - DF_DEMOGRAFICOS_2_RELLENADO.csv → features estáticas por estudiante
  - DF_DESERTORES.csv               → etiqueta DESERTOR por estudiante
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)

# ── Columnas de features ───────────────────────────────────────────────────────

# Variables que CAMBIAN semestre a semestre (forman la secuencia temporal)
FEATURES_TEMPORALES: List[str] = [
    "PROMEDIO_ACADEMICO",
    "MATERIAS_CURSADAS",
    "MATERIAS_APROBADAS",
    "MATERIAS_REPROBADAS",
    "NOTA_MAXIMA",
    "NOTA_MINIMA",
    "TASA_APROBACION",
    "ACUMULACION_MATERIAS_CURSADAS",
]

# Variables que NO cambian semestre a semestre (se repiten en cada timestep)
# Justificación: las RNN pueden usar contexto socioeconómico en cada paso
FEATURES_ESTATICAS: List[str] = [
    "ESTRATO",
    "SITUACION_LABORAL",
    "COMUNIDAD_NEGRA",
    "PUEBLO_INDIGENA",
    "DISCAPACIDAD",
    "PROCEDENCIA",
    "MUNICIPIO_PROCEDENCIA_RURAL",
    "TIPO_INSTITUCION",
    "EDAD_INGRESO",
    "TIEMPO_RETENCION_EST",
]

MAX_SEMESTRES = 12   # techo global fijo — ver justificación en modelo_rnn.py
PADDING_VALUE = 0.0  # valor de relleno para padded timesteps


# ── Clase principal ────────────────────────────────────────────────────────────

class PreparadorSecuencias:
    """
    Convierte DataFrames tabulares en tensores (N, T, F) para RNN.

    Parámetros
    ----------
    max_semestres : int
        Longitud máxima de la secuencia (timesteps). Se aplica zero-padding
        al final para estudiantes con menos semestres registrados.
        Justificación: los programas de UNIMAYOR tienen hasta 10 semestres,
        por lo que T=10 es el techo natural. Sequences más largas se truncan.
    """

    def __init__(self, max_semestres: int = MAX_SEMESTRES):
        self.max_semestres   = max_semestres
        self.scaler          = StandardScaler()
        self.scaler_ajustado = False
        self.n_features      = len(FEATURES_TEMPORALES) + len(FEATURES_ESTATICAS)

    # ── carga ────────────────────────────────────────────────────────────────

    @staticmethod
    def cargar_datasets(
        ruta_acad:  str | Path,
        ruta_demo:  str | Path,
        ruta_des:   str | Path,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Carga y retorna los tres DataFrames del pipeline."""
        df_ac  = pd.read_csv(ruta_acad,  low_memory=False)
        df_dem = pd.read_csv(ruta_demo,  low_memory=False)
        df_des = pd.read_csv(ruta_des,   low_memory=False)
        logger.info(
            f"Académico: {len(df_ac):,} filas | "
            f"Demográfico: {len(df_dem):,} filas | "
            f"Desertores: {len(df_des):,} filas"
        )
        return df_ac, df_dem, df_des

    # ── construcción de secuencias ────────────────────────────────────────────

    def construir_secuencias(
        self,
        df_academico:    pd.DataFrame,
        df_demografico:  pd.DataFrame,
        df_desertores:   pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """
        Construye el tensor de entrada (N, T, F) y el vector de etiquetas (N,).

        Estrategia de construcción:
        ─────────────────────────────────────────────────────────────────────────
        Para cada estudiante s con P semestres registrados (P ≤ max_semestres):

          Secuencia: [sem_1 | sem_2 | ... | sem_P | PAD | ... | PAD]
                      ───── T=10 timesteps ─────

        Cada timestep tiene F = 18 features = 8 temporales + 10 estáticas.

        Las features estáticas se repiten en cada timestep. Aunque podría
        parecer redundante, esta estrategia permite a la red asociar el
        contexto socioeconómico con cada punto de la trayectoria académica
        (Hochreiter & Schmidhuber, 1997).

        El padding al FINAL (post-padding) es la práctica estándar cuando
        se usa Masking y las secuencias son de longitud variable. Con el
        padding al inicio (pre-padding) el estado de la RNN en el último
        timestep real puede degradarse por los pasos enmascarados.

        Returns
        ───────
        X         : (N, T, F) float32 — tensor de secuencias
        y         : (N,) int32        — etiquetas DESERTOR
        mask_lens : (N,) int32        — longitud real de cada secuencia
        ids       : List[int]         — ID_EST en el mismo orden
        """
        # 1. Features estáticas por estudiante (un registro por ID_EST+PROGRAMA)
        demo_cols = ["ID_EST", "PROGRAMA"] + [
            c for c in FEATURES_ESTATICAS if c in df_demografico.columns
        ]
        df_est = (
            df_demografico[demo_cols]
            .drop_duplicates(subset=["ID_EST", "PROGRAMA"])
            .set_index(["ID_EST", "PROGRAMA"])
        )

        # 2. Etiquetas
        etiquetas = (
            df_desertores[["ID_EST", "PROGRAMA", "DESERTOR"]]
            .drop_duplicates(subset=["ID_EST", "PROGRAMA"])
            .set_index(["ID_EST", "PROGRAMA"])["DESERTOR"]
        )

        # 3. Estudiantes con etiqueta disponible
        pares_validos = etiquetas.index.tolist()

        X_list:    List[np.ndarray] = []
        y_list:    List[int]        = []
        lens_list: List[int]        = []
        ids_list:  List[int]        = []

        for (id_est, programa) in pares_validos:
            # Secuencia académica del estudiante
            mask_est = (
                (df_academico["ID_EST"]   == id_est) &
                (df_academico["PROGRAMA"] == programa)
            )
            df_s = (
                df_academico[mask_est]
                .sort_values("NUMERO_SEMESTRE")
            )
            n_sem_real = len(df_s)
            if n_sem_real == 0:
                continue

            # ── Recorte si el estudiante supera el techo global ───────────────
            # Se conservan los ÚLTIMOS semestres (los más recientes son más
            # informativos para detectar el riesgo de deserción inminente).
            if n_sem_real > self.max_semestres:
                logger.warning(
                    f"[PrepData] Estudiante {id_est}/{programa}: "
                    f"{n_sem_real} semestres > techo {self.max_semestres}. "
                    f"Recortando a los últimos {self.max_semestres}."
                )
                df_s = df_s.tail(self.max_semestres)
                n_sem_real = self.max_semestres

            # Doble seguro contra broadcasting
            n_sem = min(n_sem_real, self.max_semestres)

            # Features temporales
            feats_temp = np.zeros((n_sem, len(FEATURES_TEMPORALES)), dtype=np.float32)
            for j, col in enumerate(FEATURES_TEMPORALES):
                if col in df_s.columns:
                    feats_temp[:, j] = df_s[col].fillna(0).values[:n_sem]

            # Features estáticas (repetidas por cada timestep)
            feats_est = np.zeros((n_sem, len(FEATURES_ESTATICAS)), dtype=np.float32)
            if (id_est, programa) in df_est.index:
                row_est = df_est.loc[(id_est, programa)]
                for j, col in enumerate(FEATURES_ESTATICAS):
                    if col in row_est.index:
                        val = row_est[col]
                        feats_est[:, j] = 0.0 if pd.isna(val) else float(val)

            # Concatenar temporales + estáticas
            seq = np.concatenate([feats_temp, feats_est], axis=1)  # (n_sem, F)

            # Padding hasta max_semestres (post-padding con PADDING_VALUE)
            pad_len = self.max_semestres - n_sem
            if pad_len > 0:
                pad = np.full((pad_len, seq.shape[1]), PADDING_VALUE, dtype=np.float32)
                seq = np.vstack([seq, pad])

            X_list.append(seq)
            y_list.append(int(etiquetas.loc[(id_est, programa)]))
            lens_list.append(n_sem)
            ids_list.append(id_est)

        X = np.array(X_list, dtype=np.float32)   # (N, T, F)
        y = np.array(y_list,  dtype=np.int32)     # (N,)
        lens = np.array(lens_list, dtype=np.int32)

        logger.info(
            f"Secuencias construidas: {X.shape} | "
            f"Desertores: {y.sum():,} ({y.mean()*100:.1f}%)"
        )
        return X, y, lens, ids_list

    # ── normalización ─────────────────────────────────────────────────────────

    def normalizar(
        self,
        X_train: np.ndarray,
        X_val:   np.ndarray,
        X_test:  np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Aplica StandardScaler feature-wise ignorando los timesteps de padding.

        El scaler se ajusta SOLO con X_train (sin data leakage).
        Los timesteps de padding (fila de ceros) se normalizan también,
        pero la capa Masking los ignora durante la propagación.

        Justificación de StandardScaler vs MinMaxScaler:
        StandardScaler es preferible cuando existen outliers (notas atípicas,
        materias acumuladas altas), lo que es frecuente en datos académicos
        reales. MinMaxScaler amplifica el efecto de outliers.
        """
        N_tr, T, F = X_train.shape

        # Reshape a 2D para el scaler: (N*T, F)
        X_tr_2d = X_train.reshape(-1, F)
        X_va_2d = X_val.reshape(-1, F)
        X_te_2d = X_test.reshape(-1, F)

        # Fit SOLO en train
        self.scaler.fit(X_tr_2d)
        self.scaler_ajustado = True

        X_tr_n = self.scaler.transform(X_tr_2d).reshape(N_tr, T, F)
        X_va_n = self.scaler.transform(X_va_2d).reshape(X_val.shape[0], T, F)
        X_te_n = self.scaler.transform(X_te_2d).reshape(X_test.shape[0], T, F)

        logger.info("Normalización StandardScaler aplicada (fit solo en train).")
        return X_tr_n.astype(np.float32), X_va_n.astype(np.float32), X_te_n.astype(np.float32)

    # ── split ─────────────────────────────────────────────────────────────────

    @staticmethod
    def split(
        X: np.ndarray,
        y: np.ndarray,
        test_size:  float = 0.15,
        val_size:   float = 0.15,
        seed:       int   = 42,
    ) -> Tuple[np.ndarray, ...]:
        """
        Split estratificado 70/15/15.

        Estratificación es crítica con desbalance 75/25: asegura que la
        proporción de desertores se preserve en cada split.
        """
        X_tr, X_test, y_tr, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed
        )
        # val_size relativo al tamaño ORIGINAL, no al train restante
        val_relative = val_size / (1.0 - test_size)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr, y_tr, test_size=val_relative, stratify=y_tr, random_state=seed
        )
        logger.info(
            f"Split → train: {len(X_tr):,} | val: {len(X_val):,} | test: {len(X_test):,}"
        )
        return X_tr, X_val, X_test, y_tr, y_val, y_test

    # ── class weights ─────────────────────────────────────────────────────────

    @staticmethod
    def calcular_class_weights(y_train: np.ndarray) -> Dict[int, float]:
        """
        Calcula class_weights para compensar el desbalance 75/25.

        class_weight = n_samples / (n_classes * n_samples_per_class)

        Con ~5680 no-desertores y ~1520 desertores:
          w_0 ≈ 0.66  |  w_1 ≈ 2.50

        Esto equivale a penalizar al modelo ~3.7x más por cada falso
        negativo (desertor no detectado), lo cual es correcto para un
        sistema de alertas tempranas donde el costo de omisión es alto.
        """
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        cw = dict(zip(classes.tolist(), weights.tolist()))
        logger.info(f"Class weights: {cw}")
        return cw

    # ── pipeline completo ─────────────────────────────────────────────────────

    def pipeline_completo(
        self,
        ruta_acad: str | Path,
        ruta_demo: str | Path,
        ruta_des:  str | Path,
    ) -> dict:
        """
        Ejecuta el pipeline completo: cargar → secuencias → split → normalizar.

        Returns
        ───────
        dict con claves:
          X_train, X_val, X_test, y_train, y_val, y_test,
          class_weights, scaler, n_features, max_semestres
        """
        df_ac, df_dem, df_des = self.cargar_datasets(ruta_acad, ruta_demo, ruta_des)
        X, y, lens, ids        = self.construir_secuencias(df_ac, df_dem, df_des)
        X_tr, X_va, X_te, y_tr, y_va, y_te = self.split(X, y)
        X_tr, X_va, X_te       = self.normalizar(X_tr, X_va, X_te)
        cw                     = self.calcular_class_weights(y_tr)

        return {
            "X_train":       X_tr,
            "X_val":         X_va,
            "X_test":        X_te,
            "y_train":       y_tr,
            "y_val":         y_va,
            "y_test":        y_te,
            "class_weights": cw,
            "scaler":        self.scaler,
            "n_features":    X_tr.shape[2],
            "max_semestres": self.max_semestres,
            "lens":          lens,
            "ids":           ids,
        }
