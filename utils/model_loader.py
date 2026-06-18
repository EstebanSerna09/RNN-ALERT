"""
utils/model_loader.py
=====================
Carga automática de TODOS los modelos entrenados (RNN + ML clásicos)
desde disco al iniciar la app, sin requerir reentrenamiento.

Modelos RNN:
  outputs/modelos/por_programa/<PROG>/modelo.keras + config.pkl + metricas.json

Modelos ML clásicos:
  outputs/modelos_ml/por_programa/<PROG>/*.pkl + config_socio.pkl + config_acad.pkl + metricas.json
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE MODELOS RNN
# ══════════════════════════════════════════════════════════════════════════════

def _nombre_humano(safe_prog: str) -> str:
    return safe_prog.replace("_", " ").title()


def _buscar_archivo_modelo_keras(dir_prog: Path) -> Optional[Path]:
    for nombre in ["modelo.keras", "mejor_modelo_rnn.keras", "modelo.h5"]:
        ruta = dir_prog / nombre
        if ruta.exists():
            return ruta
    return None


def _cargar_modelo_keras(ruta: Path) -> Optional[Any]:
    try:
        from tensorflow.keras.models import load_model
        modelo = load_model(str(ruta))
        logger.info(f"[RNN-Loader] Modelo cargado: {ruta.name}")
        return modelo
    except ImportError:
        logger.error("[RNN-Loader] TensorFlow no disponible.")
        return None
    except Exception as e:
        logger.error(f"[RNN-Loader] Error cargando {ruta}: {e}")
        return None


def _config_rnn_completo(config: Dict) -> tuple[bool, list]:
    claves = ["features_acad_cols", "max_semestres", "num_features_acad", "num_features_socio"]
    faltantes = [k for k in claves if k not in config]
    return len(faltantes) == 0, faltantes


def cargar_modelos_rnn_desde_disco(outputs_dir: Path) -> Dict[str, Dict]:
    """
    Escanea outputs/modelos/por_programa/ y carga todos los modelos RNN.

    Returns:
        {nombre_modelo: info_dict}
    """
    dir_por_prog = outputs_dir / "modelos" / "por_programa"
    if not dir_por_prog.exists():
        return {}

    resultado: Dict[str, Dict] = {}

    for dir_prog in sorted(dir_por_prog.iterdir()):
        if not dir_prog.is_dir():
            continue
        safe_prog = dir_prog.name

        ruta_modelo = _buscar_archivo_modelo_keras(dir_prog)
        if ruta_modelo is None:
            logger.warning(f"[RNN-Loader] Sin modelo .keras en {dir_prog.name}")
            continue

        ruta_cfg = dir_prog / "config.pkl"
        if not ruta_cfg.exists():
            logger.warning(f"[RNN-Loader] Sin config.pkl en {dir_prog.name}")
            continue

        try:
            with open(ruta_cfg, "rb") as f:
                config = pickle.load(f)
        except Exception as e:
            logger.error(f"[RNN-Loader] Error leyendo config de {safe_prog}: {e}")
            continue

        completo, faltantes = _config_rnn_completo(config)
        if not completo:
            logger.warning(f"[RNN-Loader] config.pkl de {safe_prog} incompleto. "
                           f"Faltan: {faltantes}. Reentrena para generar config completo.")
            continue

        modelo = _cargar_modelo_keras(ruta_modelo)
        if modelo is None:
            continue

        metricas = {}
        ruta_met = dir_prog / "metricas.json"
        if ruta_met.exists():
            try:
                with open(ruta_met, encoding="utf-8") as f:
                    metricas = json.load(f)
            except Exception as e:
                logger.warning(f"[RNN-Loader] metricas.json de {safe_prog}: {e}")

        umbral = float(metricas.get("umbral_optimo", 0.5))
        nombre_mod = f"RNN Multitarea ({_nombre_humano(safe_prog)})"
        datos_rnn = {
            "max_semestres":      config["max_semestres"],
            "num_features_acad":  config["num_features_acad"],
            "num_features_socio": config["num_features_socio"],
            "features_acad_cols": config["features_acad_cols"],
            "features_socio_cols": config.get("features_socio_cols", []),
            "umbral_optimo":      umbral,
            "programa":           config.get("programa", safe_prog),
            "_cargado_desde_disco": True,
            "_ruta_modelo": str(ruta_modelo),
            # Tensores no disponibles desde disco
            "X_sec_train": None, "X_sec_test": None,
            "X_est_train": None, "X_est_test": None,
            "y_des_train": None, "y_des_test": None,
            "y_not_train": None, "y_not_test": None,
            "estudiantes_comunes": [], "df_socio": None, "df_acad": None,
        }

        resultado[nombre_mod] = {
            "modelo":    modelo,
            "scaler":    None,
            "metricas":  metricas,
            "tipo":      "rnn_multi",
            "programa":  config.get("programa", safe_prog),
            "datos_rnn": datos_rnn,
        }
        logger.info(f"[RNN-Loader] ✔ {nombre_mod}")

    logger.info(f"[RNN-Loader] {len(resultado)} modelo(s) RNN cargados desde disco.")
    return resultado


# ══════════════════════════════════════════════════════════════════════════════
# CARGA AUTOMÁTICA UNIFICADA (RNN + ML)
# ══════════════════════════════════════════════════════════════════════════════

def auto_cargar_modelos_en_session(outputs_dir: Path) -> int:
    """
    Carga automáticamente TODOS los modelos (RNN + ML clásicos) desde disco
    y los registra en st.session_state["modelos_entrenados"].

    - Evita duplicados (no sobreescribe modelos ya en sesión).
    - Actualiza modelo Keras si estaba registrado con modelo=None.
    - Ejecuta solo una vez por sesión (guarda bandera).

    Returns:
        Total de modelos nuevos registrados.
    """
    import streamlit as st

    if st.session_state.get("_modelos_cargados_desde_disco", False):
        return 0

    modelos_actuales = st.session_state.get("modelos_entrenados", {})
    registrados = 0

    # ── 1. Cargar RNN ──────────────────────────────────────────────────
    try:
        rnn_disco = cargar_modelos_rnn_desde_disco(outputs_dir)
        for nombre, info in rnn_disco.items():
            if nombre in modelos_actuales:
                # Si el modelo Keras era None (venía de autoregistro ligero), actualizar
                if modelos_actuales[nombre].get("modelo") is None:
                    modelos_actuales[nombre]["modelo"] = info["modelo"]
                    modelos_actuales[nombre]["datos_rnn"] = info.get("datos_rnn")
                modelos_actuales[nombre].setdefault("programa", info.get("programa"))
            else:
                modelos_actuales[nombre] = info
                registrados += 1
    except Exception as e:
        logger.warning(f"[AutoCarga] Error cargando RNN: {e}")

    # ── 2. Cargar ML clásicos ──────────────────────────────────────────
    try:
        from modelos.modelos_ml_por_programa import cargar_modelos_ml_desde_disco
        ml_disco = cargar_modelos_ml_desde_disco(outputs_dir)
        for nombre, info in ml_disco.items():
            if nombre not in modelos_actuales:
                modelos_actuales[nombre] = info
                registrados += 1
    except Exception as e:
        logger.warning(f"[AutoCarga] Error cargando ML clásicos: {e}")

    st.session_state["modelos_entrenados"] = modelos_actuales
    st.session_state["_modelos_cargados_desde_disco"] = True

    # ── 3. rnn_datos compatibilidad ────────────────────────────────────
    if st.session_state.get("rnn_datos") is None:
        rnn_mods = {k: v for k, v in modelos_actuales.items()
                    if v.get("tipo") == "rnn_multi"}
        if len(rnn_mods) == 1:
            datos = next(iter(rnn_mods.values())).get("datos_rnn")
            if datos:
                st.session_state["rnn_datos"] = datos

    if registrados > 0:
        logger.info(f"[AutoCarga] {registrados} modelo(s) nuevos cargados desde disco.")
    return registrados


def config_pkl_rnn_es_completo(dir_prog: Path) -> tuple[bool, list]:
    """Verifica si el config.pkl de un programa RNN tiene claves mínimas."""
    ruta = dir_prog / "config.pkl"
    if not ruta.exists():
        return False, ["config.pkl no encontrado"]
    try:
        with open(ruta, "rb") as f:
            config = pickle.load(f)
    except Exception as e:
        return False, [f"Error leyendo config.pkl: {e}"]
    return _config_rnn_completo(config)
