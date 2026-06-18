"""utils/session.py — Gestión del estado global de sesión Streamlit."""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict

import streamlit as st

logger = logging.getLogger(__name__)

_KEYS = [
    "df_demo_raw", "df_acad_raw",
    "df_demo_1", "df_acad_1", "df_demo_2", "df_acad_2", "df_acad_3",
    "df_desertores", "pipeline_ok",
    "modelos_entrenados",
    "X_train", "X_val", "X_test", "y_train", "y_val", "y_test",
    "class_weights", "n_features", "scaler_rnn",
    "rnn_datos",
    "_rnn_auto_registrado",
]

def init_session() -> None:
    for k in _KEYS:
        if k not in st.session_state:
            if k == "modelos_entrenados":
                st.session_state[k] = {}
            elif k in ("pipeline_ok", "_rnn_auto_registrado"):
                st.session_state[k] = False
            else:
                st.session_state[k] = None

def registrar_modelo(
    nombre: str,
    modelo: Any,
    scaler: Any,
    metricas: Dict,
    tipo: str,
    programa: str | None = None,
    **extra: Any,
) -> None:
    """
    Registra un modelo entrenado en session_state.

    Nota importante:
    - `programa` se guarda explícitamente para que las vistas de
      Comparación de Modelos y Evaluación General no muestren las RNN
      por programa como "Global".
    - `extra` permite conservar metadatos adicionales sin romper llamadas
      existentes.
    """
    info = {
        "modelo": modelo,
        "scaler": scaler,
        "metricas": metricas,
        "tipo": tipo,
    }
    if programa:
        info["programa"] = str(programa)
    if extra:
        info.update(extra)
    st.session_state["modelos_entrenados"][nombre] = info

def obtener_modelos() -> Dict[str, Dict]:
    return st.session_state.get("modelos_entrenados", {})

def pipeline_listo() -> bool:
    return bool(st.session_state.get("pipeline_ok", False))

def datos_rnn_listos() -> bool:
    return st.session_state.get("X_train") is not None

def auto_registrar_modelos_rnn(outputs_dir: Path) -> int:
    """
    Escanea outputs/modelos/por_programa/ y registra en session_state
    todos los programas RNN que tengan artefactos en disco (modelo + config + metricas).

    Se llama automáticamente al cargar la app para que los modelos RNN entrenados
    aparezcan en el selector sin necesidad de reentrenar en cada sesión.

    Returns:
        Número de modelos nuevos registrados.
    """
    # Evitar registrar dos veces en la misma sesión
    if st.session_state.get("_rnn_auto_registrado", False):
        return 0

    try:
        from modelos.entrenamiento_masivo import safe_a_nombre
    except ImportError:
        logger.warning("[AutoRegistro] No se pudo importar entrenamiento_masivo.")
        st.session_state["_rnn_auto_registrado"] = True
        return 0

    modelos_actuales = obtener_modelos()
    dir_por_prog = outputs_dir / "modelos" / "por_programa"
    if not dir_por_prog.exists():
        st.session_state["_rnn_auto_registrado"] = True
        return 0

    registrados = 0
    for dir_prog in sorted(dir_por_prog.iterdir()):
        if not dir_prog.is_dir():
            continue
        safe_prog = dir_prog.name

        ruta_cfg = dir_prog / "config.pkl"
        tiene_modelo = (
            (dir_prog / "modelo.keras").exists() or
            (dir_prog / "mejor_modelo_rnn.keras").exists() or
            (dir_prog / "modelo.h5").exists()
        )
        if not (ruta_cfg.exists() and tiene_modelo):
            continue

        nombre_humano = safe_a_nombre(safe_prog)
        nombre_mod    = f"RNN Multitarea ({nombre_humano})"

        if nombre_mod in modelos_actuales:
            continue

        # Leer métricas del JSON para mostrarlas en comparación/evaluación
        metricas: dict = {}
        ruta_met = dir_prog / "metricas.json"
        if ruta_met.exists():
            try:
                with open(ruta_met, encoding="utf-8") as f:
                    metricas = json.load(f)
            except Exception as e:
                logger.warning(f"[AutoRegistro] No se pudieron leer métricas de {dir_prog}: {e}")

        # Registrar con modelo=None; el modelo Keras se carga al predecir
        registrar_modelo(
            nombre=nombre_mod,
            modelo=None,
            scaler=None,
            metricas=metricas,
            tipo="rnn_multi",
            programa=safe_prog,
        )
        registrados += 1
        logger.info(f"[AutoRegistro] Registrado desde disco: {nombre_mod}")

    st.session_state["_rnn_auto_registrado"] = True
    if registrados > 0:
        logger.info(f"[AutoRegistro] {registrados} modelo(s) RNN registrados desde disco.")

    return registrados
