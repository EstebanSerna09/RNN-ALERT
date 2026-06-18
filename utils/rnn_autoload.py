"""
utils/rnn_autoload.py
=====================
Auto-registro de modelos RNN desde disco al iniciar la app.

Cuando la app se recarga (nueva sesión de Streamlit), st.session_state queda vacío
y los modelos entrenados desaparecen de la UI, aunque sus artefactos existan en disco.

Esta función detecta los modelos RNN guardados en:
    outputs/modelos/por_programa/<SAFE_PROG>/

Y los registra en session_state con modelo=None (el modelo Keras se carga bajo demanda
en el momento de la predicción individual desde _cargar_artefactos_rnn_para_prediccion).

Esto permite que los modelos RNN aparezcan en el selector de "Predicción Individual"
sin necesidad de reentrenar en cada sesión.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def auto_registrar_modelos_rnn(
    outputs_dir: Path,
    registrar_modelo_fn,
    obtener_modelos_fn,
) -> int:
    """
    Escanea outputs/modelos/por_programa/ y registra en session_state
    todos los programas que tengan artefactos completos (modelo + config + metricas).

    Args:
        outputs_dir         : Directorio raíz de outputs (BASE_DIR / "outputs")
        registrar_modelo_fn : Función registrar_modelo() de utils/session.py
        obtener_modelos_fn  : Función obtener_modelos() de utils/session.py

    Returns:
        Número de modelos nuevos registrados en esta llamada.
    """
    try:
        from modelos.entrenamiento_masivo import safe_a_nombre
    except ImportError:
        logger.warning("[AutoRegistro] No se pudo importar entrenamiento_masivo.")
        return 0

    modelos_actuales = obtener_modelos_fn()
    dir_por_prog = outputs_dir / "modelos" / "por_programa"
    if not dir_por_prog.exists():
        return 0

    registrados = 0
    for dir_prog in sorted(dir_por_prog.iterdir()):
        if not dir_prog.is_dir():
            continue
        safe_prog = dir_prog.name

        # Verificar artefactos mínimos
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

        # No registrar duplicados
        if nombre_mod in modelos_actuales:
            continue

        # Leer métricas para mostrarlas en comparación/evaluación
        metricas: dict = {}
        ruta_met = dir_prog / "metricas.json"
        if ruta_met.exists():
            try:
                with open(ruta_met, encoding="utf-8") as f:
                    metricas = json.load(f)
            except Exception as e:
                logger.warning(f"[AutoRegistro] No se pudieron leer métricas de {dir_prog}: {e}")

        # Registrar con modelo=None; el modelo Keras se carga al predecir
        registrar_modelo_fn(
            nombre=nombre_mod,
            modelo=None,
            scaler=None,
            metricas=metricas,
            tipo="rnn_multi",
        )
        registrados += 1
        logger.info(f"[AutoRegistro] ✔ Registrado desde disco: {nombre_mod}")

    if registrados > 0:
        logger.info(f"[AutoRegistro] {registrados} modelo(s) RNN cargados desde disco.")

    return registrados
