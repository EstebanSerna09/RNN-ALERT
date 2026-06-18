"""
utils/persistencia.py
=====================
Funciones de persistencia de DataFrames procesados por los pipelines.

Garantiza que los DataFrames importantes queden guardados en disco
para que la app pueda recuperarlos después de recargar Streamlit.

Convención de rutas:
  outputs/intermedios/   ← ya usada por el pipeline actual (no se cambia)

Los DataFrames clave se detectan automáticamente desde esa carpeta.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

# ── Nombres canónicos de los DataFrames clave ─────────────────────────────────
# Mapeados a sus rutas reales dentro de outputs/intermedios/
_DF_RUTAS: dict[str, str] = {
    "df_entrenamiento_clasificable":   "DF_ENTRENAMIENTO_CLASIFICABLES.csv",
    "df_no_clasificable_prediccion":   "DF_NO_CLASIFICABLES_PREDICCION.csv",
    "df_entrenamiento_no_clasificable":"DF_NO_CLASIFICABLES_DEMOGRAFICOS.csv",
    "df_academico_final":              "DF_ACADEMICO_3_FINAL.csv",
    "df_demograficos_propedeutico":    "DF_DEMOGRAFICOS_3_PROPEDEUTICO.csv",
    "df_desertores":                   "DF_DESERTORES.csv",
}


def get_inter_dir(base_dir: Path) -> Path:
    """Retorna el directorio outputs/intermedios/ creándolo si no existe."""
    d = base_dir / "outputs" / "intermedios"
    d.mkdir(parents=True, exist_ok=True)
    return d


def existe_dataframe(nombre_canonico: str, base_dir: Path) -> bool:
    """Retorna True si el CSV del DataFrame existe en disco."""
    nombre_archivo = _DF_RUTAS.get(nombre_canonico)
    if not nombre_archivo:
        return False
    ruta = get_inter_dir(base_dir) / nombre_archivo
    return ruta.exists() and ruta.stat().st_size > 0


def cargar_dataframe_csv(nombre_canonico: str, base_dir: Path) -> Optional[pd.DataFrame]:
    """
    Carga un DataFrame desde disco.

    Args:
        nombre_canonico : clave del mapa _DF_RUTAS
        base_dir        : directorio raíz del proyecto

    Returns:
        DataFrame o None si el archivo no existe o hay error.
    """
    nombre_archivo = _DF_RUTAS.get(nombre_canonico)
    if not nombre_archivo:
        logger.warning(f"[Persistencia] Nombre desconocido: {nombre_canonico}")
        return None
    ruta = get_inter_dir(base_dir) / nombre_archivo
    if not ruta.exists():
        return None
    try:
        df = pd.read_csv(ruta)
        logger.info(f"[Persistencia] Cargado {nombre_canonico} ({len(df):,} filas) desde {ruta}")
        return df
    except Exception as e:
        logger.error(f"[Persistencia] Error cargando {nombre_canonico}: {e}")
        return None


def get_bytes_csv(nombre_canonico: str, base_dir: Path) -> Optional[bytes]:
    """
    Lee el CSV de disco y devuelve los bytes crudos para st.download_button.
    Si el DataFrame ya está en session_state, lo usa directamente.

    Returns:
        bytes del CSV, o None si no existe.
    """
    # Intentar desde session_state primero (más rápido)
    df_ss = st.session_state.get(nombre_canonico)
    if df_ss is not None and isinstance(df_ss, pd.DataFrame) and len(df_ss) > 0:
        return df_ss.to_csv(index=False).encode("utf-8")

    # Fallback: leer desde disco
    nombre_archivo = _DF_RUTAS.get(nombre_canonico)
    if not nombre_archivo:
        return None
    ruta = get_inter_dir(base_dir) / nombre_archivo
    if not ruta.exists():
        return None
    try:
        return ruta.read_bytes()
    except Exception as e:
        logger.error(f"[Persistencia] Error leyendo bytes de {nombre_canonico}: {e}")
        return None


def boton_descarga_df(
    nombre_canonico: str,
    base_dir: Path,
    label_boton: str = None,
    nombre_archivo_descarga: str = None,
    key: str = None,
) -> None:
    """
    Muestra un botón de descarga para un DataFrame, leyendo desde disco
    si no está en session_state. Si el archivo no existe, muestra aviso.

    Args:
        nombre_canonico          : clave en _DF_RUTAS
        base_dir                 : raíz del proyecto
        label_boton              : texto del botón (por defecto auto-generado)
        nombre_archivo_descarga  : nombre del archivo descargado
        key                      : key único de Streamlit
    """
    nombre_archivo = _DF_RUTAS.get(nombre_canonico, f"{nombre_canonico}.csv")
    label = label_boton or f"⬇️ Descargar {nombre_archivo}"
    fname = nombre_archivo_descarga or nombre_archivo
    k     = key or f"dl_{nombre_canonico}"

    data = get_bytes_csv(nombre_canonico, base_dir)
    if data is None:
        st.warning(
            f"⚠️ **{nombre_archivo}** aún no ha sido generado. "
            "Ejecuta primero el pipeline de limpieza."
        )
    else:
        st.download_button(
            label    = label,
            data     = data,
            file_name= fname,
            mime     = "text/csv",
            key      = k,
        )


def registrar_dataframes_en_session(base_dir: Path) -> int:
    """
    Al iniciar la app, detecta qué DataFrames existen en disco y los carga
    en st.session_state para que estén disponibles sin tener que reejecutar
    los pipelines.

    Returns:
        Número de DataFrames cargados nuevamente en sesión.
    """
    cargados = 0
    for nombre_canonico in _DF_RUTAS:
        # Solo cargar si todavía no está en session_state
        if st.session_state.get(nombre_canonico) is not None:
            continue
        df = cargar_dataframe_csv(nombre_canonico, base_dir)
        if df is not None:
            st.session_state[nombre_canonico] = df
            cargados += 1
    if cargados:
        logger.info(f"[Persistencia] {cargados} DataFrames restaurados desde disco a session_state.")
    return cargados
