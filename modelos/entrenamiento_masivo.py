"""
modelos/entrenamiento_masivo.py
================================
Motor de entrenamiento masivo del modelo RNN Multimodal Multitarea.

Recorre todos los programas académicos disponibles, entrena un modelo
independiente por programa, guarda los artefactos y genera un reporte
consolidado.

Funciones públicas:
  - obtener_programas_disponibles()
  - entrenar_programa_individual()
  - entrenar_todos_los_programas()
  - cargar_reporte_masivo()

Estructura de salida:
  outputs/modelos/por_programa/
      Administracion_Empresas/
          modelo.keras
          config.pkl
          metricas.json
      Ingenieria_Informatica/
          modelo.keras
          config.pkl
          metricas.json
      ...
  outputs/resultados/
      metricas_por_programa.csv
      resumen_entrenamiento.json
"""
from __future__ import annotations

import json
import logging
import pickle
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from modelos.modelo_rnn import (
    preparar_datos_rnn,
    entrenar_modelo_rnn,
    evaluar_modelo_rnn,
)

logger = logging.getLogger(__name__)


# ── Rutas por defecto ─────────────────────────────────────────────────────────
def _dir_carrera(outputs_dir: Path) -> Path:
    return outputs_dir / "por_carrera"

def _dir_modelos_prog(outputs_dir: Path) -> Path:
    return outputs_dir / "modelos" / "por_programa"

def _dir_resultados(outputs_dir: Path) -> Path:
    return outputs_dir / "resultados"


# ══════════════════════════════════════════════════════════════════════════════
# DETECCIÓN DE PROGRAMAS
# ══════════════════════════════════════════════════════════════════════════════

def obtener_programas_disponibles(outputs_dir: Path) -> List[str]:
    """
    Detecta los programas académicos disponibles leyendo los CSV generados
    por el Paso 7 del pipeline.

    Un programa se considera disponible si existen los tres archivos:
      - demograficos/DF_{SAFE}_DEMOGRAFICOS.csv
      - academicos/DF_{SAFE}_ACADEMICOS.csv
      - desertores/DF_{SAFE}_DESERTORES.csv

    Returns:
        Lista de nombres de programa en formato seguro para archivos
        (e.g. "ADMINISTRACION_DE_EMPRESAS")
    """
    dir_demo = _dir_carrera(outputs_dir) / "demograficos"
    dir_acad = _dir_carrera(outputs_dir) / "academicos"
    dir_des  = _dir_carrera(outputs_dir) / "desertores"

    if not dir_demo.exists():
        return []

    programas = []
    for csv_demo in sorted(dir_demo.glob("DF_*_DEMOGRAFICOS.csv")):
        # Extraer nombre seguro del programa del nombre del archivo
        safe = csv_demo.stem.replace("DF_", "").replace("_DEMOGRAFICOS", "")
        ruta_a = dir_acad / f"DF_{safe}_ACADEMICOS.csv"
        ruta_l = dir_des  / f"DF_{safe}_DESERTORES.csv"
        if ruta_a.exists() and ruta_l.exists():
            programas.append(safe)

    logger.info(f"[Masivo] {len(programas)} programas detectados.")
    return programas


def safe_a_nombre(safe: str) -> str:
    """Convierte 'ADMINISTRACION_DE_EMPRESAS' → 'Administración De Empresas'."""
    return safe.replace("_", " ").title()


# ══════════════════════════════════════════════════════════════════════════════
# GUARDAR ARTEFACTOS POR PROGRAMA
# ══════════════════════════════════════════════════════════════════════════════

def _guardar_artefactos_programa(
    modelo:       Any,
    metricas:     Dict,
    datos_config: Dict,
    safe_prog:    str,
    outputs_dir:  Path,
) -> Path:
    """
    Guarda modelo .keras, config .pkl y métricas .json para un programa.

    Returns:
        Ruta del directorio del programa.
    """
    dir_prog = _dir_modelos_prog(outputs_dir) / safe_prog
    dir_prog.mkdir(parents=True, exist_ok=True)

    # Modelo Keras
    modelo.save(str(dir_prog / "modelo.keras"))

    # Configuración
    config = {
        "max_semestres":      datos_config["max_semestres"],
        "num_features_acad":  datos_config["num_features_acad"],
        "num_features_socio": datos_config["num_features_socio"],
        "features_acad_cols": datos_config["features_acad_cols"],
        "features_socio_cols": datos_config.get("features_socio_cols", []),
        "programa":           safe_prog,
    }
    with open(dir_prog / "config.pkl", "wb") as f:
        pickle.dump(config, f)

    # Métricas (solo valores serializables, sin arrays grandes)
    metricas_guardables = {
        k: v for k, v in metricas.items()
        if k not in ["fpr", "tpr", "prec_curve", "rec_curve",
                     "y_proba", "y_true", "y_pred_nota", "y_true_nota", "cm"]
    }
    metricas_guardables["programa"]  = safe_prog
    metricas_guardables["timestamp"] = datetime.now().isoformat()

    with open(dir_prog / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(metricas_guardables, f, ensure_ascii=False, indent=2)

    logger.info(f"[Masivo] Artefactos guardados en: {dir_prog}")
    return dir_prog


# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO INDIVIDUAL (wrapper con manejo de errores)
# ══════════════════════════════════════════════════════════════════════════════

def entrenar_programa_individual(
    safe_prog:   str,
    outputs_dir: Path,
    epochs:      int   = 150,
    batch_size:  int   = 32,
    patience:    int   = 15,
    seed:        int   = 42,
) -> Dict[str, Any]:
    """
    Entrena el modelo RNN para UN programa académico.

    Args:
        safe_prog   : Nombre seguro del programa (e.g. "ADMINISTRACION_DE_EMPRESAS")
        outputs_dir : Directorio raíz de outputs del proyecto
        epochs      : Épocas máximas
        batch_size  : Tamaño de lote
        patience    : Paciencia para EarlyStopping
        seed        : Semilla de reproducibilidad

    Returns:
        dict con claves:
          "ok"       : bool — True si se entrenó correctamente
          "programa" : str  — nombre seguro del programa
          "metricas" : dict — métricas del modelo (vacío si falló)
          "modelo"   : modelo Keras (None si falló)
          "datos"    : datos de preparación (None si falló)
          "error"    : str  — mensaje de error (vacío si ok)
          "duracion_seg": float — segundos de entrenamiento
    """
    dir_demo = _dir_carrera(outputs_dir) / "demograficos" / f"DF_{safe_prog}_DEMOGRAFICOS.csv"
    dir_acad = _dir_carrera(outputs_dir) / "academicos"   / f"DF_{safe_prog}_ACADEMICOS.csv"
    dir_des  = _dir_carrera(outputs_dir) / "desertores"   / f"DF_{safe_prog}_DESERTORES.csv"

    t_inicio = time.time()
    logger.info(f"[Masivo] ── Iniciando entrenamiento: {safe_prog}")

    try:
        # 1. Cargar datos
        df_s = pd.read_csv(dir_demo)
        df_a = pd.read_csv(dir_acad)
        df_l = pd.read_csv(dir_des)

        # Validación mínima: necesitamos al menos 10 estudiantes para hacer split
        ids_comunes = set(df_s["ID_EST"]).intersection(set(df_a["ID_EST"])).intersection(set(df_l["ID_EST"]))
        if len(ids_comunes) < 10:
            raise ValueError(f"Estudiantes insuficientes: {len(ids_comunes)} (mínimo 10)")

        # 2. Preparar datos
        logger.info(f"[Masivo]   Preparando datos — {len(ids_comunes)} estudiantes")
        datos = preparar_datos_rnn(df_s, df_a, df_l, seed=seed)

        # 3. Entrenar
        dir_ckpt = _dir_modelos_prog(outputs_dir) / safe_prog
        dir_ckpt.mkdir(parents=True, exist_ok=True)
        logger.info(f"[Masivo]   Entrenando ({epochs} épocas máx, patience={patience})")
        modelo, _ = entrenar_modelo_rnn(
            datos,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            ruta_salida=dir_ckpt,
            seed=seed,
        )

        # 4. Evaluar
        logger.info(f"[Masivo]   Evaluando modelo")
        metricas = evaluar_modelo_rnn(modelo, datos)

        # 5. Guardar artefactos
        _guardar_artefactos_programa(modelo, metricas, datos, safe_prog, outputs_dir)

        duracion = round(time.time() - t_inicio, 1)
        logger.info(f"[Masivo] ✔ {safe_prog} completado en {duracion}s — "
                    f"AUC={metricas['roc_auc']:.4f} Recall={metricas['recall_opt']:.4f}")

        return {
            "ok":           True,
            "programa":     safe_prog,
            "metricas":     metricas,
            "modelo":       modelo,
            "datos":        datos,
            "error":        "",
            "duracion_seg": duracion,
        }

    except Exception as e:
        duracion = round(time.time() - t_inicio, 1)
        msg = f"{type(e).__name__}: {e}"
        logger.error(f"[Masivo] ✘ {safe_prog} falló ({duracion}s) — {msg}")
        logger.debug(traceback.format_exc())
        return {
            "ok":           False,
            "programa":     safe_prog,
            "metricas":     {},
            "modelo":       None,
            "datos":        None,
            "error":        msg,
            "duracion_seg": duracion,
        }


# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO MASIVO — TODOS LOS PROGRAMAS
# ══════════════════════════════════════════════════════════════════════════════

def entrenar_todos_los_programas(
    outputs_dir:      Path,
    epochs:           int   = 150,
    batch_size:       int   = 32,
    patience:         int   = 15,
    seed:             int   = 42,
    callback_progreso: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Entrena el modelo RNN para TODOS los programas académicos disponibles.

    Continúa automáticamente aunque un programa falle.

    Args:
        outputs_dir       : Directorio raíz de outputs
        epochs            : Épocas máximas por programa
        batch_size        : Tamaño de lote
        patience          : Paciencia EarlyStopping
        seed              : Semilla de reproducibilidad
        callback_progreso : Función opcional llamada tras cada programa.
                            Firma: callback_progreso(idx_actual, total, safe_prog)
                            Útil para actualizar st.progress() desde la UI.

    Returns:
        dict con claves:
          "resultados"     : List[dict] — resultado de cada programa
          "programas_ok"   : List[str]  — programas entrenados con éxito
          "programas_error": List[str]  — programas que fallaron
          "resumen"        : dict       — estadísticas globales
          "df_metricas"    : pd.DataFrame — tabla comparativa
    """
    programas = obtener_programas_disponibles(outputs_dir)
    if not programas:
        raise ValueError("No hay programas disponibles. Ejecuta primero el Pipeline.")

    total      = len(programas)
    resultados: List[Dict] = []
    ok_list:    List[str]  = []
    err_list:   List[str]  = []

    logger.info(f"[Masivo] ══ Iniciando entrenamiento masivo: {total} programas")
    t_global = time.time()

    for idx, safe_prog in enumerate(programas):
        logger.info(f"[Masivo] ── Programa {idx+1}/{total}: {safe_prog}")

        if callback_progreso:
            try:
                callback_progreso(idx, total, safe_prog)
            except Exception:
                pass  # El callback nunca debe detener el entrenamiento

        resultado = entrenar_programa_individual(
            safe_prog=safe_prog,
            outputs_dir=outputs_dir,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            seed=seed,
        )
        resultados.append(resultado)

        if resultado["ok"]:
            ok_list.append(safe_prog)
        else:
            err_list.append(safe_prog)

    # Notificar finalización al callback
    if callback_progreso:
        try:
            callback_progreso(total, total, "COMPLETADO")
        except Exception:
            pass

    duracion_total = round(time.time() - t_global, 1)
    logger.info(f"[Masivo] ══ Finalizado: {len(ok_list)}/{total} OK | "
                f"{len(err_list)} errores | {duracion_total}s totales")

    # ── Tabla de métricas ─────────────────────────────────────────────────
    filas_metricas = []
    for r in resultados:
        if r["ok"] and r["metricas"]:
            m = r["metricas"]
            filas_metricas.append({
                "programa":        r["programa"],
                "roc_auc":         round(m.get("roc_auc",0),    4),
                "recall":          round(m.get("recall_opt",0),  4),
                "precision":       round(m.get("precision_opt",0),4),
                "f1":              round(m.get("f1_opt",0),      4),
                "pr_auc":          round(m.get("pr_auc",0),      4),
                "mae":             round(m.get("mae",0),         4),
                "r2":              round(m.get("r2",0),          4),
                "umbral_optimo":   round(m.get("umbral_optimo",0.5),2),
                "duracion_seg":    r["duracion_seg"],
                "estado":          "✅ OK",
            })
        else:
            filas_metricas.append({
                "programa":    r["programa"],
                "roc_auc":     None, "recall": None, "precision": None,
                "f1":          None, "pr_auc": None, "mae":       None,
                "r2":          None, "umbral_optimo": None,
                "duracion_seg":r["duracion_seg"],
                "estado":      f"❌ {r['error'][:60]}",
            })
    df_metricas = pd.DataFrame(filas_metricas)

    # ── Resumen estadístico ───────────────────────────────────────────────
    df_ok = df_metricas[df_metricas["estado"] == "✅ OK"]
    resumen: Dict[str, Any] = {
        "total":           total,
        "ok":              len(ok_list),
        "errores":         len(err_list),
        "duracion_total_seg": duracion_total,
        "timestamp":       datetime.now().isoformat(),
        "programas_ok":    ok_list,
        "programas_error": err_list,
    }
    if not df_ok.empty:
        resumen["promedio_auc"]    = round(df_ok["roc_auc"].mean(), 4)
        resumen["promedio_recall"] = round(df_ok["recall"].mean(),  4)
        resumen["promedio_f1"]     = round(df_ok["f1"].mean(),      4)
        resumen["mejor_programa"]  = df_ok.loc[df_ok["recall"].idxmax(), "programa"]
        resumen["peor_programa"]   = df_ok.loc[df_ok["recall"].idxmin(), "programa"]
        resumen["mejor_recall"]    = round(df_ok["recall"].max(), 4)
        resumen["peor_recall"]     = round(df_ok["recall"].min(), 4)

    # ── Guardar resultados en disco ───────────────────────────────────────
    _guardar_reporte(df_metricas, resumen, outputs_dir)

    return {
        "resultados":      resultados,
        "programas_ok":    ok_list,
        "programas_error": err_list,
        "resumen":         resumen,
        "df_metricas":     df_metricas,
    }


# ══════════════════════════════════════════════════════════════════════════════
# GUARDAR Y CARGAR REPORTE
# ══════════════════════════════════════════════════════════════════════════════

def _guardar_reporte(
    df_metricas: pd.DataFrame,
    resumen:     Dict,
    outputs_dir: Path,
) -> None:
    """Guarda metricas_por_programa.csv y resumen_entrenamiento.json."""
    dir_res = _dir_resultados(outputs_dir)
    dir_res.mkdir(parents=True, exist_ok=True)

    ruta_csv  = dir_res / "metricas_por_programa.csv"
    ruta_json = dir_res / "resumen_entrenamiento.json"

    df_metricas.to_csv(ruta_csv, index=False, encoding="utf-8")
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    logger.info(f"[Masivo] Reporte guardado: {ruta_csv} | {ruta_json}")


def cargar_reporte_masivo(outputs_dir: Path) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    """
    Carga el reporte de un entrenamiento masivo previo.

    Returns:
        (df_metricas, resumen) o (None, None) si no existe.
    """
    dir_res   = _dir_resultados(outputs_dir)
    ruta_csv  = dir_res / "metricas_por_programa.csv"
    ruta_json = dir_res / "resumen_entrenamiento.json"

    if not ruta_csv.exists():
        return None, None

    df = pd.read_csv(ruta_csv)
    resumen = {}
    if ruta_json.exists():
        with open(ruta_json, encoding="utf-8") as f:
            resumen = json.load(f)

    return df, resumen
