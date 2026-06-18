"""
experimentos_rnn.py
===================
Script de experimentos controlados para optimización de la RNN — RNN-ALERT

Ejecuta 5 configuraciones independientes y compara sus métricas.
Guarda resultados en outputs/resultados/experimentos_comparativos.csv
y actualiza docs/optimizacion_rnn_resultados.md

Uso desde la carpeta raíz del proyecto:
    python experimentos_rnn.py

Configuraciones evaluadas:
    A: RNN original         (stratify=False, loss_weight=1.0, sample_weight=False)
    B: +stratify            (stratify=True,  loss_weight=1.0, sample_weight=False)
    C: +stratify +lw15      (stratify=True,  loss_weight=1.5, sample_weight=False)
    D: +stratify +lw20      (stratify=True,  loss_weight=2.0, sample_weight=False)
    E: +stratify +lw25      (stratify=True,  loss_weight=2.5, sample_weight=False)
    F: +stratify +lw20 +sw  (stratify=True,  loss_weight=2.0, sample_weight=True)
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Rutas ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
DOCS_DIR    = BASE_DIR / "docs"
DOCS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(OUTPUTS_DIR / "resultados" / "experimentos_rnn.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("experimentos")

from modelos.modelo_rnn import preparar_datos_rnn, entrenar_modelo_rnn, evaluar_modelo_rnn
from modelos.entrenamiento_masivo import obtener_programas_disponibles


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIONES
# ══════════════════════════════════════════════════════════════════════════════

CONFIGS = [
    {
        "id": "A_original",
        "nombre": "RNN original",
        "stratify_split":    False,
        "loss_weight_des":   1.0,
        "usar_sample_weight": False,
    },
    {
        "id": "B_stratify",
        "nombre": "RNN + stratify",
        "stratify_split":    True,
        "loss_weight_des":   1.0,
        "usar_sample_weight": False,
    },
    {
        "id": "C_str_lw15",
        "nombre": "RNN + stratify + lw1.5",
        "stratify_split":    True,
        "loss_weight_des":   1.5,
        "usar_sample_weight": False,
    },
    {
        "id": "D_str_lw20",
        "nombre": "RNN + stratify + lw2.0",
        "stratify_split":    True,
        "loss_weight_des":   2.0,
        "usar_sample_weight": False,
    },
    {
        "id": "E_str_lw25",
        "nombre": "RNN + stratify + lw2.5",
        "stratify_split":    True,
        "loss_weight_des":   2.5,
        "usar_sample_weight": False,
    },
    {
        "id": "F_str_lw20_sw",
        "nombre": "RNN + stratify + lw2.0 + sample_weight",
        "stratify_split":    True,
        "loss_weight_des":   2.0,
        "usar_sample_weight": True,
    },
]

# Programas prioritarios para análisis detallado
PROGRAMAS_PRIORITARIOS = [
    "ADMINISTRACION_DE_EMPRESAS",
    "TECNOLOGIA_EN_GESTION_COMERCIAL_Y_DE_MERCADOS",
    "TECNOLOGIA_EN_GESTION_EMPRESARIAL",
]


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL POR EXPERIMENTO
# ══════════════════════════════════════════════════════════════════════════════

def ejecutar_experimento(cfg: dict, programas: list) -> list:
    """
    Entrena y evalúa la configuración `cfg` en todos los programas.
    Devuelve lista de dicts con métricas por programa.
    """
    print(f"\n{'='*70}")
    print(f"  EXPERIMENTO: {cfg['nombre']}")
    print(f"  stratify={cfg['stratify_split']} | "
          f"lw_des={cfg['loss_weight_des']} | "
          f"sample_weight={cfg['usar_sample_weight']}")
    print(f"{'='*70}")

    resultados = []

    for prog in programas:
        dir_carrera = OUTPUTS_DIR / "por_carrera"
        ruta_demo = dir_carrera / "demograficos" / f"DF_{prog}_DEMOGRAFICOS.csv"
        ruta_acad = dir_carrera / "academicos"   / f"DF_{prog}_ACADEMICOS.csv"
        ruta_des  = dir_carrera / "desertores"   / f"DF_{prog}_DESERTORES.csv"

        if not (ruta_demo.exists() and ruta_acad.exists() and ruta_des.exists()):
            print(f"  [SKIP] {prog} — archivos no encontrados")
            continue

        t0 = time.time()
        print(f"  → {prog} ...", end=" ", flush=True)

        try:
            df_s = pd.read_csv(ruta_demo)
            df_a = pd.read_csv(ruta_acad)
            df_l = pd.read_csv(ruta_des)

            # Preparar datos con la configuración de split
            datos = preparar_datos_rnn(
                df_s, df_a, df_l,
                seed=42,
                stratify_split=cfg["stratify_split"],
            )

            # Directorio temporal para checkpoint (no sobreescribe producción)
            dir_exp = OUTPUTS_DIR / "intermedios" / "experimentos" / cfg["id"] / prog
            dir_exp.mkdir(parents=True, exist_ok=True)

            # Entrenar con la configuración
            modelo, _ = entrenar_modelo_rnn(
                datos,
                epochs=150,
                batch_size=32,
                patience=15,
                ruta_salida=dir_exp,
                seed=42,
                loss_weight_des=cfg["loss_weight_des"],
                usar_sample_weight=cfg["usar_sample_weight"],
            )

            metricas = evaluar_modelo_rnn(modelo, datos)

            duracion = round(time.time() - t0, 1)
            print(
                f"AUC={metricas['roc_auc']:.4f} | "
                f"Recall={metricas['recall_opt']:.4f} | "
                f"F1={metricas['f1_opt']:.4f} | "
                f"MAE={metricas['mae']:.4f} | "
                f"{duracion}s"
            )

            resultados.append({
                "experimento_id":   cfg["id"],
                "experimento":      cfg["nombre"],
                "programa":         prog,
                "stratify":         cfg["stratify_split"],
                "loss_weight_des":  cfg["loss_weight_des"],
                "sample_weight":    cfg["usar_sample_weight"],
                "roc_auc":          round(metricas["roc_auc"],       4),
                "recall":           round(metricas["recall_opt"],    4),
                "precision":        round(metricas["precision_opt"], 4),
                "f1":               round(metricas["f1_opt"],        4),
                "pr_auc":           round(metricas["pr_auc"],        4),
                "mae":              round(metricas["mae"],            4),
                "rmse":             round(metricas.get("rmse", 0),   4),
                "r2":               round(metricas["r2"],            4),
                "umbral_optimo":    round(metricas["umbral_optimo"], 2),
                "duracion_seg":     duracion,
                "ok":               True,
                "error":            "",
            })

        except Exception as e:
            duracion = round(time.time() - t0, 1)
            print(f"ERROR: {e}")
            logger.exception(f"Error en {cfg['id']} / {prog}: {e}")
            resultados.append({
                "experimento_id": cfg["id"],
                "experimento":    cfg["nombre"],
                "programa":       prog,
                "stratify":       cfg["stratify_split"],
                "loss_weight_des": cfg["loss_weight_des"],
                "sample_weight":  cfg["usar_sample_weight"],
                "roc_auc": None, "recall": None, "precision": None,
                "f1": None, "pr_auc": None, "mae": None, "rmse": None,
                "r2": None, "umbral_optimo": None, "duracion_seg": duracion,
                "ok": False, "error": str(e)[:120],
            })

    return resultados


# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS Y DECISIÓN
# ══════════════════════════════════════════════════════════════════════════════

def analizar_resultados(df: pd.DataFrame) -> dict:
    """
    Calcula promedios por experimento y determina la mejor configuración.
    Criterio: mejora F1 y recall promedio sin dañar gravemente MAE ni R².
    """
    df_ok = df[df["ok"] == True].copy()

    resumen = {}
    for exp_id, grupo in df_ok.groupby("experimento_id"):
        nombre = grupo["experimento"].iloc[0]
        resumen[exp_id] = {
            "nombre":            nombre,
            "n_programas":       len(grupo),
            "auc_promedio":      round(grupo["roc_auc"].mean(), 4),
            "recall_promedio":   round(grupo["recall"].mean(),  4),
            "f1_promedio":       round(grupo["f1"].mean(),      4),
            "mae_promedio":      round(grupo["mae"].mean(),     4),
            "r2_promedio":       round(grupo["r2"].mean(),      4),
            # Programas prioritarios
            "recall_adm_emp":    _get_metrica(grupo, "ADMINISTRACION_DE_EMPRESAS",                    "recall"),
            "recall_gest_com":   _get_metrica(grupo, "TECNOLOGIA_EN_GESTION_COMERCIAL_Y_DE_MERCADOS", "recall"),
            "recall_gest_emp":   _get_metrica(grupo, "TECNOLOGIA_EN_GESTION_EMPRESARIAL",             "recall"),
            "f1_adm_emp":        _get_metrica(grupo, "ADMINISTRACION_DE_EMPRESAS",                    "f1"),
            "f1_gest_com":       _get_metrica(grupo, "TECNOLOGIA_EN_GESTION_COMERCIAL_Y_DE_MERCADOS", "f1"),
            "f1_gest_emp":       _get_metrica(grupo, "TECNOLOGIA_EN_GESTION_EMPRESARIAL",             "f1"),
        }

    # Selección de la mejor configuración basada en criterios de tesis
    base = resumen.get("A_original", {})
    mejor_id  = "A_original"
    mejor_score = 0.0

    for exp_id, stats in resumen.items():
        if exp_id == "A_original":
            continue
        # Puntuación compuesta: mejora en F1 y recall, penalización por daño en MAE y R²
        delta_f1     = stats["f1_promedio"]    - base.get("f1_promedio",    0)
        delta_recall = stats["recall_promedio"] - base.get("recall_promedio", 0)
        delta_mae    = stats["mae_promedio"]    - base.get("mae_promedio",    0)
        delta_r2     = stats["r2_promedio"]     - base.get("r2_promedio",     0)

        score = (delta_f1 * 0.40 + delta_recall * 0.40
                 - max(0, delta_mae) * 0.10
                 - max(0, -delta_r2) * 0.10)

        if score > mejor_score:
            mejor_score = score
            mejor_id    = exp_id

    return {
        "resumen_por_experimento": resumen,
        "mejor_config_id":   mejor_id,
        "mejor_config_nombre": resumen.get(mejor_id, {}).get("nombre", "original"),
        "mejor_score":       round(mejor_score, 5),
        "base_f1":           base.get("f1_promedio", 0),
        "base_recall":       base.get("recall_promedio", 0),
        "base_mae":          base.get("mae_promedio", 0),
    }


def _get_metrica(df: pd.DataFrame, programa: str, metrica: str) -> float:
    fila = df[df["programa"] == programa]
    if fila.empty:
        return float("nan")
    return round(float(fila[metrica].iloc[0]), 4)


# ══════════════════════════════════════════════════════════════════════════════
# GENERACIÓN DEL DOCUMENTO MARKDOWN
# ══════════════════════════════════════════════════════════════════════════════

def generar_markdown(df: pd.DataFrame, analisis: dict) -> str:
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M")
    res  = analisis["resumen_por_experimento"]
    best = analisis["mejor_config_id"]
    base = res.get("A_original", {})

    # Tabla comparativa de promedios
    tabla_promedios = "| Configuración | AUC | Recall | F1 | MAE | R² |\n"
    tabla_promedios += "|---|---|---|---|---|---|\n"
    for eid, stats in res.items():
        marca = " ✅ **GANADORA**" if eid == best else ""
        tabla_promedios += (
            f"| {stats['nombre']}{marca} | {stats['auc_promedio']} | "
            f"{stats['recall_promedio']} | {stats['f1_promedio']} | "
            f"{stats['mae_promedio']} | {stats['r2_promedio']} |\n"
        )

    # Tabla por programas prioritarios
    tabla_prio = "| Programa | Métrica | " + " | ".join(
        f"{r['nombre'][:20]}" for r in res.values()
    ) + " |\n"
    tabla_prio += "|---|---|" + "|".join(["---"] * len(res)) + "|\n"

    for prog, label in [
        ("ADMINISTRACION_DE_EMPRESAS",                    "Adm. Empresas"),
        ("TECNOLOGIA_EN_GESTION_COMERCIAL_Y_DE_MERCADOS", "Gest. Comercial"),
        ("TECNOLOGIA_EN_GESTION_EMPRESARIAL",             "Gest. Empresarial"),
    ]:
        for metrica in ["recall", "f1"]:
            row = f"| {label} | {metrica.upper()} |"
            for eid, stats in res.items():
                key = f"{metrica}_{'adm_emp' if 'EMPRESA' in prog and 'GESTION' not in prog else 'gest_com' if 'COMERCIAL' in prog else 'gest_emp'}"
                val = stats.get(key, float("nan"))
                marca = " ✅" if eid == best else ""
                row += f" {val}{marca} |"
            tabla_prio += row + "\n"

    # Delta con respecto al original
    mejor = res.get(best, {})
    delta_f1     = round(mejor.get("f1_promedio", 0)     - base.get("f1_promedio", 0),     4)
    delta_recall = round(mejor.get("recall_promedio", 0) - base.get("recall_promedio", 0), 4)
    delta_mae    = round(mejor.get("mae_promedio", 0)    - base.get("mae_promedio", 0),    4)
    delta_r2     = round(mejor.get("r2_promedio", 0)     - base.get("r2_promedio", 0),     4)

    signo_f1     = "+" if delta_f1     >= 0 else ""
    signo_recall = "+" if delta_recall >= 0 else ""
    signo_mae    = "+" if delta_mae    >= 0 else ""
    signo_r2     = "+" if delta_r2     >= 0 else ""

    mejor_es_original = best == "A_original"
    recomendacion = (
        "**No se recomienda modificar la configuración.** Ninguna optimización superó "
        "al modelo original de forma consistente. Se mantiene la configuración original."
        if mejor_es_original else
        f"**Se recomienda aplicar la configuración `{best}` ({mejor.get('nombre','')})** "
        f"como configuración de producción. Esta configuración mejoró el F1 promedio en "
        f"{signo_f1}{delta_f1} y el recall promedio en {signo_recall}{delta_recall}, "
        f"con un cambio en MAE de {signo_mae}{delta_mae} y en R² de {signo_r2}{delta_r2}."
    )

    md = f"""# Optimización RNN — RNN-ALERT · UNIMAYOR

**Generado automáticamente:** {ts}
**Script:** `experimentos_rnn.py`

---

## 1. Objetivo

Mejorar el desempeño del modelo RNN multimodal multitarea (RNN-ALERT) en la tarea de clasificación de riesgo de deserción estudiantil, especialmente en los programas con menor F1-score y recall, sin deteriorar la predicción de nota del siguiente semestre.

## 2. Problema identificado

Análisis del modelo original (`metricas_por_programa.csv`) reveló:

- **ADMINISTRACION_DE_EMPRESAS**: Recall = 0.50, AUC = 0.83 (peor programa)
- **TECNOLOGIA_EN_GESTION_COMERCIAL_Y_DE_MERCADOS**: Recall = 0.559, AUC = 0.718
- Causas identificadas:
  1. Split sin estratificación (`train_test_split` sin `stratify=`): en programas pequeños el azar puede generar test sin desertores suficientes.
  2. Sin `loss_weights`: el MSE de nota domina sobre la binary_crossentropy de deserción.
  3. Sin `sample_weight`/`class_weight`: el desbalance ~75/25 no está compensado en train.

## 3. Archivos modificados

| Archivo | Cambio |
|---|---|
| `modelos/modelo_rnn.py` | Agregados parámetros `stratify_split`, `loss_weight_des`, `usar_sample_weight` |
| `modelos/modelo_rnn.py.bak` | Respaldo del original (sin modificar) |
| `modelos/entrenamiento_masivo.py.bak` | Respaldo del original |

## 4. Configuraciones probadas

| ID | Nombre | stratify | loss_weight_des | sample_weight |
|---|---|---|---|---|
| A | RNN original | False | 1.0 | False |
| B | RNN + stratify | True | 1.0 | False |
| C | RNN + stratify + lw1.5 | True | 1.5 | False |
| D | RNN + stratify + lw2.0 | True | 2.0 | False |
| E | RNN + stratify + lw2.5 | True | 2.5 | False |
| F | RNN + stratify + lw2.0 + sample_weight | True | 2.0 | True |

## 5. Tabla comparativa — promedios globales

{tabla_promedios}

## 6. Resultados en programas prioritarios

{tabla_prio}

## 7. Mejor configuración

**Ganadora:** `{best}` — {mejor.get("nombre", "")}

| Métrica | Original | Mejor | Δ |
|---|---|---|---|
| F1 promedio | {base.get("f1_promedio", "N/A")} | {mejor.get("f1_promedio", "N/A")} | {signo_f1}{delta_f1} |
| Recall promedio | {base.get("recall_promedio", "N/A")} | {mejor.get("recall_promedio", "N/A")} | {signo_recall}{delta_recall} |
| MAE promedio | {base.get("mae_promedio", "N/A")} | {mejor.get("mae_promedio", "N/A")} | {signo_mae}{delta_mae} |
| R² promedio | {base.get("r2_promedio", "N/A")} | {mejor.get("r2_promedio", "N/A")} | {signo_r2}{delta_r2} |

## 8. Justificación académica

### Split estratificado
Kohavi (1995) demuestra que la estratificación reduce la varianza del estimador de error de generalización en datasets pequeños con clases desbalanceadas. Para programas con N < 100, el azar puede concentrar desertores en train o en test, generando métricas no representativas.

### loss_weights
En modelos multitarea (Caruana, 1997; Ruder, 2017), la escala de las pérdidas de cada salida determina su influencia en el gradiente compartido. La binary_crossentropy (valores en [0,1]) suele ser superada por el MSE cuando los valores de nota están en rangos más amplios. Ajustar `loss_weights` es la técnica estándar para calibrar este balance (Cipolla et al., 2018 — *Uncertainty Weighting*).

### sample_weight / class_weight
Con una proporción de ~75% no desertores y ~25% desertores, el modelo puede alcanzar alta exactitud simplemente prediciendo siempre la clase mayoritaria. Los `sample_weight` corrigen este incentivo penalizando proporcionalmente más los errores en la clase minoritaria (He & Garcia, 2009 — *Learning from Imbalanced Data*).

## 9. Recomendación final

{recomendacion}

---
*Documento generado por `experimentos_rnn.py` · RNN-ALERT · UNIMAYOR 2026*
"""
    return md


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*70)
    print("  RNN-ALERT — Experimentos de Optimización Controlada")
    print("="*70)

    programas = obtener_programas_disponibles(OUTPUTS_DIR)
    if not programas:
        print("[ERROR] No hay programas disponibles. Ejecuta primero el Pipeline.")
        sys.exit(1)

    print(f"\nProgramas detectados ({len(programas)}): {', '.join(programas)}\n")

    todos_resultados = []
    t_total = time.time()

    for cfg in CONFIGS:
        resultados = ejecutar_experimento(cfg, programas)
        todos_resultados.extend(resultados)

    df = pd.DataFrame(todos_resultados)

    # Guardar CSV completo
    ruta_csv = OUTPUTS_DIR / "resultados" / "experimentos_comparativos.csv"
    df.to_csv(ruta_csv, index=False, encoding="utf-8")
    print(f"\n✅ Resultados guardados en: {ruta_csv}")

    # Análisis y decisión
    analisis = analizar_resultados(df)

    # Guardar análisis JSON
    ruta_json = OUTPUTS_DIR / "resultados" / "experimentos_analisis.json"
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(analisis, f, ensure_ascii=False, indent=2, default=str)
    print(f"✅ Análisis guardado en: {ruta_json}")

    # Generar y guardar Markdown
    md = generar_markdown(df, analisis)
    ruta_md = DOCS_DIR / "optimizacion_rnn_resultados.md"
    with open(ruta_md, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"✅ Documento guardado en: {ruta_md}")

    # Resumen en consola
    print(f"\n{'='*70}")
    print("  RESUMEN FINAL")
    print(f"{'='*70}")
    res = analisis["resumen_por_experimento"]
    for eid, stats in res.items():
        marca = " ← GANADORA" if eid == analisis["mejor_config_id"] else ""
        print(
            f"  {stats['nombre']:<45} "
            f"F1={stats['f1_promedio']:.4f}  "
            f"Recall={stats['recall_promedio']:.4f}  "
            f"MAE={stats['mae_promedio']:.4f}{marca}"
        )

    print(f"\n  Duración total: {round(time.time()-t_total, 1)}s")
    print(f"  Mejor configuración: {analisis['mejor_config_id']} — {analisis['mejor_config_nombre']}")
    print(f"\n  Revisa docs/optimizacion_rnn_resultados.md para el informe completo.")


if __name__ == "__main__":
    main()
