"""
pipeline/paso5_propedeutico.py
===============================
Paso 5 — Unificación de Trayectorias Propedéuticas
RNN-ALERT · UNIMAYOR · 2026

CONTEXTO INSTITUCIONAL
─────────────────────────────────────────────────────────────────────────────
UNIMAYOR tiene dos cadenas propedéuticas:

  Cadena 1:
    Tecnología en Gestión Empresarial  (sem 1-6, plan propio)
    → Administración de Empresas       (sem 7-10, plan propio)

  Cadena 2:
    Tecnología en Gestión Financiera   (sem 1-6, plan propio)
    → Administración Financiera        (sem 7-10, plan propio)

Los planes de estudio del archivo de referencia YA reflejan esta estructura:
  - TGE: semestres 1–6
  - AE:  semestres 7–10
  - TGF: semestres 1–6
  - AF:  semestres 7–10

PROBLEMA QUE RESUELVE ESTE PASO
─────────────────────────────────────────────────────────────────────────────
Cuando el mismo ID_EST aparece en TGE y en AE, el pipeline anterior los
trataba como dos registros/programas separados. Esto generaba:
  • Historiales fragmentados (solo 6 semestres o solo 4 semestres)
  • DESERTOR calculado de forma independiente y posiblemente errónea
  • Duplicados en los datasets por programa
  • Modelos entrenados sin la trayectoria completa

ESTRATEGIA DE UNIFICACIÓN
─────────────────────────────────────────────────────────────────────────────
Caso A — Estudiante con TGE Y AE (propedéutico completo):
  → Se construye una sola trayectoria bajo PROGRAMA = "ADMINISTRACION DE EMPRESAS"
  → Semestres 1-6 vienen de TGE, semestres 7-10 vienen de AE
  → Las materias ya tienen el semestre correcto desde el plan de estudios

Caso B — Estudiante SOLO en TGE (tecnología sin continuar):
  → Se conserva bajo "TECNOLOGIA EN GESTION EMPRESARIAL" para análisis
  → Se exporta como estudiante_solo_tecnologia para revisión
  → NO se mueve a AE (no completó la cadena)

Caso C — Estudiante SOLO en AE (ingresó directo a profesional):
  → Se conserva bajo "ADMINISTRACION DE EMPRESAS" normalmente

Mismo patrón para la cadena TGF → AF.

TRAZABILIDAD
─────────────────────────────────────────────────────────────────────────────
Se generan columnas de auditoría en cada registro académico:
  - PROGRAMA_ORIGINAL:   nombre del programa en el dato fuente
  - PROGRAMA_UNIFICADO:  nombre final del programa tras unificación
  - ES_PROPEDEUTICO:     1 si el registro viene de una cadena propedéutica
  - ORIGEN_TRAYECTORIA:  "tecnologia", "profesional", "tecnologia+profesional"

Entradas:
  df_acad_con_semestre : salida del Paso 4 (DF_ACADEMICO_2_RELLENADO.csv)
  df_demo_rellenado    : salida del Paso 3 (DF_DEMOGRAFICOS_2_RELLENADO.csv)
  ruta_auditoria       : carpeta donde guardar archivos de auditoría

Salidas:
  df_acad_unificado    → DF_ACADEMICO_3_PROPEDEUTICO.csv
  df_demo_unificado    → DF_DEMOGRAFICOS_3_PROPEDEUTICO.csv
  archivos de auditoría en ruta_auditoria/
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Mapeo de cadenas propedéuticas ─────────────────────────────────────────────
# Clave: nombre normalizado de la tecnología
# Valor: nombre normalizado de la carrera profesional
CADENAS_PROPEDEUTICAS: Dict[str, str] = {
    "TECNOLOGIA EN GESTION EMPRESARIAL": "ADMINISTRACION DE EMPRESAS",
    "TECNOLOGIA EN GESTION FINANCIERA":  "ADMINISTRACION FINANCIERA",
}

# Rango de semestres por tipo en la cadena
SEM_TECNOLOGIA  = (1, 6)   # semestres 1-6 en la tecnología
SEM_PROFESIONAL = (7, 10)  # semestres 7-10 en la profesional


def _normalizar_programa(nombre: str) -> str:
    """Normaliza nombre de programa: mayúsculas y sin espacios extra."""
    if pd.isna(nombre):
        return ""
    return str(nombre).strip().upper()


def _detectar_propedeuticos(
    df_acad: pd.DataFrame,
) -> Tuple[Dict[str, set], Dict[str, set], Dict[str, set]]:
    """
    Detecta qué estudiantes están en cada posición de la cadena propedéutica.

    Returns:
        ids_ambos : {prog_prof: set de IDs con tecnología + profesional}
        ids_solo_tec: {prog_tec: set de IDs solo en tecnología}
        ids_solo_prof: {prog_prof: set de IDs solo en profesional}
    """
    df_acad = df_acad.copy()
    df_acad["_PROG_N"] = df_acad["PROGRAMA"].apply(_normalizar_programa)

    ids_ambos:     Dict[str, set] = {}
    ids_solo_tec:  Dict[str, set] = {}
    ids_solo_prof: Dict[str, set] = {}

    for prog_tec, prog_prof in CADENAS_PROPEDEUTICAS.items():
        ids_tec  = set(df_acad[df_acad["_PROG_N"] == prog_tec ]["ID_EST"].unique())
        ids_prof = set(df_acad[df_acad["_PROG_N"] == prog_prof]["ID_EST"].unique())

        ambos    = ids_tec & ids_prof
        solo_tec = ids_tec - ids_prof
        solo_prof= ids_prof - ids_tec

        ids_ambos[prog_prof]     = ambos
        ids_solo_tec[prog_tec]   = solo_tec
        ids_solo_prof[prog_prof] = solo_prof

        logger.info(
            f"[Paso5] {prog_tec} → {prog_prof}: "
            f"ambos={len(ambos)}, solo_tec={len(solo_tec)}, solo_prof={len(solo_prof)}"
        )

    return ids_ambos, ids_solo_tec, ids_solo_prof


def ejecutar(
    df_acad:  pd.DataFrame,
    df_demo:  pd.DataFrame,
    ruta_auditoria: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Unifica trayectorias propedéuticas en los datasets académico y demográfico.

    Args:
        df_acad          : DataFrame con columnas PROGRAMA, SEMESTRE, etc.
                           (salida del Paso 4 — DF_ACADEMICO_2_RELLENADO)
        df_demo          : DataFrame demográfico (salida Paso 3)
        ruta_auditoria   : Directorio donde guardar CSVs de auditoría

    Returns:
        (df_acad_unificado, df_demo_unificado, stats)
    """
    stats: dict = {"paso": "5_propedeutico"}
    logger.info("[Paso5] Iniciando unificación propedéutica")

    if ruta_auditoria is not None:
        Path(ruta_auditoria).mkdir(parents=True, exist_ok=True)

    df_a = df_acad.copy()
    df_d = df_demo.copy()

    # Normalizar nombres de programa para comparación interna
    df_a["_PROG_N"] = df_a["PROGRAMA"].apply(_normalizar_programa)
    df_d["_PROG_N"] = df_d["PROGRAMA"].apply(_normalizar_programa)

    # Añadir columnas de trazabilidad
    df_a["PROGRAMA_ORIGINAL"]  = df_a["PROGRAMA"]
    df_a["PROGRAMA_UNIFICADO"] = df_a["PROGRAMA"]
    df_a["ES_PROPEDEUTICO"]    = 0
    df_a["ORIGEN_TRAYECTORIA"] = "independiente"

    df_d["PROGRAMA_ORIGINAL"]  = df_d["PROGRAMA"]
    df_d["PROGRAMA_UNIFICADO"] = df_d["PROGRAMA"]
    df_d["ES_PROPEDEUTICO"]    = 0
    df_d["ORIGEN_TRAYECTORIA"] = "independiente"

    ids_ambos, ids_solo_tec, ids_solo_prof = _detectar_propedeuticos(df_a)

    # Listas para auditoría
    audit_propedeuticos   = []
    audit_solo_tec        = []
    audit_solo_prof       = []
    audit_duplicados      = []
    total_unificados      = 0

    # ── Procesar cada cadena propedéutica ────────────────────────────────────
    for prog_tec, prog_prof in CADENAS_PROPEDEUTICAS.items():

        ambos     = ids_ambos.get(prog_prof, set())
        solo_tec  = ids_solo_tec.get(prog_tec, set())
        solo_p    = ids_solo_prof.get(prog_prof, set())

        # ── Caso A: Estudiante con AMBOS programas (propedéutico completo) ──
        for id_est in ambos:
            # Registros académicos
            mask_tec  = (df_a["ID_EST"] == id_est) & (df_a["_PROG_N"] == prog_tec)
            mask_prof = (df_a["ID_EST"] == id_est) & (df_a["_PROG_N"] == prog_prof)

            # Verificar que no haya solapamiento de semestres
            sems_tec  = set(df_a.loc[mask_tec,  "SEMESTRE"].dropna().astype(int).tolist())
            sems_prof = set(df_a.loc[mask_prof, "SEMESTRE"].dropna().astype(int).tolist())
            solapados = sems_tec & sems_prof

            if solapados:
                logger.warning(
                    f"[Paso5] ID {id_est}: semestres solapados {solapados} "
                    f"entre {prog_tec} y {prog_prof}. Se conserva el de la profesional."
                )
                # Eliminar los solapados del registro de tecnología
                df_a = df_a[~(mask_tec & df_a["SEMESTRE"].isin(solapados))]
                audit_duplicados.append({
                    "ID_EST": id_est,
                    "prog_tec": prog_tec, "prog_prof": prog_prof,
                    "semestres_solapados": str(sorted(solapados)),
                    "resolucion": "eliminado de tecnologia",
                })
                # Recalcular mask tras modificación
                mask_tec = (df_a["ID_EST"] == id_est) & (df_a["_PROG_N"] == prog_tec)

            # Reasignar PROGRAMA a la carrera profesional en registros de tecnología
            df_a.loc[mask_tec,  "PROGRAMA"]          = prog_prof
            df_a.loc[mask_tec,  "PROGRAMA_UNIFICADO"] = prog_prof
            df_a.loc[mask_tec,  "ES_PROPEDEUTICO"]    = 1
            df_a.loc[mask_tec,  "ORIGEN_TRAYECTORIA"] = "tecnologia+profesional"

            df_a.loc[mask_prof, "PROGRAMA_UNIFICADO"] = prog_prof
            df_a.loc[mask_prof, "ES_PROPEDEUTICO"]    = 1
            df_a.loc[mask_prof, "ORIGEN_TRAYECTORIA"] = "tecnologia+profesional"

            # Reasignar PROGRAMA en demográfico (puede haber dos filas: tec + prof)
            mask_d_tec  = (df_d["ID_EST"] == id_est) & (df_d["_PROG_N"] == prog_tec)
            mask_d_prof = (df_d["ID_EST"] == id_est) & (df_d["_PROG_N"] == prog_prof)

            # Consolidar: conservar solo el registro demográfico de la profesional
            # Si solo existe el de tecnología, reasignar a profesional
            if mask_d_prof.sum() > 0:
                # Hay registro en profesional: eliminar el de tecnología del demo
                df_d = df_d[~mask_d_tec].copy()
            else:
                # Solo hay registro en tecnología: reasignarlo a la profesional
                df_d.loc[mask_d_tec, "PROGRAMA"]          = prog_prof
                df_d.loc[mask_d_tec, "PROGRAMA_UNIFICADO"] = prog_prof

            # Actualizar trazabilidad en demo
            mask_d_final = (df_d["ID_EST"] == id_est) & \
                           df_d["_PROG_N"].str.startswith(prog_prof[:10])
            df_d.loc[mask_d_final, "ES_PROPEDEUTICO"]    = 1
            df_d.loc[mask_d_final, "ORIGEN_TRAYECTORIA"] = "tecnologia+profesional"

            audit_propedeuticos.append({
                "ID_EST": id_est,
                "programa_tecnologia": prog_tec,
                "programa_profesional": prog_prof,
                "semestres_tec": sorted(sems_tec - solapados),
                "semestres_prof": sorted(sems_prof),
                "tipo": "propedeutico_completo",
            })
            total_unificados += 1

        # ── Caso B: Solo tecnología ─────────────────────────────────────────
        for id_est in solo_tec:
            mask_tec = (df_a["ID_EST"] == id_est) & (df_a["_PROG_N"] == prog_tec)
            df_a.loc[mask_tec, "ES_PROPEDEUTICO"]    = 1
            df_a.loc[mask_tec, "ORIGEN_TRAYECTORIA"] = "solo_tecnologia"
            audit_solo_tec.append({
                "ID_EST": id_est,
                "programa_tecnologia": prog_tec,
                "observacion": "Cursó tecnología pero no continuó carrera profesional",
            })

        # ── Caso C: Solo profesional ────────────────────────────────────────
        for id_est in solo_p:
            mask_prof = (df_a["ID_EST"] == id_est) & (df_a["_PROG_N"] == prog_prof)
            df_a.loc[mask_prof, "ORIGEN_TRAYECTORIA"] = "solo_profesional"
            audit_solo_prof.append({
                "ID_EST": id_est,
                "programa_profesional": prog_prof,
                "observacion": "Ingresó directamente a carrera profesional",
            })

    # ── Recalcular _PROG_N tras las reasignaciones ────────────────────────────
    df_a["_PROG_N"] = df_a["PROGRAMA"].apply(_normalizar_programa)
    df_d["_PROG_N"] = df_d["PROGRAMA"].apply(_normalizar_programa)

    # ── Estadísticas finales ──────────────────────────────────────────────────
    stats["total_propedeuticos_unificados"] = total_unificados
    stats["solo_tecnologia"] = len(audit_solo_tec)
    stats["solo_profesional"] = len(audit_solo_prof)
    stats["registros_duplicados_resueltos"] = len(audit_duplicados)
    stats["total_filas_acad_final"] = len(df_a)
    stats["total_filas_demo_final"] = len(df_d)

    logger.info(
        f"[Paso5] Completado: {total_unificados} propedéuticos unificados | "
        f"{len(audit_solo_tec)} solo tecnología | "
        f"{len(audit_solo_prof)} solo profesional"
    )

    # ── Exportar auditoría ─────────────────────────────────────────────────────
    if ruta_auditoria is not None:
        p = Path(ruta_auditoria)

        if audit_propedeuticos:
            pd.DataFrame(audit_propedeuticos).to_csv(
                p / "estudiantes_tecnologia_y_profesional.csv", index=False)
        if audit_solo_tec:
            pd.DataFrame(audit_solo_tec).to_csv(
                p / "estudiantes_solo_tecnologia.csv", index=False)
        if audit_solo_prof:
            pd.DataFrame(audit_solo_prof).to_csv(
                p / "estudiantes_solo_profesional.csv", index=False)
        if audit_duplicados:
            pd.DataFrame(audit_duplicados).to_csv(
                p / "registros_duplicados_resueltos.csv", index=False)

        # Resumen de unificación
        resumen = [{
            "cadena": f"{tec} → {prof}",
            "propedeuticos_completos": len(ids_ambos.get(prof, set())),
            "solo_tecnologia":         len(ids_solo_tec.get(tec, set())),
            "solo_profesional":        len(ids_solo_prof.get(prof, set())),
        } for tec, prof in CADENAS_PROPEDEUTICAS.items()]
        pd.DataFrame(resumen).to_csv(
            p / "resumen_unificacion_propedeutica.csv", index=False)

        logger.info(f"[Paso5] Archivos de auditoría guardados en: {p}")

    # Eliminar columna auxiliar interna
    df_a = df_a.drop(columns=["_PROG_N"], errors="ignore")
    df_d = df_d.drop(columns=["_PROG_N"], errors="ignore")

    return df_a, df_d, stats
