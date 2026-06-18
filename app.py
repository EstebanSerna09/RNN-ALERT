"""
RNN-ALERT v4 — Sistema de Alertas Tempranas · Deserción Estudiantil
Institución Universitaria Colegio Mayor del Cauca (UNIMAYOR) · 2026

Modelos: RNN Multitarea · Modelos Machine Learning por Programa
Pipeline: 10 pasos con unificación propedéutica y clasificación de estudiantes
"""
from __future__ import annotations
import io, json, logging, pickle, shutil, sys, time, zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ── Rutas ──────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
REFERENCIA_DIR = BASE_DIR / "data" / "referencia"
OUTPUTS_DIR    = BASE_DIR / "outputs"
INTER_DIR      = OUTPUTS_DIR / "intermedios"
MOD_DIR        = OUTPUTS_DIR / "modelos"
AUDIT_DIR      = OUTPUTS_DIR / "auditoria"
UPLOADS_DIR    = BASE_DIR / "uploads"
ASSETS_DIR     = BASE_DIR / "assets"
for d in [REFERENCIA_DIR, INTER_DIR, MOD_DIR, AUDIT_DIR, UPLOADS_DIR, ASSETS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RUTA_RURAL  = REFERENCIA_DIR / "DATASET_MUNICIPIOS_RURALES_COLOMBIA.csv"
RUTA_PLANES = REFERENCIA_DIR / "DF_PLANES_ESTUDIO_UNIMAYOR_FINAL.csv"

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("rnn_alert")

sys.path.insert(0, str(BASE_DIR))
from pipeline import paso1, paso2, paso3, paso4, paso5, paso6, paso7
from pipeline.paso5_propedeutico import ejecutar as paso5_prop
from pipeline.paso7_clasificar import ejecutar as paso7_clas
from utils.validators import cargar_archivo, validar_no_vacio
from utils.zipper import crear_zip
from utils.session import init_session, registrar_modelo, obtener_modelos, pipeline_listo
from utils.theme import inyectar_css, sidebar_logo, header, kpi, badge, arch_card, arch_arrow
from utils.persistencia import boton_descarga_df, registrar_dataframes_en_session
from utils.model_loader import auto_cargar_modelos_en_session
# NOTA: modelos_ml.py se mantiene en disco pero ya no se usa para entrenar modelos individuales.
# Solo se importa preparar_features_estaticas para predicción masiva de modelos ML globales.
try:
    from modelos.modelos_ml import (
        preparar_features_estaticas, FEATURES_ACAD, FEATURES_DEMO, FEATURES_TODOS
    )
except ImportError:
    preparar_features_estaticas = None
    FEATURES_ACAD = FEATURES_DEMO = FEATURES_TODOS = []

st.set_page_config(
    page_title="RNN-ALERT · UNIMAYOR",
    page_icon=ASSETS_DIR / "logo_unimayor.webp",
    layout="wide",
    initial_sidebar_state="expanded"
)
init_session()
inyectar_css()

# ── Ajuste visual: toolbar superior transparente ───────────────────────────────
# Nota: se usan selectores estables de Streamlit (data-testid) y no clases
# autogeneradas "st-emotion-cache", porque esas clases cambian entre ejecuciones.
st.markdown("""
<style>
/* Header superior de Streamlit transparente */
header[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
    backdrop-filter: none !important;
}

/* Toolbar superior transparente */
[data-testid="stToolbar"] {
    background: transparent !important;
    box-shadow: none !important;
    backdrop-filter: none !important;
}

/* Contenedores internos del toolbar transparentes */
[data-testid="stToolbar"] > div,
[data-testid="stToolbar"] div {
    background: transparent !important;
    box-shadow: none !important;
}

/* Mantener visible el botón Deploy */
[data-testid="stAppDeployButton"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}

/* Mantener visible el menú de tres puntos */
#MainMenu,
[data-testid="stMainMenu"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}

/* Botón Deploy y botón de tres puntos sin fondo */
[data-testid="stBaseButton-header"],
[data-testid="stMainMenuButton"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Color del texto Deploy */
[data-testid="stAppDeployButton"] span {
    color: #B8C1CC !important;
}

/* Color del ícono de tres puntos */
[data-testid="stMainMenu"] svg,
#MainMenu svg {
    color: #B8C1CC !important;
    fill: #B8C1CC !important;
}

/* Quitar decoración superior de Streamlit si aparece */
[data-testid="stDecoration"] {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ── Persistencia automática al iniciar ────────────────────────────────────────
try:
    _n_m = auto_cargar_modelos_en_session(OUTPUTS_DIR)
    if _n_m > 0: logger.info(f"[Inicio] {_n_m} modelo(s) recuperados desde disco.")
except Exception as _e: logger.warning(f"[Inicio] Auto-carga: {_e}")
try:
    _n_d = registrar_dataframes_en_session(BASE_DIR)
    if _n_d > 0: logger.info(f"[Inicio] {_n_d} DataFrame(s) restaurados.")
except Exception as _e: logger.warning(f"[Inicio] Restauración DF: {_e}")

plt.rcParams.update({
    "figure.facecolor":"#172A45","axes.facecolor":"#0A192F",
    "axes.edgecolor":"#1E3A5F","axes.labelcolor":"#B8C1CC",
    "xtick.color":"#B8C1CC","ytick.color":"#B8C1CC",
    "text.color":"#B8C1CC","grid.color":"#1E3A5F","grid.alpha":0.5,"axes.grid":True,
})

# ── Helpers generales ──────────────────────────────────────────────────────────
def _save_upload(f) -> Path:
    dest = UPLOADS_DIR / f.name
    with open(dest,"wb") as o: o.write(f.getbuffer())
    return dest

def _log(container, msgs, texto, tipo="ok"):
    icons={"ok":"✔","err":"✘","info":"ℹ"}
    msgs.append(f'<span class="log-{tipo}">{icons.get(tipo,"•")}&nbsp; {texto}</span>')
    container.markdown(f'<div class="log-box">{"<br>".join(msgs)}</div>', unsafe_allow_html=True)
    return msgs

def _ref_ok(p): return p.exists() and p.stat().st_size > 100

def _nombre_completo(n: str) -> str: return str(n).strip()

def _nombre_archivo_seguro(nombre: str) -> str:
    return (str(nombre).strip()
            .replace("/","_").replace(":","_").replace(" ","_")
            .replace("(","").replace(")","").replace("á","a")
            .replace("é","e").replace("í","i").replace("ó","o")
            .replace("ú","u").replace("ñ","n"))

def _fmt_met(valor, decimales: int = 4) -> str:
    """Formatea una métrica numérica de forma segura. Retorna 'N/D' si no es válida."""
    if valor is None:
        return "N/D"
    try:
        v = float(valor)
        if v != v:  # NaN check
            return "N/D"
        return f"{v:.{decimales}f}"
    except (TypeError, ValueError):
        return str(valor) if valor != "" else "N/D"

def _plot_cm(cm_list):
    cm = np.array(cm_list)
    fig, ax = plt.subplots(figsize=(4,3.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_xticklabels(["No Des.","Desertor"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["No Des.","Desertor"])
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=15,
                    fontweight="bold",color="white" if cm[i,j]>cm.max()*0.4 else "#B8C1CC")
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    return fig

def _plot_roc(fpr,tpr,auc_val):
    fig,ax=plt.subplots(figsize=(4.5,3.8))
    ax.plot(fpr,tpr,color="#0074D9",lw=2.5,label=f"AUC={auc_val:.4f}")
    ax.plot([0,1],[0,1],color="#1E3A5F",lw=1.5,ls="--")
    ax.fill_between(fpr,tpr,alpha=0.12,color="#0074D9")
    ax.set_xlabel("FPR"); ax.set_ylabel("Recall"); ax.legend(loc="lower right",fontsize=9)
    return fig

def _mostrar_resultados_rnn(nombre, metricas):
    m = metricas
    st.markdown(f'<div class="seccion-titulo"> {_nombre_completo(nombre)}</div>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("ROC-AUC",  f"{m.get('roc_auc',0):.4f}")
    c2.metric("Recall",   f"{m.get('recall_opt',0):.4f}")
    c3.metric("F1-Score", f"{m.get('f1_opt',0):.4f}")
    c4.metric("Precision",f"{m.get('precision_opt',0):.4f}")
    c5.metric("PR-AUC",   f"{m.get('pr_auc',0):.4f}")
    if "mae" in m:
        ca,cb = st.columns(2)
        ca.metric("MAE (nota)",f"{m.get('mae',0):.4f}")
        cb.metric("R² (nota)", f"{m.get('r2',0):.4f}")
    col_cm,col_roc = st.columns(2)
    with col_cm:
        if "cm" in m:
            st.markdown("**Matriz de Confusión**")
            st.pyplot(_plot_cm(m["cm"]), use_container_width=False)
    with col_roc:
        if "fpr" in m:
            st.markdown("**Curva ROC**")
            st.pyplot(_plot_roc(np.array(m["fpr"]),np.array(m["tpr"]),m.get("roc_auc",0)), use_container_width=False)
    with st.expander("Ver reporte completo"):
        st.code(m.get("reporte",""))
    datos_dl = {k:v for k,v in m.items() if k not in ["fpr","tpr","prec_curve","rec_curve","y_proba","y_true","cm","reporte","y_pred_nota","y_true_nota"]}
    st.download_button(f" Métricas {_nombre_completo(nombre)}",
        data=pd.DataFrame([datos_dl]).to_csv(index=False).encode(),
        file_name=f"metricas_{_nombre_archivo_seguro(nombre)}.csv", mime="text/csv")

def _formatear_semestres(sems_raw: list) -> str:
    sems_int = []
    for s in sorted(sems_raw):
        try:
            v = float(s); sems_int.append(str(int(v)) if v == int(v) else str(v))
        except Exception:
            sems_int.append(str(s))
    if not sems_int: return "—"
    if len(sems_int)==1: return sems_int[0]
    return ", ".join(sems_int[:-1]) + " y " + sems_int[-1]


def _programa_desde_modelo(nombre_modelo: str, info: dict) -> str:
    """Obtiene el programa asociado a un modelo, evitando mostrar RNN como Global."""
    metricas = info.get("metricas", {}) if isinstance(info, dict) else {}
    candidatos = [
        info.get("programa") if isinstance(info, dict) else None,
        (info.get("datos_rnn") or {}).get("programa") if isinstance(info, dict) else None,
        metricas.get("programa") if isinstance(metricas, dict) else None,
    ]
    if "(" in nombre_modelo and ")" in nombre_modelo:
        try:
            candidatos.append(nombre_modelo.split("(", 1)[1].rsplit(")", 1)[0].strip())
        except Exception:
            pass
    for c in candidatos:
        if c is not None and str(c).strip() and str(c).strip().lower() not in ("global", "none", "nan"):
            return str(c).replace("_", " ").title()
    return "Global"


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE ARTEFACTOS RNN DESDE DISCO (fallback predicción individual)
# ══════════════════════════════════════════════════════════════════════════════

def _extraer_safe_prog_de_nombre(nombre_modelo: str) -> str:
    import re
    m = re.search(r'\((.+)\)', nombre_modelo)
    if m:
        return _nombre_archivo_seguro(m.group(1).strip()).upper()
    return _nombre_archivo_seguro(nombre_modelo).upper()

def _cargar_artefactos_rnn_para_prediccion(nombre_modelo: str) -> Optional[dict]:
    try:
        from tensorflow.keras.models import load_model as _tf_load
    except ImportError:
        st.error("❌ TensorFlow no está instalado."); return None

    safe_prog = _extraer_safe_prog_de_nombre(nombre_modelo)
    dir_prog  = OUTPUTS_DIR / "modelos" / "por_programa" / safe_prog

    if not dir_prog.exists():
        st.error(f"❌ Sin carpeta de artefactos para **{nombre_modelo}**. Entrena el modelo primero."); return None

    for nombre in ["modelo.keras","mejor_modelo_rnn.keras","modelo.h5"]:
        if (dir_prog/nombre).exists():
            ruta_modelo = dir_prog/nombre; break
    else:
        st.error(f"❌ Sin archivo .keras/.h5 para **{nombre_modelo}**."); return None

    ruta_cfg = dir_prog / "config.pkl"
    if not ruta_cfg.exists():
        st.error(f"❌ Sin config.pkl para **{nombre_modelo}**."); return None
    try:
        with open(ruta_cfg,"rb") as f: config = pickle.load(f)
    except Exception as e:
        st.error(f"❌ Error leyendo config.pkl: {e}"); return None

    claves = ["features_acad_cols","max_semestres","num_features_acad","num_features_socio"]
    faltantes = [k for k in claves if k not in config]
    if faltantes:
        st.error(f"❌ config.pkl incompleto. Faltan: {faltantes}"); return None
    if not config.get("features_acad_cols"):
        st.error("❌ Sin columnas académicas en config.pkl."); return None

    try:
        modelo = _tf_load(str(ruta_modelo))
    except Exception as e:
        st.error(f"❌ Error cargando modelo Keras: {e}"); return None

    umbral_optimo = 0.5
    ruta_met = dir_prog/"metricas.json"
    if ruta_met.exists():
        try:
            with open(ruta_met,encoding="utf-8") as f: met_j = json.load(f)
            umbral_optimo = float(met_j.get("umbral_optimo",0.5))
        except Exception: pass

    datos_rnn_disco = {
        "max_semestres":config["max_semestres"],"num_features_acad":config["num_features_acad"],
        "num_features_socio":config["num_features_socio"],"features_acad_cols":config["features_acad_cols"],
        "features_socio_cols":config.get("features_socio_cols", []),
        "umbral_optimo":umbral_optimo,"programa":config.get("programa",safe_prog),
        "X_sec_train":None,"X_sec_test":None,"X_est_train":None,"X_est_test":None,
        "y_des_train":None,"y_des_test":None,"y_not_train":None,"y_not_test":None,
        "estudiantes_comunes":[],"df_socio":None,"df_acad":None,
        "_cargado_desde_disco":True,"_ruta_modelo":str(ruta_modelo),
    }
    ms = obtener_modelos()
    if nombre_modelo in ms:
        ms[nombre_modelo]["modelo"] = modelo
        ms[nombre_modelo]["datos_rnn"] = datos_rnn_disco
    st.success(f" Artefactos RNN cargados para **{nombre_modelo}**.")
    return datos_rnn_disco


def _safe_prog_modelo(nombre_modelo: str, info: dict) -> str:
    """Obtiene el código SAFE del programa asociado al modelo seleccionado."""
    candidatos = []
    if isinstance(info, dict):
        candidatos.extend([
            info.get("programa"),
            (info.get("datos_rnn") or {}).get("programa"),
            (info.get("metricas") or {}).get("programa"),
        ])
    candidatos.append(_extraer_safe_prog_de_nombre(nombre_modelo))
    for c in candidatos:
        if c is None:
            continue
        txt = str(c).strip()
        if not txt or txt.lower() in ("global", "none", "nan"):
            continue
        return _nombre_archivo_seguro(txt).upper()
    return _extraer_safe_prog_de_nombre(nombre_modelo)


def _ruta_dataset_programa(safe_prog: str, tipo: str) -> Path:
    carpetas = {
        "demo": "demograficos",
        "acad": "academicos",
        "des": "desertores",
    }
    sufijos = {
        "demo": "DEMOGRAFICOS",
        "acad": "ACADEMICOS",
        "des": "DESERTORES",
    }
    return OUTPUTS_DIR / "por_carrera" / carpetas[tipo] / f"DF_{safe_prog}_{sufijos[tipo]}.csv"


def _inferir_features_socio_rnn(datos_rnn: dict, safe_prog: str) -> list[str]:
    """Recupera el orden de columnas sociodemográficas usado por la RNN."""
    cols = list((datos_rnn or {}).get("features_socio_cols") or [])
    if cols:
        return cols
    ruta_demo = _ruta_dataset_programa(safe_prog, "demo")
    if ruta_demo.exists():
        try:
            cols_demo = pd.read_csv(ruta_demo, nrows=0).columns.tolist()
            cols = [c for c in cols_demo if c not in ["ID_EST", "PROGRAMA"]]
            n = int((datos_rnn or {}).get("num_features_socio", len(cols)))
            return cols[:n]
        except Exception:
            pass
    n = int((datos_rnn or {}).get("num_features_socio", 0) or 0)
    return [f"VARIABLE_SOCIO_{i+1}" for i in range(n)]


def _cargar_datos_programa(safe_prog: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ruta_demo = _ruta_dataset_programa(safe_prog, "demo")
    ruta_acad = _ruta_dataset_programa(safe_prog, "acad")
    if not ruta_demo.exists() or not ruta_acad.exists():
        faltan = [str(r) for r in [ruta_demo, ruta_acad] if not r.exists()]
        raise FileNotFoundError("No se encontraron datasets del programa: " + "; ".join(faltan))
    df_demo = pd.read_csv(ruta_demo)
    df_acad = pd.read_csv(ruta_acad)
    for df in [df_demo, df_acad]:
        if "ID_EST" in df.columns:
            df["ID_EST"] = pd.to_numeric(df["ID_EST"], errors="coerce").fillna(-1).astype(int)
    return df_demo, df_acad


def _asegurar_rnn_cargada(nombre_modelo: str, info: dict) -> tuple[Optional[dict], Optional[object], str]:
    safe_prog = _safe_prog_modelo(nombre_modelo, info)
    datos_rnn = info.get("datos_rnn") if isinstance(info, dict) else None
    if datos_rnn is None or info.get("modelo") is None:
        with st.spinner("Cargando artefactos RNN desde disco…"):
            datos_rnn = _cargar_artefactos_rnn_para_prediccion(nombre_modelo)
    info_act = obtener_modelos().get(nombre_modelo, info)
    modelo_rnn = info_act.get("modelo") if isinstance(info_act, dict) else None
    datos_rnn = (info_act.get("datos_rnn") if isinstance(info_act, dict) else datos_rnn) or datos_rnn
    return datos_rnn, modelo_rnn, safe_prog


def _render_prediccion(prob_pct: float, nota: Optional[float], sem: Optional[int], umbral_val: float, modelo_txt: str):
    umbral_pct = umbral_val * 100.0
    es_alto = prob_pct >= umbral_pct
    cls = "pred-alto" if es_alto else "pred-bajo"
    col_p = "#E74C3C" if es_alto else "#2ECC71"
    nota_txt = ""
    if nota is not None:
        if sem is not None:
            nota_txt = f"Nota proyectada semestre {sem}: <strong>{nota:.2f}</strong>"
        else:
            nota_txt = f"Nota proyectada: <strong>{nota:.2f}</strong>"
    st.markdown(f"""<div class="pred-resultado {cls}">
        <div class="pred-prob" style="color:{col_p}">{prob_pct:.1f}%</div>
        <div class="pred-label">{" RIESGO ALTO" if es_alto else " BAJO RIESGO"}</div>
        <div class="pred-nota">{nota_txt} &nbsp;·&nbsp; Umbral: {umbral_val:.2f} ({umbral_pct:.0f}%)<br>{modelo_txt}</div>
    </div>""", unsafe_allow_html=True)



def _normalizar_id_col(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza ID_EST para búsquedas y predicciones masivas."""
    out = df.copy()
    if "ID_EST" in out.columns:
        out["ID_EST"] = pd.to_numeric(out["ID_EST"], errors="coerce").fillna(-1).astype(int)
    return out


def _filtrar_df_por_modelo(df: pd.DataFrame, safe_prog: str) -> pd.DataFrame:
    """Filtra un DataFrame al programa del modelo usando nombre seguro."""
    if df is None or df.empty:
        return pd.DataFrame(columns=getattr(df, "columns", []))
    out = _normalizar_id_col(df)
    if "PROGRAMA" in out.columns:
        prog_safe = out["PROGRAMA"].apply(lambda x: _nombre_archivo_seguro(str(x)).upper())
        out = out[prog_safe == safe_prog].copy()
    return out


def _crear_zip_no_clasificables_por_carrera() -> Optional[bytes]:
    """Crea un ZIP en memoria con no clasificables académicos y demográficos por carrera."""
    ruta_acad = INTER_DIR / "DF_NO_CLASIFICABLES_PREDICCION.csv"
    ruta_demo = INTER_DIR / "DF_NO_CLASIFICABLES_DEMOGRAFICOS.csv"
    if not ruta_acad.exists() or not ruta_demo.exists():
        return None

    df_acad = pd.read_csv(ruta_acad)
    df_demo = pd.read_csv(ruta_demo)
    if "PROGRAMA" not in df_acad.columns:
        raise ValueError("El archivo académico no clasificable no contiene la columna PROGRAMA.")

    for df in [df_acad, df_demo]:
        if "ID_EST" in df.columns:
            df["ID_EST"] = pd.to_numeric(df["ID_EST"], errors="coerce").fillna(-1).astype(int)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        programas = sorted(df_acad["PROGRAMA"].dropna().astype(str).unique().tolist())
        for prog in programas:
            safe = _nombre_archivo_seguro(prog).upper()
            acad_prog = df_acad[df_acad["PROGRAMA"].astype(str) == prog].copy()

            if "PROGRAMA" in df_demo.columns:
                demo_prog = df_demo[df_demo["PROGRAMA"].astype(str) == prog].copy()
            elif "ID_EST" in df_demo.columns:
                demo_prog = df_demo[df_demo["ID_EST"].isin(acad_prog["ID_EST"].unique())].copy()
            else:
                demo_prog = pd.DataFrame(columns=df_demo.columns)

            zf.writestr(
                f"Por carrera/Academicos/DF_{safe}_ACADEMICOS_NO_CLASIFICABLES_PREDICCION.csv",
                acad_prog.to_csv(index=False),
            )
            zf.writestr(
                f"Por carrera/Demograficos/DF_{safe}_DEMOGRAFICOS_NO_CLASIFICABLES.csv",
                demo_prog.to_csv(index=False),
            )

    buffer.seek(0)
    return buffer.getvalue()


def _generar_predicciones_masivas(
    modelo_sel: str,
    info: dict,
    tipo_actual: str,
    tarea_actual: str,
    umbral_val: float,
    df_demo_input: Optional[pd.DataFrame] = None,
    df_acad_input: Optional[pd.DataFrame] = None,
    origen_datos: str = "datasets internos",
) -> pd.DataFrame:
    """Genera predicciones masivas desde datasets internos o CSV cargados por el usuario."""
    safe_prog = _safe_prog_modelo(modelo_sel, info)

    if df_demo_input is not None and df_acad_input is not None:
        df_demo = _filtrar_df_por_modelo(df_demo_input, safe_prog)
        df_acad = _filtrar_df_por_modelo(df_acad_input, safe_prog)
        if df_demo.empty or df_acad.empty:
            raise ValueError(
                "Los archivos cargados no contienen estudiantes del programa asociado al modelo seleccionado: "
                f"{safe_prog.replace('_', ' ').title()}."
            )
    else:
        df_demo, df_acad = _cargar_datos_programa(safe_prog)
        origen_datos = "datasets internos por programa"

    if "ID_EST" not in df_demo.columns or "ID_EST" not in df_acad.columns:
        raise ValueError("Los archivos académico y demográfico deben contener la columna ID_EST.")

    df_demo = _normalizar_id_col(df_demo)
    df_acad = _normalizar_id_col(df_acad)
    ids = sorted(set(df_demo["ID_EST"]).intersection(set(df_acad["ID_EST"])))
    filas = []
    fecha_pred = datetime.now().isoformat(timespec="seconds")

    if tipo_actual == "rnn_multi":
        from modelos.modelo_rnn import predecir_estudiante as _pred_rnn, MAX_SEMESTRES_GLOBAL
        datos_rnn, modelo_rnn, safe_prog = _asegurar_rnn_cargada(modelo_sel, info)
        if datos_rnn is None or modelo_rnn is None:
            raise RuntimeError("No se pudo cargar el modelo RNN seleccionado.")
        fac = datos_rnn["features_acad_cols"]
        fsoc = _inferir_features_socio_rnn(datos_rnn, safe_prog)
        max_sem = datos_rnn.get("max_semestres", MAX_SEMESTRES_GLOBAL)
        for id_est in ids:
            datos_soc = df_demo[df_demo["ID_EST"] == id_est].head(1).to_dict("records")
            hist = df_acad[df_acad["ID_EST"] == id_est]
            if not datos_soc or hist.empty:
                continue
            try:
                prob, nota, sem = _pred_rnn(modelo_rnn, datos_soc[0], hist, max_sem, fac, features_socio_cols=fsoc)
                sems = _formatear_semestres(hist["NUMERO_SEMESTRE"].unique().tolist()) if "NUMERO_SEMESTRE" in hist.columns else "—"
                filas.append({
                    "ID_EST": id_est,
                    "PROGRAMA": hist["PROGRAMA"].iloc[0] if "PROGRAMA" in hist.columns and len(hist) else safe_prog,
                    "SEMESTRES_DISPONIBLES": sems,
                    "MODELO": modelo_sel,
                    "PROB_DESERCION_%": round(prob, 4),
                    "UMBRAL": umbral_val,
                    "RIESGO": "ALTO" if prob >= umbral_val * 100 else "BAJO",
                    "NOTA_PROYECTADA": round(nota, 4),
                    "SEMESTRE_PROYECTADO": sem,
                    "ORIGEN_DATOS": origen_datos,
                    "FECHA_PREDICCION": fecha_pred,
                })
            except Exception as e:
                filas.append({
                    "ID_EST": id_est, "PROGRAMA": safe_prog, "MODELO": modelo_sel,
                    "ORIGEN_DATOS": origen_datos, "FECHA_PREDICCION": fecha_pred,
                    "ERROR": str(e),
                })
    else:
        from modelos.modelos_ml_por_programa import predecir_individual_ml
        for id_est in ids:
            datos_soc = df_demo[df_demo["ID_EST"] == id_est].head(1).to_dict("records")
            hist = df_acad[df_acad["ID_EST"] == id_est]
            if not datos_soc:
                datos_soc = [{}]
            try:
                pred_res = predecir_individual_ml(info, datos_soc[0], hist)
                fila = {
                    "ID_EST": id_est,
                    "PROGRAMA": hist["PROGRAMA"].iloc[0] if "PROGRAMA" in hist.columns and len(hist) else safe_prog,
                    "MODELO": modelo_sel,
                    "TAREA": tarea_actual,
                    "ORIGEN_DATOS": origen_datos,
                    "FECHA_PREDICCION": fecha_pred,
                }
                if "NUMERO_SEMESTRE" in hist.columns:
                    fila["SEMESTRES_DISPONIBLES"] = _formatear_semestres(hist["NUMERO_SEMESTRE"].unique().tolist())
                if tarea_actual == "clasificacion_desercion":
                    prob_pct = float(pred_res.get("probabilidad", 0)) * 100
                    fila.update({
                        "PROB_DESERCION_%": round(prob_pct, 4),
                        "UMBRAL": umbral_val,
                        "RIESGO": "ALTO" if prob_pct >= umbral_val * 100 else "BAJO",
                    })
                elif tarea_actual == "regresion_nota":
                    fila.update({"NOTA_PROYECTADA": round(float(pred_res.get("nota", 0)), 4)})
                filas.append(fila)
            except Exception as e:
                filas.append({
                    "ID_EST": id_est, "PROGRAMA": safe_prog, "MODELO": modelo_sel,
                    "ORIGEN_DATOS": origen_datos, "FECHA_PREDICCION": fecha_pred,
                    "ERROR": str(e),
                })

    return pd.DataFrame(filas)


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
sidebar_logo()
with st.sidebar:
    pagina = st.radio("", [
        "  Dashboard",
        "  Pipeline de Limpieza",
        "  Entrenamiento de Modelos",
        "  Comparación de Modelos",
        "  Evaluación General",
        "  Predicción Individual",
    ], label_visibility="collapsed")
    st.markdown("---")
    r_ok=_ref_ok(RUTA_RURAL); p_ok=_ref_ok(RUTA_PLANES)
    st.markdown(" Referencia integrada")
    st.markdown(f"{'✅' if r_ok else '❌'} Municipios rurales")
    st.markdown(f"{'✅' if p_ok else '❌'} Planes de estudio")
    st.markdown("---")
    modelos_dict = obtener_modelos()
    if pipeline_listo(): st.markdown(badge("Pipeline ✓","ok"), unsafe_allow_html=True)
    if modelos_dict:     st.markdown(badge(f"{len(modelos_dict)} modelo(s)","info"), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if "Dashboard" in pagina:
    header(" RNN-Alert — Sistema de Alertas Tempranas",
           "Institución Universitaria Colegio Mayor del Cauca · Ingeniería Informática · 2026")
    ruta_des = INTER_DIR/"DF_DESERTORES.csv"; ruta_ac3 = INTER_DIR/"DF_ACADEMICO_3_FINAL.csv"
    if ruta_des.exists() and ruta_ac3.exists():
        @st.cache_data
        def _load_dash(): return pd.read_csv(ruta_des), pd.read_csv(ruta_ac3)
        df_des,df_ac3 = _load_dash()
        n_est=int(df_des["ID_EST"].nunique()); n_des=int(df_des["DESERTOR"].sum())
        tasa=round(n_des/n_est*100,1) if n_est else 0
        n_prog=int(df_des["PROGRAMA"].nunique()); n_mod=len(obtener_modelos())
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.markdown(kpi("Estudiantes únicos",f"{n_est:,}",f"{n_prog} programas"), unsafe_allow_html=True)
        c2.markdown(kpi("Desertores",f"{n_des:,}",f"{tasa}% del total","rojo"), unsafe_allow_html=True)
        c3.markdown(kpi("No desertores",f"{n_est-n_des:,}","","azul"), unsafe_allow_html=True)
        c4.markdown(kpi("Programas",str(n_prog),"pregrado UNIMAYOR","naranja"), unsafe_allow_html=True)
        c5.markdown(kpi("Modelos",str(n_mod),"entrenados","blanco"), unsafe_allow_html=True)
        st.markdown("---")
        col_a,col_b = st.columns([3,2])
        with col_a:
            st.markdown('<div class="seccion-titulo">Deserción por programa</div>', unsafe_allow_html=True)
            pg=df_des.groupby("PROGRAMA")["DESERTOR"].agg(["sum","count"]).reset_index()
            pg["tasa"]=(pg["sum"]/pg["count"]*100).round(1); pg=pg.sort_values("tasa",ascending=True)
            fig,ax=plt.subplots(figsize=(7,max(4,len(pg)*0.42)))
            bars=ax.barh(pg["PROGRAMA"],pg["tasa"],color="#2ECC71",alpha=0.9,height=0.65)
            for b in bars:
                ax.text(b.get_width()+0.3,b.get_y()+b.get_height()/2,
                    f"{b.get_width():.1f}%",va="center",fontsize=8,color="#B8C1CC")
            ax.set_xlabel("Tasa de deserción (%)"); ax.tick_params(axis="y",labelsize=8)
            fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
        with col_b:
            st.markdown('<div class="seccion-titulo">Distribución global</div>', unsafe_allow_html=True)
            fig2,ax2=plt.subplots(figsize=(5,4.5))
            wedges,texts,autotexts=ax2.pie([n_des,n_est-n_des],labels=["Desertores","No desertores"],
                colors=["#E74C3C","#2ECC71"],autopct="%1.1f%%",startangle=90,
                textprops={"color":"#B8C1CC","fontsize":10},wedgeprops={"linewidth":2,"edgecolor":"#0A192F"})
            for at in autotexts: at.set_fontsize(11); at.set_fontweight("bold"); at.set_color("white")
            ax2.set_facecolor("none"); fig2.set_facecolor("none")
            st.pyplot(fig2,use_container_width=True); plt.close(fig2)
        ruta_clas = INTER_DIR/"DF_ENTRENAMIENTO_CLASIFICABLES.csv"
        ruta_nclas= INTER_DIR/"DF_NO_CLASIFICABLES_PREDICCION.csv"
        if ruta_clas.exists() and ruta_nclas.exists():
            st.markdown("---")
            df_clas=pd.read_csv(ruta_clas); df_nc=pd.read_csv(ruta_nclas)
            cc1,cc2 = st.columns(2)
            cc1.markdown(kpi("Clasificables (entrenamiento)",f"{df_clas['ID_EST'].nunique():,}","con ≥4 semestres"), unsafe_allow_html=True)
            cc2.markdown(kpi("No clasificables (predicción temprana)",f"{df_nc['ID_EST'].nunique():,}","con <4 semestres","naranja"), unsafe_allow_html=True)
    else:
        st.info(" Ejecuta el **Pipeline de Limpieza** primero.")
        c1,c2,c3 = st.columns(3)
        c1.markdown(kpi("Pipeline","Pendiente","10 pasos"), unsafe_allow_html=True)
        c2.markdown(kpi("Modelos",str(len(obtener_modelos())),"entrenados","azul"), unsafe_allow_html=True)
        c3.markdown(kpi("Referencia",f"{'✓' if r_ok and p_ok else '✗'}","integrada"), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
elif "Pipeline" in pagina:
    header(" Pipeline de Limpieza de Datos",
           "10 pasos · Unificación propedéutica · Clasificación de estudiantes")
    if not (_ref_ok(RUTA_RURAL) and _ref_ok(RUTA_PLANES)):
        st.error("❌ Archivos de referencia no encontrados en data/referencia/"); st.stop()
    with st.expander("ℹ Pasos del pipeline"):
        for num,desc in [("1","Formalizar datos demográficos"),("2","Formalizar datos académicos"),
            ("3","Imputar TIPO_INSTITUCION"),("4","Asignar semestres a materias"),
            ("5","Unificar trayectorias propedéuticas (TGE→AE, TGF→AF)"),
            ("6","Calcular estadísticas académicas por semestre"),("7","Calcular desertores"),
            ("8","Clasificar estudiantes entrenables vs no clasificables"),
            ("9","Distribuir datasets por programa")]:
            st.markdown(f"**Paso {num}:** {desc}")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**Dataset demográfico**")
        f_demo = st.file_uploader("DATASET_DATOS_DEMOGRAFICOS (.csv/.xlsx)",type=["csv","xlsx"])
    with c2:
        st.markdown("**Dataset académico**")
        f_acad = st.file_uploader("DATASET_DATOS_ACADEMICOS (.csv/.xlsx)",type=["csv","xlsx"])
    st.info(" **Municipios Rurales** y **Planes de Estudio** ya están integrados en el proyecto.")
    archivos_ok = f_demo is not None and f_acad is not None
    ejecutar_pipe = st.button(" Iniciar Pipeline (10 pasos)",disabled=not archivos_ok)

    if ejecutar_pipe and archivos_ok:
        # Limpiar solo salidas del pipeline, conservando modelos entrenados y métricas de modelos.
        for _ruta_tmp in [INTER_DIR, AUDIT_DIR, OUTPUTS_DIR / "por_carrera"]:
            shutil.rmtree(_ruta_tmp, ignore_errors=True)
        INTER_DIR.mkdir(parents=True, exist_ok=True)
        MOD_DIR.mkdir(parents=True, exist_ok=True)
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUTS_DIR / "por_carrera").mkdir(parents=True, exist_ok=True)
        ruta_demo=_save_upload(f_demo); ruta_acad=_save_upload(f_acad)
        barra=st.progress(0,"Iniciando…")
        c_log,c_met=st.columns([3,2])
        with c_log: st.markdown("**Log de ejecución**"); log_c=st.empty()
        with c_met: st.markdown("**Métricas**"); met_c=st.empty()
        msgs=[]
        try:
            barra.progress(5,"Paso 1 · Formalizar demográficos…")
            df_dm_r=cargar_archivo(ruta_demo); df_ru_r=cargar_archivo(RUTA_RURAL)
            ok,er=validar_no_vacio(df_dm_r,"demograficos")
            if not ok: raise ValueError("\n".join(er))
            df_dm1,s1=paso1.ejecutar(df_dm_r,df_ru_r)
            df_dm1.to_csv(INTER_DIR/"DF_DEMOGRAFICOS_1_LIMPIO.csv",index=False)
            msgs=_log(log_c,msgs,f"✔ Paso 1 — {s1['total_final']:,} registros"); barra.progress(12,"Paso 1 ✓"); time.sleep(0.1)

            barra.progress(14,"Paso 2 · Formalizar académicos…")
            df_ac_r=cargar_archivo(ruta_acad); ok,er=validar_no_vacio(df_ac_r,"academicos")
            if not ok: raise ValueError("\n".join(er))
            df_ac2,s2=paso2.ejecutar(df_ac_r)
            df_ac2.to_csv(INTER_DIR/"DF_ACADEMICOS_1_LIMPIO.csv",index=False)
            msgs=_log(log_c,msgs,f"✔ Paso 2 — {s2['total_final']:,} registros"); barra.progress(22,"Paso 2 ✓"); time.sleep(0.1)

            barra.progress(24,"Paso 3 · Imputar TIPO_INSTITUCION…")
            df_dm3,s3=paso3.ejecutar(df_dm1)
            df_dm3.to_csv(INTER_DIR/"DF_DEMOGRAFICOS_2_RELLENADO.csv",index=False)
            msgs=_log(log_c,msgs,f"✔ Paso 3 — Imputados: {s3['imputados']}"); barra.progress(32,"Paso 3 ✓"); time.sleep(0.1)

            barra.progress(34,"Paso 4 · Asignar semestres…")
            df_pl=cargar_archivo(RUTA_PLANES); df_ac4,s4=paso4.ejecutar(df_ac2,df_pl)
            df_ac4.to_csv(INTER_DIR/"DF_ACADEMICO_2_RELLENADO.csv",index=False)
            msgs=_log(log_c,msgs,f"✔ Paso 4 — {s4['semestres_asignados']:,}/{s4['total_final']:,} ({s4['cobertura_pct']}%)"); barra.progress(44,"Paso 4 ✓"); time.sleep(0.1)

            barra.progress(46,"Paso 5 · Unificando propedéuticos…")
            df_ac_prop,df_dm_prop,s5_prop=paso5_prop(df_ac4,df_dm3,ruta_auditoria=AUDIT_DIR)
            df_ac_prop.to_csv(INTER_DIR/"DF_ACADEMICO_3_PROPEDEUTICO.csv",index=False)
            df_dm_prop.to_csv(INTER_DIR/"DF_DEMOGRAFICOS_3_PROPEDEUTICO.csv",index=False)
            msgs=_log(log_c,msgs,f"✔ Paso 5 — {s5_prop['total_propedeuticos_unificados']} propedéuticos | {s5_prop['solo_tecnologia']} solo tec.")
            barra.progress(54,"Paso 5 ✓"); time.sleep(0.1)

            barra.progress(56,"Paso 6 · Estadísticas por semestre…")
            df_ac5,s5=paso5.ejecutar(df_ac_prop,df_dm_prop)
            df_ac5.to_csv(INTER_DIR/"DF_ACADEMICO_3_FINAL.csv",index=False)
            msgs=_log(log_c,msgs,f"✔ Paso 6 — {s5['total_final']:,} filas | {s5['estudiantes_unicos']:,} estudiantes"); barra.progress(64,"Paso 6 ✓"); time.sleep(0.1)

            barra.progress(66,"Paso 7 · Calcular desertores…")
            df_de6,s6=paso6.ejecutar(df_ac5,df_dm_prop)
            df_de6.to_csv(INTER_DIR/"DF_DESERTORES.csv",index=False)
            msgs=_log(log_c,msgs,f"✔ Paso 7 — Desertores: {s6['desertores']:,} ({s6['tasa_desercion_pct']}%)"); barra.progress(74,"Paso 7 ✓"); time.sleep(0.1)

            barra.progress(76,"Paso 8 · Clasificar entrenables…")
            df_clas_ac,df_clas_dm,df_nc_ac,df_nc_dm,s8=paso7_clas(df_ac5,df_dm_prop,df_de6,ruta_auditoria=AUDIT_DIR)
            df_clas_ac.to_csv(INTER_DIR/"DF_ENTRENAMIENTO_CLASIFICABLES.csv",index=False)
            df_nc_ac.to_csv(INTER_DIR/"DF_NO_CLASIFICABLES_PREDICCION.csv",index=False)
            df_nc_dm.to_csv(INTER_DIR/"DF_NO_CLASIFICABLES_DEMOGRAFICOS.csv",index=False)
            msgs=_log(log_c,msgs,f"✔ Paso 8 — Entrenables: {s8['clasificables']:,} | No clasificables: {s8['no_clasificables']:,}")
            barra.progress(84,"Paso 8 ✓"); time.sleep(0.1)

            barra.progress(86,"Paso 9 · Distribuir por carrera…")
            _,s7=paso7.ejecutar(df_dm_prop,df_ac5,df_de6,OUTPUTS_DIR)
            msgs=_log(log_c,msgs,f"✔ Paso 9 — {s7['total_archivos']} archivos generados")
            barra.progress(100," Pipeline completado"); st.session_state["pipeline_ok"]=True

            with met_c.container():
                st.markdown("---"); m1,m2=st.columns(2)
                m1.markdown(kpi("Registros demo",f"{s1['total_final']:,}"), unsafe_allow_html=True)
                m2.markdown(kpi("Registros acad",f"{s2['total_final']:,}"), unsafe_allow_html=True)
                m3,m4=st.columns(2)
                m3.markdown(kpi("Desertores",f"{s6['desertores']:,}","","rojo"), unsafe_allow_html=True)
                m4.markdown(kpi("Propedéuticos",f"{s5_prop['total_propedeuticos_unificados']}","unificados","azul"), unsafe_allow_html=True)
            st.success(" Pipeline completado.")
            d1,d2=st.columns(2)
            with d1:
                z1=crear_zip(OUTPUTS_DIR/"por_carrera")
                if z1: st.download_button("⬇ Datasets por carrera",z1,"datasets_por_carrera.zip","application/zip",use_container_width=True)
            with d2:
                z2=crear_zip(INTER_DIR)
                if z2: st.download_button(" Datasets intermedios",z2,"intermedios.zip","application/zip",use_container_width=True)
        except Exception as e:
            barra.progress(100,"❌ Error"); msgs=_log(log_c,msgs,f"ERROR: {e}","err")
            st.error(f"❌ {e}"); import traceback; logger.error(traceback.format_exc())

    # ── Descargas persistentes de datasets procesados ─────────────────────
    st.markdown("---")
    st.markdown("###  Descargas de datasets procesados")
    st.caption("Disponibles después de ejecutar el pipeline, incluso tras recargar la página.")
    col_dl1,col_dl2,col_dl3,col_dl4 = st.columns(4)
    with col_dl1:
        boton_descarga_df("df_entrenamiento_clasificable",BASE_DIR,
            label_boton="⬇ Estudiantes entrenables",
            nombre_archivo_descarga="DF_ENTRENAMIENTO_CLASIFICABLES.csv",key="dl_clas_pipe")
    with col_dl2:
        boton_descarga_df("df_no_clasificable_prediccion",BASE_DIR,
            label_boton="⬇ No clasificables (predicción)",
            nombre_archivo_descarga="DF_NO_CLASIFICABLES_PREDICCION.csv",key="dl_nc_pred_pipe")
    with col_dl3:
        boton_descarga_df("df_entrenamiento_no_clasificable",BASE_DIR,
            label_boton="⬇ No clasificables (demográficos)",
            nombre_archivo_descarga="DF_NO_CLASIFICABLES_DEMOGRAFICOS.csv",key="dl_nc_demo_pipe")
    with col_dl4:
        try:
            zip_no_clas = _crear_zip_no_clasificables_por_carrera()
            if zip_no_clas:
                st.download_button(
                    "⬇ No clasificables por carrera (ZIP)",
                    data=zip_no_clas,
                    file_name="no_clasificables_por_carrera.zip",
                    mime="application/zip",
                    use_container_width=True,
                    key="dl_nc_por_carrera_zip",
                )
            else:
                st.button("⬇ No clasificables por carrera (ZIP)", disabled=True, use_container_width=True)
        except Exception as e_zip_nc:
            st.warning(f"No se pudo preparar el ZIP de no clasificables: {e_zip_nc}")

# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO DE MODELOS — solo 2 opciones: RNN Multitarea + Modelos ML
# ══════════════════════════════════════════════════════════════════════════════
elif "Entrenamiento" in pagina:
    header(" Entrenamiento de Modelos",
           "RNN Multitarea · Modelos Machine Learning por Programa")

    if not pipeline_listo():
        st.warning(" Ejecuta el **Pipeline de Limpieza** primero."); st.stop()

    # Solo dos tabs: RNN y Modelos Machine Learning
    tabs = st.tabs([" RNN Multitarea", " Modelos Machine Learning"])

    # ── TAB 1: RNN Multitarea ─────────────────────────────────────────────
    with tabs[0]:
        st.markdown("####  RNN Multimodal Multitarea")
        st.caption("Arquitectura secuencial con datos académicos y sociodemográficos · Dos salidas: deserción + nota proyectada")
        try:
            from modelos.entrenamiento_masivo import (
                obtener_programas_disponibles, entrenar_todos_los_programas,
                cargar_reporte_masivo, safe_a_nombre
            )
        except ImportError as e:
            st.error(f"❌ Error importando módulo RNN: {e}"); st.stop()

        programas_safe = obtener_programas_disponibles(OUTPUTS_DIR)
        if not programas_safe:
            st.warning(" Ejecuta el Pipeline primero (Paso 9 genera datasets por carrera).")
        else:
            with st.expander(" Configuración de entrenamiento"):
                ep_rnn=st.slider("Épocas máx",20,200,150,10,key="ep_rnn")
                bs_rnn=st.slider("Batch size",16,64,32,8,key="bs_rnn")
                pa_rnn=st.slider("Patience",5,30,15,5,key="pa_rnn")

            opciones_rnn = ["— Todos los programas —"] + [safe_a_nombre(s) for s in programas_safe]
            seleccion_rnn = st.selectbox("Programa a entrenar",opciones_rnn,key="rnn_prog_selector")
            modo_masivo_rnn = (seleccion_rnn == "— Todos los programas —")
            lbl_rnn = " Entrenar RNN — todos los programas" if modo_masivo_rnn else f" Entrenar RNN — {seleccion_rnn}"
            ejecutar_rnn = st.button(lbl_rnn, key="btn_rnn")

            if ejecutar_rnn and not modo_masivo_rnn:
                try:
                    from modelos.modelo_rnn import preparar_datos_rnn, entrenar_modelo_rnn, evaluar_modelo_rnn, guardar_modelo
                    safe_prog = _nombre_archivo_seguro(seleccion_rnn).upper()
                    ruta_s=OUTPUTS_DIR/"por_carrera"/"demograficos"/f"DF_{safe_prog}_DEMOGRAFICOS.csv"
                    ruta_a=OUTPUTS_DIR/"por_carrera"/"academicos"  /f"DF_{safe_prog}_ACADEMICOS.csv"
                    ruta_l=OUTPUTS_DIR/"por_carrera"/"desertores"  /f"DF_{safe_prog}_DESERTORES.csv"
                    if not all(r.exists() for r in [ruta_s,ruta_a,ruta_l]):
                        st.error(f"❌ Archivos no encontrados para: {seleccion_rnn}"); st.stop()
                    bar_ind=st.progress(0,f"Preparando {seleccion_rnn}…")
                    with st.spinner("Preparando datos…"):
                        df_s=pd.read_csv(ruta_s); df_a=pd.read_csv(ruta_a); df_l=pd.read_csv(ruta_l)
                        datos_rnn=preparar_datos_rnn(df_s,df_a,df_l)
                        st.session_state["rnn_datos"]=datos_rnn
                    bar_ind.progress(30,f"Entrenando {seleccion_rnn}…")
                    with st.spinner(f"Entrenando RNN — {seleccion_rnn}…"):
                        modelo_rnn,hist_rnn=entrenar_modelo_rnn(datos_rnn,epochs=ep_rnn,batch_size=bs_rnn,patience=pa_rnn,ruta_salida=MOD_DIR)
                    bar_ind.progress(80,"Evaluando…"); met_rnn=evaluar_modelo_rnn(modelo_rnn,datos_rnn)
                    guardar_modelo(modelo_rnn,datos_rnn,MOD_DIR); bar_ind.progress(100,"")
                    registrar_modelo(
                        f"RNN Multitarea ({seleccion_rnn})",
                        modelo_rnn,
                        None,
                        met_rnn,
                        "rnn_multi",
                        programa=safe_prog,
                        datos_rnn=datos_rnn,
                    )
                    st.success(f" RNN Multitarea entrenada para {seleccion_rnn}")
                except ImportError: st.error("❌ Instala TensorFlow: pip install tensorflow>=2.15")
                except Exception as e: st.error(f"❌ {e}"); logger.exception("RNN error")

            elif ejecutar_rnn and modo_masivo_rnn:
                try:
                    total_progs=len(programas_safe); barra_m=st.progress(0,f"0/{total_progs} programas")
                    kc1,kc2=st.columns(2); kpi_act=kc1.empty(); kpi_comp=kc2.empty()
                    log_m=st.empty(); msgs_m=[]; t0=time.time()
                    def _cb_rnn(idx,total,sp):
                        pct=int(idx/total*100) if total else 0
                        nm=safe_a_nombre(sp) if sp!="COMPLETADO" else " Completado"
                        barra_m.progress(pct,f"Programa {idx}/{total} — {nm}")
                        kpi_act.markdown(kpi("Actual",nm,""), unsafe_allow_html=True)
                        kpi_comp.markdown(kpi("Completados",f"{idx}/{total}",f"{round(time.time()-t0)}s","azul"), unsafe_allow_html=True)
                    res_masivo=entrenar_todos_los_programas(OUTPUTS_DIR,epochs=ep_rnn,batch_size=bs_rnn,patience=pa_rnn,callback_progreso=_cb_rnn)
                    for res in res_masivo["resultados"]:
                        if res["ok"] and res["modelo"] is not None:
                            nombre_prog_completo = safe_a_nombre(res["programa"])
                            registrar_modelo(
                                f"RNN Multitarea ({nombre_prog_completo})",
                                res["modelo"],
                                None,
                                res["metricas"],
                                "rnn_multi",
                                programa=res["programa"],
                                datos_rnn=res.get("datos"),
                            )
                            msgs_m=_log(log_m,msgs_m,f"✔ {nombre_prog_completo} — AUC={res['metricas'].get('roc_auc',0):.4f}")
                        else:
                            msgs_m=_log(log_m,msgs_m,f"✘ {safe_a_nombre(res['programa'])} — {res['error']}","err")
                    res_r=res_masivo["resumen"]; st.success(f" {res_r['ok']}/{res_r['total']} programas completados")
                    df_met=res_masivo["df_metricas"]
                    st.dataframe(df_met[["programa","roc_auc","recall","f1","estado"]].assign(
                        programa=df_met["programa"].apply(safe_a_nombre)),use_container_width=True,hide_index=True)
                except ImportError: st.error("❌ Instala TensorFlow")
                except Exception as e: st.error(f"❌ {e}"); logger.exception("RNN masivo error")

        # Mostrar RNNs ya entrenadas
        rnn_keys = [k for k in obtener_modelos() if "RNN Multitarea" in k]
        if rnn_keys:
            st.markdown("---")
            st.markdown("RNN Multitarea ya entrenada:")
            for k in rnn_keys:
                with st.expander(f" {_nombre_completo(k)}"):
                    _mostrar_resultados_rnn(k, obtener_modelos()[k]["metricas"])

    # ── TAB 2: Modelos Machine Learning ───────────────────────────────────
    with tabs[1]:
        st.markdown("####  Modelos Machine Learning")
        st.caption(
            "Entrena 6 modelos por programa/carrera:\n"
            "**Clasificación (datos sociodemográficos):** SVM Classifier · RF Classifier · Regresión Logística\n"
            "**Regresión (datos académicos):** SVM Regressor · RF Regressor · Regresión Lineal"
        )
        try:
            from modelos.modelos_ml_por_programa import (
                entrenar_programa_ml, entrenar_todos_los_programas_ml,
                _NOMBRE_LEGIBLE, _TIPO_TAREA, auto_cargar_modelos_ml_en_session
            )
            from modelos.entrenamiento_masivo import (
                obtener_programas_disponibles as _get_progs,
                safe_a_nombre as _sn
            )
        except ImportError as e:
            st.error(f"❌ Error importando módulos Machine Learning: {e}"); st.stop()

        programas_ml = _get_progs(OUTPUTS_DIR)
        if not programas_ml:
            st.warning(" Ejecuta el Pipeline primero para generar los datasets por programa.")
        else:
            opciones_ml = ["— Todos los programas —"] + [_sn(s) for s in programas_ml]
            sel_ml = st.selectbox("Programa a entrenar", opciones_ml, key="ml_prog_selector")
            modo_masivo_ml = (sel_ml == "— Todos los programas —")
            lbl_ml = " Entrenar Modelos Machine Learning — todos los programas" if modo_masivo_ml else f" Entrenar Modelos Machine Learning — {sel_ml}"
            ejecutar_ml = st.button(lbl_ml, key="btn_ml_prog")

            if ejecutar_ml and not modo_masivo_ml:
                safe_p = _nombre_archivo_seguro(sel_ml).upper()
                with st.spinner(f"Entrenando 6 modelos Machine Learning para {sel_ml}…"):
                    try:
                        res = entrenar_programa_ml(safe_p, OUTPUTS_DIR)
                        if res["ok"]:
                            st.success(f" {len(res['modelos'])} Modelos Machine Learning entrenados para {sel_ml}")
                            for clave, info_mod in res["modelos"].items():
                                nom = f"{_NOMBRE_LEGIBLE.get(clave,clave)} — {safe_p}"
                                cfg = res["config_socio"] if "clas" in clave or "rl" in clave else res["config_acad"]
                                registrar_modelo(nom, info_mod["modelo"], cfg.get("scaler"), info_mod["metricas"], "ml_por_programa")
                                obtener_modelos()[nom]["tarea"]    = _TIPO_TAREA.get(clave,"")
                                obtener_modelos()[nom]["clave"]    = clave
                                obtener_modelos()[nom]["programa"] = safe_p
                                obtener_modelos()[nom]["features"] = cfg.get("features",[])
                                obtener_modelos()[nom]["config"]   = cfg
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.markdown("**Clasificación — AUC**")
                                for k in ["svm_clas","rf_clas","rl_clas"]:
                                    m = res["metricas_global"].get(k,{})
                                    st.metric(_NOMBRE_LEGIBLE.get(k,k), _fmt_met(m.get("roc_auc")))
                            with col_b:
                                st.markdown("**Regresión — R²**")
                                for k in ["svm_reg","rf_reg","lineal_reg"]:
                                    m = res["metricas_global"].get(k,{})
                                    st.metric(_NOMBRE_LEGIBLE.get(k,k), _fmt_met(m.get("r2")))
                        else:
                            st.error(f"❌ {res['error']}")
                    except Exception as e:
                        st.error(f"❌ Error en entrenamiento: {e}"); logger.exception("ML prog error")

            elif ejecutar_ml and modo_masivo_ml:
                total_ml = len(programas_ml)
                barra_ml = st.progress(0, f"0/{total_ml} programas")
                log_ml = st.empty(); t0_ml = time.time()
                def _cb_ml(idx,total,sp):
                    pct = int(idx/total*100) if total else 0
                    barra_ml.progress(pct, f"Programa {idx}/{total} — {_sn(sp) if sp!='COMPLETADO' else ' Listo'}")
                try:
                    res_ml = entrenar_todos_los_programas_ml(OUTPUTS_DIR, programas_ml, callback_progreso=_cb_ml)
                    st.session_state["_modelos_ml_cargados_disco"] = False
                    auto_cargar_modelos_ml_en_session(OUTPUTS_DIR)
                    res_r = res_ml["resumen"]
                    st.success(f" {res_r['ok']}/{res_r['total']} programas completados")
                    df_met_ml = res_ml["df_metricas"]
                    clas_df = df_met_ml[df_met_ml["tipo"]=="clasificacion_desercion"].dropna(subset=["roc_auc"])
                    reg_df  = df_met_ml[df_met_ml["tipo"]=="regresion_nota"].dropna(subset=["r2"])
                    if not clas_df.empty:
                        st.markdown("**Clasificación — ROC-AUC**")
                        st.dataframe(clas_df[["programa","modelo","roc_auc","recall","f1"]],use_container_width=True,hide_index=True)
                    if not reg_df.empty:
                        st.markdown("**Regresión — R²**")
                        st.dataframe(reg_df[["programa","modelo","mae","rmse","r2"]],use_container_width=True,hide_index=True)
                except Exception as e:
                    st.error(f"❌ Error en entrenamiento masivo: {e}"); logger.exception("ML masivo error")

        # Modelos Machine Learning ya entrenados
        ml_prog_keys = [k for k,v in obtener_modelos().items() if v.get("tipo")=="ml_por_programa"]
        if ml_prog_keys:
            st.markdown("---")
            st.markdown("** Modelos Machine Learning ya entrenados:**")
            progs_con_ml = sorted(set(obtener_modelos()[k].get("programa","?") for k in ml_prog_keys))
            for prog in progs_con_ml:
                with st.expander(f"📁 {prog.replace('_',' ').title()}"):
                    clas_keys = [k for k in ml_prog_keys
                                 if obtener_modelos()[k].get("programa")==prog
                                 and obtener_modelos()[k].get("tarea")=="clasificacion_desercion"]
                    reg_keys  = [k for k in ml_prog_keys
                                 if obtener_modelos()[k].get("programa")==prog
                                 and obtener_modelos()[k].get("tarea")=="regresion_nota"]
                    if clas_keys:
                        st.markdown("*Clasificación de deserción (datos sociodemográficos):*")
                        for k in clas_keys:
                            m = obtener_modelos()[k]["metricas"]
                            auc_txt = _fmt_met(m.get("roc_auc"))
                            f1_txt  = _fmt_met(m.get("f1_opt"))
                            st.markdown(f"  • **{_nombre_completo(k.split('—')[0])}** — AUC: `{auc_txt}` · F1: `{f1_txt}`")
                    if reg_keys:
                        st.markdown("*Regresión de nota (datos académicos):*")
                        for k in reg_keys:
                            m = obtener_modelos()[k]["metricas"]
                            r2_txt  = _fmt_met(m.get("r2"))
                            mae_txt = _fmt_met(m.get("mae"))
                            st.markdown(f"  • **{_nombre_completo(k.split('—')[0])}** — R²: `{r2_txt}` · MAE: `{mae_txt}`")

# ══════════════════════════════════════════════════════════════════════════════
# COMPARACIÓN DE MODELOS — por bloques: Clasificación y Regresión
# ══════════════════════════════════════════════════════════════════════════════
elif "Comparación" in pagina:
    header(" Comparación de Modelos","Bloque 1: Clasificación de Deserción · Bloque 2: Regresión de Nota")
    from modelos.comparacion_modelos import construir_tablas_comparacion, resumen_global, programas_en_modelos

    modelos=obtener_modelos()
    if not modelos:
        st.warning(" Entrena al menos un modelo primero."); st.stop()

    # Filtrar solo tipos válidos: rnn_multi y ml_por_programa (ignorar tipos "ml" antiguo individual)
    modelos_validos = {k:v for k,v in modelos.items() if v.get("tipo") in ("rnn_multi","ml_por_programa")}

    if not modelos_validos:
        st.info(" Entrena modelos desde **Entrenamiento de Modelos** para ver la comparación.")
        st.stop()

    progs_disponibles = programas_en_modelos(modelos_validos)
    filtro_opciones   = ["— Global (todos los programas) —"] + progs_disponibles
    prog_filtro_ui    = st.selectbox("Filtrar por programa", filtro_opciones, key="comp_prog")
    prog_filtro       = None if prog_filtro_ui.startswith("—") else prog_filtro_ui

    df_clas, df_reg = construir_tablas_comparacion(modelos_validos, prog_filtro)

    st.markdown("---")
    # ── Bloque 1: Clasificación ────────────────────────────────────────────
    st.markdown("##  Bloque 1 — Clasificación de Deserción")
    st.caption("Métricas: ROC-AUC · PR-AUC · Recall · Precision · F1 · Accuracy · Umbral óptimo")
    if df_clas.empty:
        st.info("No hay modelos de clasificación disponibles para este filtro.")
    else:
        st.dataframe(df_clas, use_container_width=True, hide_index=True)
        df_c2 = df_clas.copy()
        df_c2["_auc"] = pd.to_numeric(df_c2["ROC-AUC"], errors="coerce")
        df_c2 = df_c2.dropna(subset=["_auc"]).sort_values("_auc", ascending=False)
        if not df_c2.empty:
            fig,ax = plt.subplots(figsize=(max(8,len(df_c2)*1.2),4))
            etiquetas = df_c2["Modelo"] + "\n" + df_c2["Programa"]
            bars = ax.bar(range(len(df_c2)), df_c2["_auc"].values, color="#0074D9", alpha=0.85)
            for b in bars:
                ax.annotate(f"{b.get_height():.4f}", xy=(b.get_x()+b.get_width()/2,b.get_height()),
                    xytext=(0,3), textcoords="offset points", ha="center", fontsize=8)
            ax.set_xticks(range(len(df_c2))); ax.set_xticklabels(etiquetas,rotation=30,ha="right",fontsize=8)
            ax.set_ylabel("ROC-AUC"); ax.set_ylim(0,1.12); ax.set_title("ROC-AUC — Clasificación de Deserción")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.download_button("⬇ Exportar clasificación (CSV)",
            data=df_clas.to_csv(index=False).encode(), file_name="comparacion_clasificacion.csv")

    st.markdown("---")
    # ── Bloque 2: Regresión ────────────────────────────────────────────────
    st.markdown("##  Bloque 2 — Regresión de Nota")
    st.caption("Métricas: MAE · MSE · RMSE · R²")
    if df_reg.empty:
        st.info("No hay modelos de regresión disponibles para este filtro.")
    else:
        st.dataframe(df_reg, use_container_width=True, hide_index=True)
        df_r2 = df_reg.copy()
        df_r2["_r2"] = pd.to_numeric(df_r2["R²"], errors="coerce")
        df_r2 = df_r2.dropna(subset=["_r2"]).sort_values("_r2", ascending=False)
        if not df_r2.empty:
            fig,ax = plt.subplots(figsize=(max(8,len(df_r2)*1.2),4))
            etiquetas = df_r2["Modelo"] + "\n" + df_r2["Programa"]
            bars = ax.bar(range(len(df_r2)), df_r2["_r2"].values, color="#2ECC71", alpha=0.85)
            for b in bars:
                ax.annotate(f"{b.get_height():.4f}", xy=(b.get_x()+b.get_width()/2,b.get_height()),
                    xytext=(0,3), textcoords="offset points", ha="center", fontsize=8)
            ax.set_xticks(range(len(df_r2))); ax.set_xticklabels(etiquetas,rotation=30,ha="right",fontsize=8)
            ax.set_ylabel("R²"); ax.set_title("R² — Regresión de Nota")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.download_button("⬇ Exportar regresión (CSV)",
            data=df_reg.to_csv(index=False).encode(), file_name="comparacion_regresion.csv")

    st.markdown("---")
    # ── Resumen global ─────────────────────────────────────────────────────
    st.markdown("##  Resumen Global")
    try:
        res = resumen_global(modelos_validos)
        cg = res.get("mejor_clas_global"); rg = res.get("mejor_reg_global")
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**Mejor modelo de clasificación (ROC-AUC)**")
            if cg: st.success(f" **{cg['Modelo']}** — {cg['Programa']}\nAUC: `{cg['ROC-AUC']}` · Recall: `{cg['Recall']}` · F1: `{cg['F1']}`")
            else:  st.info("Sin datos suficientes.")
        with c2:
            st.markdown("**Mejor modelo de regresión (R²)**")
            if rg: st.success(f" **{rg['Modelo']}** — {rg['Programa']}\nR²: `{rg['R²']}` · MAE: `{rg['MAE']}` · RMSE: `{rg['RMSE']}`")
            else:  st.info("Sin datos suficientes.")
        if res["mejores_clas_por_prog"] or res["mejores_reg_por_prog"]:
            st.markdown("---"); st.markdown("**Mejores por programa:**")
            all_progs = sorted(set(list(res["mejores_clas_por_prog"]) + list(res["mejores_reg_por_prog"])))
            for p in all_progs:
                with st.expander(f" {p}"):
                    clas_p = res["mejores_clas_por_prog"].get(p)
                    reg_p  = res["mejores_reg_por_prog"].get(p)
                    if clas_p: st.markdown(f" **Clasificación:** {clas_p.get('Modelo','—')} — AUC: `{clas_p.get('ROC-AUC','—')}` · Recall: `{clas_p.get('Recall','—')}`")
                    if reg_p:  st.markdown(f" **Regresión:** {reg_p.get('Modelo','—')} — R²: `{reg_p.get('R²','—')}` · MAE: `{reg_p.get('MAE','—')}`")
    except Exception as e:
        st.warning(f"No se pudo generar el resumen global: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# EVALUACIÓN GENERAL — curvas ROC y matrices de confusión
# ══════════════════════════════════════════════════════════════════════════════
elif "Evaluación" in pagina:
    header(" Evaluación General","Curvas ROC y matrices de confusión")
    modelos=obtener_modelos()
    if not modelos: st.warning(" No hay modelos entrenados."); st.stop()
    # Solo modelos de clasificación (tienen ROC y CM): RNN y ML clasificación
    modelos_clas = {k:v for k,v in modelos.items()
                    if v.get("tipo") == "rnn_multi"
                    or (v.get("tipo") == "ml_por_programa" and v.get("tarea") == "clasificacion_desercion")}
    if not modelos_clas:
        st.info("ℹ No hay modelos de clasificación entrenados aún."); st.stop()
    n=len(modelos_clas); nombres=list(modelos_clas.keys())
    cols_per_row=3
    for i in range(0,n,cols_per_row):
        cols=st.columns(min(cols_per_row,n-i))
        for j,col in enumerate(cols):
            if i+j>=n: break
            nombre=nombres[i+j]; m=modelos_clas[nombre]["metricas"]
            with col:
                prog_eval = _programa_desde_modelo(nombre, modelos_clas[nombre])
                st.markdown(f"**{_nombre_completo(nombre)}**")
                if prog_eval != "Global":
                    st.caption(f"Programa: {prog_eval}")
                auc_v = m.get("roc_auc", 0)
                rec_v = m.get("recall_opt", 0)
                st.markdown(f"AUC: `{_fmt_met(auc_v)}` · Recall: `{_fmt_met(rec_v)}`")
                if "fpr" in m:
                    fig=_plot_roc(np.array(m["fpr"]),np.array(m["tpr"]),m.get("roc_auc",0))
                    st.pyplot(fig,use_container_width=True); plt.close(fig)
                if "cm" in m:
                    fig2=_plot_cm(m["cm"]); st.pyplot(fig2,use_container_width=True); plt.close(fig2)

# ══════════════════════════════════════════════════════════════════════════════
# PREDICCIÓN INDIVIDUAL
# ══════════════════════════════════════════════════════════════════════════════
elif "Predicción" in pagina:
    header(" Predicción Individual",
           "Busca en clasificables, no clasificables y trayectorias propedéuticas")
    modelos=obtener_modelos()
    if not modelos:
        st.warning(" Entrena al menos un modelo primero."); st.stop()

    # Solo mostrar modelos válidos (RNN y ML por programa), excluir ML individuales antiguos
    modelos_validos = {k:v for k,v in modelos.items() if v.get("tipo") in ("rnn_multi","ml_por_programa")}
    if not modelos_validos:
        st.warning(" No hay modelos RNN Multitarea o Modelos Machine Learning entrenados."); st.stop()

    modelo_sel = st.selectbox("Modelo a usar", [_nombre_completo(k) for k in modelos_validos.keys()])
    info = modelos_validos[modelo_sel]
    tarea_actual = info.get("tarea","")
    tipo_actual  = info.get("tipo","")

    umbral=None
    if tipo_actual == "rnn_multi" or tarea_actual == "clasificacion_desercion":
        umbral=st.slider("Umbral de clasificación",0.1,0.9,float(info["metricas"].get("umbral_optimo",0.5)),0.01)

    def buscar_estudiante(id_est_input):
        try: id_est = int(float(str(id_est_input).strip()))
        except Exception: return {"encontrado":False,"motivo":"ID_EST no es numérico válido"}
        resultado={"id_est":id_est,"encontrado":False,"en_clasificables":False,"en_no_clasificables":False,"programas":[],"es_propedeutico":False,"motivo":""}
        ruta_ac=INTER_DIR/"DF_ENTRENAMIENTO_CLASIFICABLES.csv"
        if not ruta_ac.exists(): ruta_ac=INTER_DIR/"DF_ACADEMICO_3_FINAL.csv"
        if ruta_ac.exists():
            df_ac=pd.read_csv(ruta_ac)
            df_ac["ID_EST"]=pd.to_numeric(df_ac["ID_EST"],errors="coerce").fillna(-1).astype(int)
            sub=df_ac[df_ac["ID_EST"]==id_est]
            if len(sub):
                resultado["en_clasificables"]=True; resultado["encontrado"]=True
                resultado["programas"]=sub["PROGRAMA"].unique().tolist(); resultado["df_acad"]=sub
                if "ES_PROPEDEUTICO" in sub.columns: resultado["es_propedeutico"]=bool(sub["ES_PROPEDEUTICO"].max())
        ruta_nc=INTER_DIR/"DF_NO_CLASIFICABLES_PREDICCION.csv"
        if ruta_nc.exists():
            df_nc=pd.read_csv(ruta_nc)
            df_nc["ID_EST"]=pd.to_numeric(df_nc["ID_EST"],errors="coerce").fillna(-1).astype(int)
            sub_nc=df_nc[df_nc["ID_EST"]==id_est]
            if len(sub_nc):
                resultado["en_no_clasificables"]=True
                if not resultado["encontrado"]:
                    resultado["encontrado"]=True; resultado["programas"]+=sub_nc["PROGRAMA"].unique().tolist(); resultado["df_acad_nc"]=sub_nc
        if not resultado["encontrado"]:
            resultado["motivo"]="El estudiante no aparece en ningún dataset procesado."
        elif resultado["en_no_clasificables"] and not resultado["en_clasificables"]:
            resultado["motivo"]=" Estudiante encontrado pero clasificado como NO CLASIFICABLE (≤3 semestres)."
        return resultado

    tab_id, tab_manual, tab_masivo = st.tabs([" Por ID_EST"," Datos manuales"," Predicción masiva"])

    with tab_id:
        id_input = st.text_input("ID_EST del estudiante",placeholder="Ej: 12345")
        if st.button(" Buscar y predecir"):
            if not id_input.strip(): st.warning("Ingresa un ID_EST.")
            else:
                resultado = buscar_estudiante(id_input)
                if not resultado["encontrado"]:
                    st.error(f"❌ Estudiante no encontrado. {resultado['motivo']}")
                    audit_row={"ID_EST":id_input,"existe_dataset_entrenamiento":False,
                        "existe_dataset_no_clasificable":False,"estado_busqueda":"NO_ENCONTRADO",
                        "observacion":resultado["motivo"],"timestamp":datetime.now().isoformat()}
                    ruta_audit=AUDIT_DIR/"auditoria_busqueda_estudiantes.csv"
                    df_audit=pd.DataFrame([audit_row])
                    if ruta_audit.exists(): df_audit=pd.concat([pd.read_csv(ruta_audit),df_audit],ignore_index=True)
                    df_audit.to_csv(ruta_audit,index=False)
                    st.info(f"Consulta registrada en auditoría: {ruta_audit}")
                else:
                    if resultado["motivo"]: st.warning(resultado["motivo"])
                    progs=resultado["programas"]
                    prog_pred=st.selectbox("Programa para predicción",progs,key="prog_pred_id") if len(progs)>1 else progs[0]
                    df_est=resultado.get("df_acad",resultado.get("df_acad_nc"))
                    if df_est is not None:
                        df_p=df_est[df_est["PROGRAMA"]==prog_pred] if "PROGRAMA" in df_est.columns else df_est
                        if "NUMERO_SEMESTRE" in df_p.columns:
                            sems_raw=sorted(df_p["NUMERO_SEMESTRE"].unique().tolist())
                            st.markdown(f"**Semestres disponibles:** {_formatear_semestres(sems_raw)}")
                        if resultado.get("es_propedeutico"): st.info("ℹ Trayectoria propedéutica unificada.")

                    # ── Modelos Machine Learning ───────────────────────────
                    if tipo_actual == "ml_por_programa":
                        from modelos.modelos_ml_por_programa import predecir_individual_ml
                        ruta_dm2=INTER_DIR/"DF_DEMOGRAFICOS_3_PROPEDEUTICO.csv"
                        if not ruta_dm2.exists(): ruta_dm2=INTER_DIR/"DF_DEMOGRAFICOS_2_RELLENADO.csv"
                        df_dm=pd.read_csv(ruta_dm2) if ruta_dm2.exists() else pd.DataFrame()
                        if len(df_dm): df_dm["ID_EST"]=pd.to_numeric(df_dm["ID_EST"],errors="coerce").fillna(-1).astype(int)
                        df_ed=df_dm[df_dm["ID_EST"]==resultado["id_est"]].head(1) if len(df_dm) else pd.DataFrame()
                        datos_socio=df_ed.iloc[0].to_dict() if len(df_ed) else {}
                        try:
                            pred_res = predecir_individual_ml(info, datos_socio, df_est if df_est is not None else pd.DataFrame())
                            if tarea_actual=="clasificacion_desercion":
                                prob=pred_res.get("probabilidad",0)
                                umb_uso=umbral if umbral else 0.5
                                pred=(1 if prob>=umb_uso else 0)
                                cls="pred-alto" if pred==1 else "pred-bajo"; col_p="#E74C3C" if pred==1 else "#2ECC71"
                                st.markdown(f"""<div class="pred-resultado {cls}"><div class="pred-prob" style="color:{col_p}">{prob*100:.1f}%</div>
                                    <div class="pred-label">{" RIESGO ALTO" if pred==1 else " BAJO RIESGO"}</div>
                                    <div class="pred-nota">{_nombre_completo(modelo_sel)} · Umbral: {umb_uso:.2f}</div></div>""",unsafe_allow_html=True)
                            elif tarea_actual=="regresion_nota":
                                nota=pred_res.get("nota",0)
                                st.metric("Nota proyectada",f"{nota:.2f}")
                                st.info(f"Modelo: {_nombre_completo(modelo_sel)}")
                        except Exception as e_ml:
                            st.error(f"❌ Error en predicción: {e_ml}"); logger.exception("ML pred individual")

                    # ── RNN Multitarea ─────────────────────────────────────
                    elif tipo_actual == "rnn_multi":
                        datos_rnn, modelo_rnn, _safe_prog_actual = _asegurar_rnn_cargada(modelo_sel, info)
                        info_act = obtener_modelos().get(modelo_sel, info)
                        if datos_rnn is None or modelo_rnn is None:
                            st.error("❌ No se pudo cargar el modelo Keras. Reentrena el modelo."); st.stop()

                        if df_est is not None:
                            from modelos.modelo_rnn import predecir_estudiante as _pred_rnn, MAX_SEMESTRES_GLOBAL
                            ruta_dm2=INTER_DIR/"DF_DEMOGRAFICOS_3_PROPEDEUTICO.csv"
                            if not ruta_dm2.exists(): ruta_dm2=INTER_DIR/"DF_DEMOGRAFICOS_2_RELLENADO.csv"
                            if not ruta_dm2.exists(): st.error("❌ Sin datos sociodemográficos."); st.stop()
                            df_dm=pd.read_csv(ruta_dm2)
                            df_dm["ID_EST"]=pd.to_numeric(df_dm["ID_EST"],errors="coerce").fillna(-1).astype(int)
                            df_ed=df_dm[df_dm["ID_EST"]==resultado["id_est"]].head(1)
                            if len(df_ed)==0: st.warning(" Sin datos sociodemográficos. Se usarán valores por defecto.")
                            datos_soc=df_ed.iloc[0].to_dict() if len(df_ed) else {}
                            fac=datos_rnn["features_acad_cols"]; max_sem=datos_rnn.get("max_semestres",MAX_SEMESTRES_GLOBAL)
                            fsoc=_inferir_features_socio_rnn(datos_rnn, _safe_prog_modelo(modelo_sel, info_act))
                            try:
                                prob,nota,sem=_pred_rnn(modelo_rnn,datos_soc,df_est,max_sem,fac,features_socio_cols=fsoc)
                                umbral_pct=umbral*100.0 if umbral else 50.0
                                es_alto=prob>=umbral_pct
                                cls="pred-alto" if es_alto else "pred-bajo"; col_p="#E74C3C" if es_alto else "#2ECC71"
                                st.markdown(f"""<div class="pred-resultado {cls}">
                                    <div class="pred-prob" style="color:{col_p}">{prob:.1f}%</div>
                                    <div class="pred-label">{" RIESGO ALTO" if es_alto else " BAJO RIESGO"}</div>
                                    <div class="pred-nota">Nota proyectada semestre {sem}: <strong>{nota:.2f}</strong>
                                    &nbsp;·&nbsp; Umbral: {umbral:.2f} ({umbral_pct:.0f}%)</div></div>""",unsafe_allow_html=True)
                            except Exception as e_pred:
                                st.error(f"❌ Error predicción RNN: {e_pred}"); logger.exception("RNN pred")
                        else:
                            st.error("❌ No se encontraron datos académicos para este ID_EST.")

    with tab_manual:
        st.markdown("### Predicción con datos manuales")
        st.caption("Ingresa variables sociodemográficas y académicas sin buscar un ID_EST existente.")
        safe_prog_manual = _safe_prog_modelo(modelo_sel, info)

        if tipo_actual == "rnn_multi":
            from modelos.modelo_rnn import predecir_estudiante as _pred_rnn, MAX_SEMESTRES_GLOBAL
            datos_rnn, modelo_rnn, safe_prog_manual = _asegurar_rnn_cargada(modelo_sel, info)
            if datos_rnn is None or modelo_rnn is None:
                st.error("❌ No se pudo cargar el modelo RNN seleccionado.")
            else:
                features_socio = _inferir_features_socio_rnn(datos_rnn, safe_prog_manual)
                features_acad = datos_rnn.get("features_acad_cols", [])
                max_sem = int(datos_rnn.get("max_semestres", MAX_SEMESTRES_GLOBAL))
                with st.form("form_pred_manual_rnn"):
                    st.markdown("**Variables sociodemográficas**")
                    datos_soc = {}
                    cols = st.columns(3)
                    for i, f in enumerate(features_socio):
                        datos_soc[f] = cols[i % 3].number_input(f, value=0.0, step=1.0, format="%.4f", key=f"manual_rnn_socio_{f}")
                    n_sem = st.slider("Semestres registrados", 1, max_sem, min(4, max_sem), key="manual_rnn_n_sem")
                    filas_acad = []
                    st.markdown("**Historia académica por semestre**")
                    for sem_i in range(1, n_sem + 1):
                        with st.expander(f"Semestre {sem_i}", expanded=(sem_i == 1)):
                            fila = {"NUMERO_SEMESTRE": sem_i, "PROGRAMA": safe_prog_manual, "ID_EST": -1}
                            cols_a = st.columns(3)
                            for j, f in enumerate(features_acad):
                                fila[f] = cols_a[j % 3].number_input(f"{f} · S{sem_i}", value=0.0, step=0.1, format="%.4f", key=f"manual_rnn_acad_{sem_i}_{f}")
                            filas_acad.append(fila)
                    ejecutar_manual = st.form_submit_button("Predecir con RNN")
                if ejecutar_manual:
                    try:
                        hist_df = pd.DataFrame(filas_acad)
                        prob, nota, sem = _pred_rnn(modelo_rnn, datos_soc, hist_df, max_sem, features_acad, features_socio_cols=features_socio)
                        _render_prediccion(prob, nota, sem, umbral if umbral else 0.5, _nombre_completo(modelo_sel))
                    except Exception as e:
                        st.error(f"❌ Error en predicción manual RNN: {e}")
                        logger.exception("RNN pred manual")

        elif tipo_actual == "ml_por_programa":
            from modelos.modelos_ml_por_programa import predecir_individual_ml
            features = info.get("features", [])
            with st.form("form_pred_manual_ml"):
                st.markdown("**Variables requeridas por el modelo**")
                valores = {}
                cols = st.columns(3)
                for i, f in enumerate(features):
                    valores[f] = cols[i % 3].number_input(f, value=0.0, step=0.1, format="%.4f", key=f"manual_ml_{f}")
                ejecutar_manual_ml = st.form_submit_button("Predecir con modelo ML")
            if ejecutar_manual_ml:
                try:
                    if tarea_actual == "clasificacion_desercion":
                        pred_res = predecir_individual_ml(info, valores, pd.DataFrame())
                        prob = float(pred_res.get("probabilidad", 0))
                        _render_prediccion(prob * 100, None, None, umbral if umbral else 0.5, _nombre_completo(modelo_sel))
                    elif tarea_actual == "regresion_nota":
                        df_manual = pd.DataFrame([valores])
                        pred_res = predecir_individual_ml(info, {}, df_manual)
                        st.metric("Nota proyectada", f"{float(pred_res.get('nota', 0)):.2f}")
                        st.info(f"Modelo: {_nombre_completo(modelo_sel)}")
                    else:
                        st.error("❌ Tarea de modelo ML no reconocida.")
                except Exception as e:
                    st.error(f"❌ Error en predicción manual ML: {e}")
                    logger.exception("ML pred manual")
        else:
            st.info("ℹ Selecciona un modelo RNN Multitarea o un Modelo Machine Learning.")

    with tab_masivo:
        st.markdown("### Predicción masiva")
        st.caption(
            "Genera predicciones para varios estudiantes. Puedes usar los datasets internos del proyecto "
            "o cargar CSV externos, por ejemplo los no clasificables exportados desde el Pipeline."
        )
        if tipo_actual in ("rnn_multi", "ml_por_programa"):
            safe_prog_masivo = _safe_prog_modelo(modelo_sel, info)
            st.info(
                f"Modelo seleccionado para el programa: **{safe_prog_masivo.replace('_',' ').title()}**. "
                "Si cargas archivos con varios programas, solo se usarán las filas que coincidan con este programa."
            )

            origen_pred = st.radio(
                "Origen de datos para predicción masiva",
                ["Usar datasets internos del sistema", "Cargar CSV académico y demográfico"],
                horizontal=True,
                key="origen_pred_masiva",
            )

            df_acad_cargado = None
            df_demo_cargado = None
            origen_txt = "datasets internos"

            if origen_pred == "Cargar CSV académico y demográfico":
                c_up1, c_up2 = st.columns(2)
                with c_up1:
                    f_acad_pred = st.file_uploader(
                        "CSV académico no clasificable / predicción",
                        type=["csv"],
                        key="upload_pred_masiva_acad",
                        help="Ejemplo: DF_*_ACADEMICOS_NO_CLASIFICABLES_PREDICCION.csv o DF_NO_CLASIFICABLES_PREDICCION.csv",
                    )
                with c_up2:
                    f_demo_pred = st.file_uploader(
                        "CSV demográfico no clasificable",
                        type=["csv"],
                        key="upload_pred_masiva_demo",
                        help="Ejemplo: DF_*_DEMOGRAFICOS_NO_CLASIFICABLES.csv o DF_NO_CLASIFICABLES_DEMOGRAFICOS.csv",
                    )

                if f_acad_pred is not None and f_demo_pred is not None:
                    try:
                        df_acad_cargado = pd.read_csv(f_acad_pred)
                        df_demo_cargado = pd.read_csv(f_demo_pred)
                        origen_txt = "CSV cargados por usuario"

                        faltantes = []
                        for nom_df, df_tmp in [("académico", df_acad_cargado), ("demográfico", df_demo_cargado)]:
                            if "ID_EST" not in df_tmp.columns:
                                faltantes.append(f"ID_EST en archivo {nom_df}")
                        if faltantes:
                            st.error("❌ Faltan columnas obligatorias: " + ", ".join(faltantes))
                        else:
                            st.success(
                                f"Archivos cargados: {len(df_acad_cargado):,} filas académicas y "
                                f"{len(df_demo_cargado):,} filas demográficas."
                            )
                            if "PROGRAMA" not in df_acad_cargado.columns or "PROGRAMA" not in df_demo_cargado.columns:
                                st.warning(
                                    "Uno de los archivos no tiene columna PROGRAMA. Se intentará predecir por ID_EST, "
                                    "pero se recomienda conservar PROGRAMA para modelos por carrera."
                                )
                    except Exception as e_up:
                        st.error(f"❌ Error leyendo archivos cargados: {e_up}")
                        df_acad_cargado = None
                        df_demo_cargado = None
                else:
                    st.info("Carga ambos archivos CSV para habilitar la predicción masiva externa.")

            puede_predecir = origen_pred == "Usar datasets internos del sistema" or (
                df_acad_cargado is not None and df_demo_cargado is not None
            )

            if st.button("Generar predicción masiva", key="btn_pred_masiva", disabled=not puede_predecir):
                try:
                    with st.spinner("Generando predicciones masivas…"):
                        if origen_pred == "Cargar CSV académico y demográfico":
                            df_pred = _generar_predicciones_masivas(
                                modelo_sel, info, tipo_actual, tarea_actual, umbral if umbral else 0.5,
                                df_demo_input=df_demo_cargado,
                                df_acad_input=df_acad_cargado,
                                origen_datos=origen_txt,
                            )
                        else:
                            df_pred = _generar_predicciones_masivas(
                                modelo_sel, info, tipo_actual, tarea_actual, umbral if umbral else 0.5,
                                origen_datos=origen_txt,
                            )
                    if df_pred.empty:
                        st.warning("No se generaron predicciones. Verifica que existan estudiantes del programa seleccionado en los archivos.")
                    else:
                        st.success(f"✅ Predicción masiva generada: {len(df_pred):,} filas")
                        if "ERROR" in df_pred.columns and df_pred["ERROR"].notna().any():
                            st.warning("Algunas filas tuvieron error. Revisa la columna ERROR en la tabla descargable.")
                        st.dataframe(df_pred, use_container_width=True, hide_index=True)
                        nombre_archivo = f"prediccion_masiva_{_nombre_archivo_seguro(modelo_sel)}.csv"
                        st.download_button(
                            "⬇ Descargar predicción masiva (CSV)",
                            data=df_pred.to_csv(index=False).encode("utf-8"),
                            file_name=nombre_archivo,
                            mime="text/csv",
                            use_container_width=True,
                        )
                except Exception as e:
                    st.error(f"❌ Error en predicción masiva: {e}")
                    logger.exception("Pred masiva")
        else:
            st.info("ℹ Selecciona un modelo RNN Multitarea o un Modelo Machine Learning para realizar predicciones.")
