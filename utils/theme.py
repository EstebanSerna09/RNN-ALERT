"""
utils/theme.py
==============
Identidad visual institucional UNIMAYOR — Paleta azul institucional.
Basada en el logo oficial y la presentación del dashboard.

Color principal: #003B73 (azul UNIMAYOR)
"""
from __future__ import annotations
import base64
from pathlib import Path
import streamlit as st

# ── Paleta UNIMAYOR ────────────────────────────────────────────────────────────
C_PRIMARIO   = "#003B73"
C_SECUNDARIO = "#0056A6"
C_APOYO      = "#0074D9"
C_FONDO      = "#0A192F"
C_FONDO2     = "#112240"
C_CARD       = "#172A45"
C_CARD2      = "#1E3A5F"
C_TEXTO      = "#FFFFFF"
C_TEXTO2     = "#B8C1CC"
C_EXITO      = "#2ECC71"
C_ALERTA     = "#F39C12"
C_CRITICO    = "#E74C3C"
C_BORDE      = "#1E3A5F"


def get_logo_b64() -> str:
    """Carga el logo UNIMAYOR en base64."""
    ruta = Path(__file__).parent.parent / "assets" / "logo_unimayor.webp"
    if ruta.exists():
        with open(ruta, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


def inyectar_css() -> None:
    """Inyecta el CSS institucional completo."""
    st.markdown(f"""<style>
/* ════════════════════════════════════════════════════════════
   RNN-ALERT · Identidad Visual UNIMAYOR
   ════════════════════════════════════════════════════════════ */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {{ box-sizing: border-box; }}

.stApp {{
    background: {C_FONDO};
    font-family: 'Inter', 'Segoe UI', sans-serif;
    color: {C_TEXTO};
}}

/* ── SIDEBAR ─────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0A192F 0%, #061525 100%);
    border-right: 1px solid {C_BORDE};
    width: 260px !important;
}}
section[data-testid="stSidebar"] .stRadio > label {{
    display: none;
}}
section[data-testid="stSidebar"] * {{
    color: {C_TEXTO2} !important;
}}
section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] {{
    background: rgba(0,116,217,0.08) !important;
    border-radius: 8px;
    margin: 2px 0;
    padding: 8px 12px;
    border: 1px solid transparent;
    transition: all 0.2s;
}}
section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"]:hover {{
    background: rgba(0,116,217,0.18) !important;
    border-color: {C_APOYO} !important;
}}

/* ── HEADER DE PÁGINA ────────────────────────────────── */
.page-header {{
    background: linear-gradient(135deg, {C_CARD2} 0%, {C_PRIMARIO} 100%);
    border-left: 4px solid {C_APOYO};
    padding: 1.4rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.8rem;
    box-shadow: 0 4px 24px rgba(0,59,115,0.35);
}}
.page-header h1 {{
    color: {C_TEXTO};
    margin: 0 0 4px;
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.3px;
}}
.page-header p {{
    color: {C_TEXTO2};
    margin: 0;
    font-size: 0.85rem;
    font-weight: 400;
}}

/* ── KPI CARDS ───────────────────────────────────────── */
.kpi-card {{
    background: {C_CARD};
    border-radius: 12px;
    padding: 1.3rem 1.4rem;
    border-left: 4px solid {C_APOYO};
    box-shadow: 0 2px 12px rgba(0,0,0,0.25);
    margin-bottom: 1rem;
    transition: transform 0.2s, box-shadow 0.2s;
    border: 1px solid {C_BORDE};
}}
.kpi-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,116,217,0.2);
}}
.kpi-label {{
    font-size: 0.7rem;
    color: {C_TEXTO2};
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
    font-weight: 500;
}}
.kpi-value {{
    font-size: 2.1rem;
    font-weight: 700;
    color: {C_EXITO};
    line-height: 1;
    font-variant-numeric: tabular-nums;
}}
.kpi-sub {{
    font-size: 0.75rem;
    color: {C_TEXTO2};
    margin-top: 5px;
    font-weight: 400;
}}
.kpi-card.rojo  .kpi-value {{ color: {C_CRITICO}; }}
.kpi-card.rojo              {{ border-left-color: {C_CRITICO}; }}
.kpi-card.azul  .kpi-value {{ color: {C_APOYO}; }}
.kpi-card.azul              {{ border-left-color: {C_APOYO}; }}
.kpi-card.naranja .kpi-value {{ color: {C_ALERTA}; }}
.kpi-card.naranja            {{ border-left-color: {C_ALERTA}; }}
.kpi-card.blanco  .kpi-value {{ color: {C_TEXTO}; }}
.kpi-card.blanco             {{ border-left-color: {C_TEXTO2}; }}

/* ── BOTÓN PRINCIPAL ─────────────────────────────────── */
div.stButton > button {{
    background: linear-gradient(135deg, {C_PRIMARIO} 0%, {C_APOYO} 100%);
    color: white !important;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 0.65rem 1.8rem;
    width: 100%;
    font-size: 0.93rem;
    letter-spacing: 0.02em;
    transition: all 0.25s;
    box-shadow: 0 2px 8px rgba(0,116,217,0.25);
}}
div.stButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(0,116,217,0.35);
    filter: brightness(1.1);
}}
div.stButton > button:active {{
    transform: translateY(0);
}}

/* ── LOG BOX ─────────────────────────────────────────── */
.log-box {{
    background: #061525;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    font-size: 0.8rem;
    max-height: 400px;
    overflow-y: auto;
    color: #A8B2C1;
    border: 1px solid {C_BORDE};
}}
.log-box::-webkit-scrollbar {{ width: 4px; }}
.log-box::-webkit-scrollbar-track {{ background: {C_FONDO}; }}
.log-box::-webkit-scrollbar-thumb {{ background: {C_PRIMARIO}; border-radius: 4px; }}
.log-ok   {{ color: {C_EXITO}; }}
.log-err  {{ color: {C_CRITICO}; }}
.log-info {{ color: {C_APOYO}; }}

/* ── SECCIÓN TÍTULO ──────────────────────────────────── */
.seccion-titulo {{
    font-size: 1rem;
    font-weight: 600;
    color: {C_APOYO};
    border-bottom: 1px solid {C_BORDE};
    padding-bottom: 8px;
    margin: 1.5rem 0 1rem;
    letter-spacing: 0.02em;
}}

/* ── TABLA COMPARATIVA ───────────────────────────────── */
.tabla-comp {{ width:100%; border-collapse:collapse; font-size:0.84rem; }}
.tabla-comp th {{
    background: {C_PRIMARIO};
    color: white;
    padding: 9px 14px;
    text-align: left;
    font-weight: 600;
    letter-spacing: 0.03em;
    font-size: 0.78rem;
    text-transform: uppercase;
}}
.tabla-comp th:first-child {{ border-radius: 8px 0 0 0; }}
.tabla-comp th:last-child  {{ border-radius: 0 8px 0 0; }}
.tabla-comp td {{
    padding: 8px 14px;
    border-bottom: 1px solid {C_BORDE};
    color: {C_TEXTO2};
}}
.tabla-comp tr:hover td {{ background: {C_CARD2}; color: {C_TEXTO}; }}
.tabla-comp .mejor {{ font-weight: 700; color: {C_EXITO} !important; }}

/* ── BADGE ───────────────────────────────────────────── */
.badge-ok   {{ background: rgba(46,204,113,0.15); color:{C_EXITO}; padding:3px 12px; border-radius:20px; font-size:0.76rem; font-weight:600; border:1px solid {C_EXITO}; }}
.badge-err  {{ background: rgba(231,76,60,0.15);  color:{C_CRITICO}; padding:3px 12px; border-radius:20px; font-size:0.76rem; font-weight:600; border:1px solid {C_CRITICO}; }}
.badge-warn {{ background: rgba(243,156,18,0.15); color:{C_ALERTA}; padding:3px 12px; border-radius:20px; font-size:0.76rem; font-weight:600; border:1px solid {C_ALERTA}; }}
.badge-info {{ background: rgba(0,116,217,0.15);  color:{C_APOYO}; padding:3px 12px; border-radius:20px; font-size:0.76rem; font-weight:600; border:1px solid {C_APOYO}; }}

/* ── ARCH CARD ───────────────────────────────────────── */
.arch-card {{
    background: {C_CARD};
    border: 1px solid {C_BORDE};
    border-left: 3px solid {C_APOYO};
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.6rem;
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    font-size: 0.8rem;
    color: {C_TEXTO2};
}}
.arch-card b {{ color: {C_TEXTO}; }}
.arch-arrow {{
    text-align: center;
    color: {C_APOYO};
    font-size: 1.2rem;
    margin: 2px 0;
}}

/* ── SEPARADORES ─────────────────────────────────────── */
hr {{ border-color: {C_BORDE}; margin: 1.2rem 0; }}

/* ── INPUTS ──────────────────────────────────────────── */
.stTextInput input, .stSelectbox select, .stNumberInput input {{
    background: {C_CARD} !important;
    border: 1px solid {C_BORDE} !important;
    color: {C_TEXTO} !important;
    border-radius: 8px !important;
}}
.stSelectbox > div[data-baseweb] {{
    background: {C_CARD} !important;
    border-color: {C_BORDE} !important;
}}
.stSlider [data-baseweb="slider"] {{
    background: {C_BORDE};
}}

/* ── TABS ────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background: {C_CARD};
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid {C_BORDE};
}}
.stTabs [data-baseweb="tab"] {{
    color: {C_TEXTO2} !important;
    border-radius: 7px;
    padding: 6px 16px;
    font-size: 0.85rem;
    font-weight: 500;
}}
.stTabs [aria-selected="true"] {{
    background: {C_PRIMARIO} !important;
    color: white !important;
}}

/* ── EXPANDER ────────────────────────────────────────── */
.streamlit-expanderHeader {{
    background: {C_CARD} !important;
    border-radius: 8px !important;
    border: 1px solid {C_BORDE} !important;
    color: {C_TEXTO2} !important;
}}
.streamlit-expanderContent {{
    background: {C_CARD} !important;
    border: 1px solid {C_BORDE} !important;
    border-top: none !important;
}}

/* ── MÉTRICAS NATIVAS ────────────────────────────────── */
[data-testid="stMetricValue"] {{
    color: {C_EXITO} !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}}
[data-testid="stMetricLabel"] {{
    color: {C_TEXTO2} !important;
    font-size: 0.78rem !important;
}}

/* ── PROGRESS BAR ────────────────────────────────────── */
.stProgress > div > div > div {{
    background: linear-gradient(90deg, {C_PRIMARIO}, {C_APOYO});
    border-radius: 4px;
}}

/* ── INFO / WARNING / ERROR ──────────────────────────── */
.stAlert {{
    border-radius: 10px !important;
    border: 1px solid {C_BORDE} !important;
}}

/* ── LOGO SIDEBAR ────────────────────────────────────── */
.logo-container {{
    text-align: center;
    padding: 1.2rem 1rem 0.6rem;
    border-bottom: 1px solid {C_BORDE};
    margin-bottom: 0.8rem;
}}
.logo-container img {{
    max-width: 120px;
    filter: brightness(0) invert(1);
    opacity: 0.9;
}}
.logo-title {{
    font-size: 1.15rem;
    font-weight: 700;
    color: white !important;
    margin-top: 8px;
    letter-spacing: 0.5px;
}}
.logo-sub {{
    font-size: 0.68rem;
    color: {C_TEXTO2} !important;
    margin-top: 2px;
}}

/* ── RESULTADO PREDICCIÓN ────────────────────────────── */
.pred-resultado {{
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1rem 0;
    border: 1px solid;
}}
.pred-alto {{
    background: rgba(231,76,60,0.12);
    border-color: {C_CRITICO};
}}
.pred-bajo {{
    background: rgba(46,204,113,0.12);
    border-color: {C_EXITO};
}}
.pred-prob {{
    font-size: 3rem;
    font-weight: 800;
    line-height: 1;
    font-variant-numeric: tabular-nums;
}}
.pred-label {{
    font-size: 1.1rem;
    font-weight: 600;
    margin-top: 6px;
}}
.pred-nota {{
    font-size: 0.9rem;
    color: {C_TEXTO2};
    margin-top: 4px;
}}

/* ── NOTA MULTIMODAL ─────────────────────────────────── */
.nota-card {{
    background: {C_CARD2};
    border-radius: 10px;
    padding: 1rem 1.2rem;
    border: 1px solid {C_BORDE};
    text-align: center;
    margin-top: 0.8rem;
}}
.nota-valor {{
    font-size: 2.4rem;
    font-weight: 700;
    color: {C_ALERTA};
}}
.nota-label {{
    font-size: 0.8rem;
    color: {C_TEXTO2};
    margin-top: 3px;
}}
</style>""", unsafe_allow_html=True)


def sidebar_logo() -> None:
    """Renderiza el logo UNIMAYOR en la sidebar."""
    b64 = get_logo_b64()
    if b64:
        st.sidebar.markdown(
            f"""<div class="logo-container">
                <img src="data:image/webp;base64,{b64}" alt="UNIMAYOR"/>
                <div class="logo-title">RNN Alert</div>
                <div class="logo-sub">UNIMAYOR · 2026</div>
            </div>""",
            unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown(
            """<div class="logo-container">
                <div style="font-size:2rem">🎓</div>
                <div class="logo-title">RNN Alert</div>
                <div class="logo-sub">UNIMAYOR · 2026</div>
            </div>""",
            unsafe_allow_html=True
        )


def header(titulo: str, subtitulo: str = "") -> None:
    st.markdown(
        f'<div class="page-header">'
        f'<h1>{titulo}</h1>'
        f'{"<p>" + subtitulo + "</p>" if subtitulo else ""}'
        f'</div>',
        unsafe_allow_html=True
    )


def kpi(label: str, valor: str, subtexto: str = "", variante: str = "") -> str:
    cls = f"kpi-card {variante}".strip()
    return (
        f'<div class="{cls}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{valor}</div>'
        f'{"<div class=kpi-sub>" + subtexto + "</div>" if subtexto else ""}'
        f'</div>'
    )


def badge(texto: str, tipo: str = "ok") -> str:
    return f'<span class="badge-{tipo}">{texto}</span>'


def arch_card(nombre: str, desc: str) -> str:
    return (
        f'<div class="arch-card"><b>{nombre}</b><br>'
        f'<span style="opacity:0.65;font-size:0.75rem">{desc}</span></div>'
    )


def arch_arrow() -> str:
    return '<div class="arch-arrow">↓</div>'
