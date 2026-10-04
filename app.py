"""ManyKGs - Height -> Weight Regression dashboard.

Serves the exported artifacts only (models/best_model.pkl + models/scaler.pkl).
Nothing is trained or simulated here.
"""
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn
import streamlit as st

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "best_model.pkl"
SCALER_PATH = BASE / "models" / "scaler.pkl"
H_MIN, H_MAX, H_DEFAULT = 140, 220, 180
LIME, CYAN = "#B6FF3B", "#4CC9F0"

st.set_page_config(page_title="ManyKGs", page_icon="ðŸ“", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_artifacts():
    """Load model + scaler. Returns (model, scaler, error_message)."""
    for p, label in ((MODEL_PATH, "Model"), (SCALER_PATH, "Scaler")):
        if not p.exists():
            return None, None, f"{label} file not found: `{p.relative_to(BASE)}`"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # sklearn version-mismatch notice
            return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH), None
    except Exception as e:  # corrupt / incompatible pickle
        return None, None, f"Could not load pickle files ({type(e).__name__}: {e})"


def predict_weight(height_cm, model, scaler):
    sample = np.array([[height_cm]])
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)
    return float(np.asarray(prediction).ravel()[0])


@st.cache_data
def response_curve(_model, _scaler, key):
    """Deployed pipeline evaluated over the height range. `key` busts cache on model change."""
    xs = np.linspace(H_MIN, H_MAX, 100)
    ys = np.asarray(_model.predict(_scaler.transform(xs.reshape(-1, 1)))).ravel()
    return xs, ys


DATA_PATH = BASE / "models_notebook" / "SOCR-HeightWeight.csv"


@st.cache_data
def load_sample(n=1500):
    """Optional real-data scatter (notebook conversions). Returns None if the CSV is absent."""
    if not DATA_PATH.exists():
        return None
    try:
        df = pd.read_csv(DATA_PATH, encoding="utf-8-sig").sample(n, random_state=42)
        return df["Height(Inches)"] * 2.54, df["Weight(Pounds)"] * 0.45359237
    except Exception:
        return None


model, scaler, load_error = load_artifacts()
if load_error:
    st.error(f"{load_error}\n\nExpected `models/best_model.pkl` and `models/scaler.pkl` next to `app.py`.")
    st.stop()

MODEL_NAME = type(model).__name__
SCALER_NAME = type(scaler).__name__
N_FEATURES = getattr(model, "n_features_in_", getattr(scaler, "n_features_in_", "n/a"))

# ---------------------------------------------------------------- state
st.session_state.setdefault("height", H_DEFAULT)
st.session_state.setdefault("result", None)  # (height, kg) of last prediction

# ---------------------------------------------------------------- CSS
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&family=JetBrains+Mono:wght@500&family=Caveat:wght@700&display=swap');
:root{--bg:#08090D;--card:#11141B;--bd:rgba(255,255,255,.08);--lime:#B6FF3B;--cyan:#4CC9F0;--tx:#F2F4F8;--mu:#8B93A3}
html,body,[class*="css"],.stApp{font-family:'Inter',sans-serif;color:var(--tx)}
.stApp{background:radial-gradient(1200px 600px at 85% -10%,rgba(76,201,240,.12),transparent 60%),radial-gradient(900px 500px at -10% 10%,rgba(182,255,59,.08),transparent 55%),var(--bg)}
header[data-testid="stHeader"]{background:transparent}
.block-container{padding:1rem 2.5vw 0;max-width:100%}
.stApp{font-size:14px}
h1,h2,h3{letter-spacing:-.02em}
section[data-testid="stSidebar"]{background:#0B0D12;border-right:1px solid var(--bd)}
.brand{font-weight:800;font-size:1.5rem;letter-spacing:-.03em}.brand b{color:var(--lime)}
.status{display:flex;align-items:center;gap:.6rem;padding:.55rem .8rem;margin:.4rem 0;background:var(--card);border:1px solid var(--bd);border-radius:12px;font-size:.85rem}
.dot{width:8px;height:8px;border-radius:50%;background:var(--lime);box-shadow:0 0 10px var(--lime);animation:pulse 2.4s ease-in-out infinite}
.status small{display:block;color:var(--mu);font-family:'JetBrains Mono',monospace;font-size:.72rem}
.glass{padding:1rem!important;background:linear-gradient(145deg,rgba(255,255,255,.045),rgba(255,255,255,.01)),var(--card);border:1px solid var(--bd);border-radius:20px;padding:1.5rem;box-shadow:0 20px 50px -25px rgba(0,0,0,.8);backdrop-filter:blur(10px);animation:fadeUp .6s ease both}
.glass h3{margin:0 0 .6rem;font-size:1.1rem}
.mu{color:var(--mu)}
.eyebrow{font:500 .72rem 'JetBrains Mono',monospace;letter-spacing:.14em;text-transform:uppercase;color:var(--mu)}
.badge{display:inline-flex;gap:.5rem;align-items:center;padding:.3rem .8rem;border:1px solid rgba(182,255,59,.35);border-radius:99px;color:var(--lime);font:500 .72rem 'JetBrains Mono',monospace;letter-spacing:.12em;background:rgba(182,255,59,.06)}
.badge i{width:7px;height:7px;border-radius:50%;background:var(--lime);animation:pulse 2s infinite}
.hero{display:grid;grid-template-columns:1.5fr 1fr;gap:2rem;align-items:center;padding:2.5rem;border-radius:28px;border:1px solid var(--bd);background:linear-gradient(135deg,rgba(255,255,255,.05),rgba(255,255,255,0));animation:fadeUp .7s ease both}
.hero h1{font-size:clamp(2.6rem,7vw,5rem);font-weight:800;margin:.8rem 0 0;line-height:.95;background:linear-gradient(100deg,#fff 0%,var(--lime) 45%,var(--cyan) 100%);background-size:200% 100%;-webkit-background-clip:text;background-clip:text;color:transparent;animation:shift 8s ease-in-out infinite alternate}
.hero h2{font-size:clamp(1.1rem,2.4vw,1.6rem);font-weight:500;margin:.6rem 0;color:var(--tx)}
.hero p{color:var(--mu);max-width:34rem;margin:0}
.viz{position:relative;height:230px;display:flex;justify-content:center;align-items:center}
.ruler{position:relative;width:64px;height:210px;border-radius:10px;border:1px solid var(--bd);background:repeating-linear-gradient(to bottom,rgba(182,255,59,.55) 0 1px,transparent 1px 10px);box-shadow:0 0 40px rgba(182,255,59,.15);animation:glow 4s ease-in-out infinite}
.ruler:before{content:"";position:absolute;left:-14px;right:-14px;top:30%;height:2px;background:var(--lime);box-shadow:0 0 14px var(--lime);animation:scan 5s ease-in-out infinite alternate}
.tag{position:absolute;right:6%;top:18%;padding:.6rem .9rem;border-radius:14px;background:rgba(17,20,27,.9);border:1px solid rgba(76,201,240,.4);font:500 .95rem 'JetBrains Mono',monospace;color:var(--cyan);animation:float 5s ease-in-out infinite}
.tag.h{right:auto;left:6%;top:auto;bottom:14%;color:var(--lime);border-color:rgba(182,255,59,.4);animation-delay:-2s}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:1rem;margin:1.2rem 0}
.kpi{padding:1.1rem 1.3rem}.kpi .v{font-size:1.7rem;font-weight:800;margin-top:.3rem;word-break:break-word}
.kpi.a .v{color:var(--lime)}.kpi.b .v{color:var(--cyan)}
.result{text-align:center;animation:fadeUp .7s ease both;border-color:rgba(182,255,59,.35);box-shadow:0 0 60px -20px rgba(182,255,59,.35)}
.result .big{font-size:clamp(2.4rem,5vw,3.4rem);font-weight:800;color:var(--lime);line-height:1;text-shadow:0 0 30px rgba(182,255,59,.4)}
.result .row{display:flex;justify-content:center;gap:2.5rem;margin-top:1rem;flex-wrap:wrap}
.pipe{display:flex;align-items:stretch;gap:.5rem;flex-wrap:nowrap}
.node{flex:1 1 0;min-width:0;text-align:center;padding:.8rem .5rem;border-radius:16px;background:var(--card);border:1px solid var(--bd);animation:fadeUp .6s ease both}
.node .v{font-weight:700;font-size:.85rem;margin-top:.25rem;color:var(--mu)}.node .eyebrow{font-size:.62rem;letter-spacing:.08em;overflow:hidden;text-overflow:ellipsis}
.arrow{align-self:center;color:var(--lime);font-size:1.6rem;animation:slide 1.8s ease-in-out infinite}
.sticky{position:relative;margin-top:1.2rem;transform:rotate(-2deg);padding:2.2rem 1.6rem 1.5rem;color:#2b2a22;background:linear-gradient(transparent 94%,rgba(0,0,0,.06) 94%) 0 0/100% 1.7rem,linear-gradient(160deg,#fff3a3,#ffe66d);border-radius:3px 3px 22px 3px;box-shadow:0 18px 30px -12px rgba(0,0,0,.7),inset 0 -20px 30px -25px rgba(0,0,0,.3);animation:sway 7s ease-in-out infinite alternate;font-size:.92rem;line-height:1.7}
.sticky:before{content:"";position:absolute;top:-12px;left:50%;width:90px;height:26px;margin-left:-45px;background:rgba(255,255,255,.55);transform:rotate(3deg);box-shadow:0 1px 4px rgba(0,0,0,.25)}
.sticky h4{font:700 1.7rem 'Caveat','Segoe Print',cursive;margin:0 0 .4rem;color:#7a2e00}
.sticky ul{margin:.2rem 0 .6rem 1.1rem;padding:0}.sticky b{color:#000}
.sticky code{background:rgba(0,0,0,.08);padding:1px 6px;border-radius:4px;color:#000}
.chip{display:inline-block;padding:.5rem .9rem;margin:.25rem .25rem 0 0;border-radius:12px;background:var(--card);border:1px solid var(--bd)}
.chip.on{border-color:var(--lime);color:var(--lime);box-shadow:0 0 20px -6px var(--lime)}
.chip small{display:block;color:var(--mu);font-size:.72rem}
.mono{font-family:'JetBrains Mono',monospace}
.foot{margin:3rem 0 1rem;padding-top:1.2rem;border-top:1px solid var(--bd);text-align:center;color:var(--mu);font-size:.8rem}
a{color:var(--cyan)!important}
/* widgets */
div[data-testid="stNumberInput"] input{background:#0B0D12;border:1px solid var(--bd);color:var(--tx);font:500 1.4rem 'JetBrains Mono',monospace;border-radius:12px}
div[data-testid="stNumberInput"] > div > div{background:#0B0D12;border-radius:12px}
div[data-testid="stNumberInput"] label p{color:var(--mu);font-size:.8rem;letter-spacing:.1em;text-transform:uppercase}
.stButton>button{width:100%;border:0;border-radius:12px;padding:.8rem 1rem;font-weight:800;letter-spacing:.1em;color:#08090D;background:linear-gradient(100deg,var(--lime),#7dff9a);box-shadow:0 10px 30px -10px rgba(182,255,59,.6);transition:transform .2s,box-shadow .2s}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 14px 34px -8px rgba(182,255,59,.8);color:#08090D}
.stButton>button p{font-weight:800}
div[role="radiogroup"] label{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:.5rem .8rem;margin-bottom:.35rem;transition:border-color .2s}
div[role="radiogroup"] label:hover{border-color:rgba(182,255,59,.5)}
@keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:none}}
@keyframes float{50%{transform:translateY(-8px)}}
@keyframes pulse{50%{opacity:.35}}
@keyframes glow{50%{box-shadow:0 0 70px rgba(182,255,59,.3)}}
@keyframes shift{to{background-position:100% 0}}
@keyframes scan{to{top:72%}}
@keyframes slide{50%{transform:translateX(5px);opacity:.6}}
@keyframes sway{to{transform:rotate(-1deg)}}
@media (max-width:820px){.pipe{flex-wrap:wrap}.arrow{display:none}}
@media (prefers-reduced-motion:reduce){*{animation:none!important}}
</style>
""",
    unsafe_allow_html=True,
)


def html(s):
    """Render HTML; strip indentation so Markdown never treats it as a code block."""
    st.markdown("\n".join(l.strip() for l in s.splitlines() if l.strip()), unsafe_allow_html=True)


def kpi(label, value, cls=""):
    return f'<div class="glass kpi {cls}"><div class="eyebrow">{label}</div><div class="v">{value}</div></div>'


def style_fig(fig, h=380):
    fig.update_layout(
        height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F2F4F8", family="Inter"), margin=dict(l=10, r=10, t=30, b=10),
        hoverlabel=dict(bgcolor="#11141B", font_color="#F2F4F8"),
    )
    return fig


# ---------------------------------------------------------------- page
html('<div class="brand">Many<b>KGs</b> <span class="mu" style="font-size:.85rem;font-weight:500;margin-left:.4rem">Height → Weight Regression</span></div>')

left, right = st.columns([1, 2], gap="large")
with left:
    h = st.slider("Height (cm)", H_MIN, H_MAX, key="height")
    kg = predict_weight(h, model, scaler)  # height -> scaler.transform -> model.predict
    html(f"""
    <div class="glass result"><div class="eyebrow">Predicted Weight</div>
    <div class="big">{kg:.1f} kg</div>
    <div class="mu" style="font-size:.72rem;margin-top:.6rem">Model estimate for {h} cm · not medical advice</div></div>
    <div class="sticky"><h4>Model Note</h4>
    <b>Deployed model:</b> <code>best_model.pkl</code><br>
    <b>Scaler:</b> <code>scaler.pkl</code><br>
    <b>Dataset:</b> SOCR Height/Weight<br>
    <b>Compared:</b> Linear, Ridge, Lasso, Decision Tree, Random Forest<br>
    <span style="font-size:.78rem">1 in = 2.54 cm · 1 lb = 0.45359237 kg ·
    <a href="https://www.kaggle.com/datasets/burnoutminer/heights-and-weights-dataset" target="_blank" style="color:#7a2e00!important">Kaggle</a></span></div>""")

with right:
    xs, ys = response_curve(model, scaler, MODEL_NAME)
    fig = go.Figure()
    sample = load_sample()
    if sample is not None:
        fig.add_scatter(x=sample[0], y=sample[1], mode="markers", name="SOCR data (sample)", hoverinfo="skip",
                        marker=dict(size=4, color="rgba(139,147,163,.35)"))
    fig.add_scatter(x=xs, y=ys, mode="lines", name="Deployed model", line=dict(color=LIME, width=3),
                    fill="tozeroy", fillcolor="rgba(182,255,59,.06)",
                    hovertemplate="%{x:.0f} cm → %{y:.1f} kg<extra></extra>")
    fig.add_scatter(x=[h], y=[kg], mode="markers", name="Selected",
                    marker=dict(size=14, color=CYAN, line=dict(color="#fff", width=2)),
                    hovertemplate="%{x:.0f} cm → %{y:.1f} kg<extra></extra>")
    fig.update_xaxes(title="Height (cm)", gridcolor="rgba(255,255,255,.06)", zeroline=False)
    fig.update_yaxes(title="Predicted Weight (kg)", gridcolor="rgba(255,255,255,.06)", zeroline=False,
                     range=[30, 95] if sample is not None else [max(0, float(ys.min()) - 10), float(ys.max()) + 10])
    fig.update_layout(legend=dict(orientation="h", y=1.1, x=0))
    st.plotly_chart(style_fig(fig, 400), width="stretch")
    st.caption("Notebook test set (20%): R² 0.26 · MAE 3.64 kg — height alone explains only part of the variation in weight.")
    html(f"""
    <div class="pipe">
    <div class="node"><div class="eyebrow">Input</div><div class="v">Height (cm)</div></div><div class="arrow">➜</div>
    <div class="node"><div class="eyebrow">Scale</div><div class="v">{SCALER_NAME}</div></div><div class="arrow">➜</div>
    <div class="node"><div class="eyebrow">Predict</div><div class="v">{MODEL_NAME}</div></div><div class="arrow">➜</div>
    <div class="node" style="border-color:rgba(182,255,59,.4)"><div class="eyebrow">Output</div><div class="v">Weight (kg)</div></div>
    </div>""")

html('<div class="foot" style="margin-top:1rem">ManyKGs · Machine Learning Regression Project · academic demo, not a medical tool</div>')
