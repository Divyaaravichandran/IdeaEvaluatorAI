from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import html
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from scorer import (
    MODEL_CONFIGS,
    MODEL_WEIGHTS,
    ENSEMBLE_METHOD,
    calculate_overall,
    evaluate_live_model,
)


st.set_page_config(page_title="AIEval - AI Evaluation Dashboard", page_icon="✦", layout="wide", initial_sidebar_state="collapsed")


background_image = Path(__file__).with_name("Bgpic.png")
try:
    background_data_uri = "data:image/png;base64," + base64.b64encode(background_image.read_bytes()).decode("ascii")
except OSError:
    background_data_uri = ""


DATA_FILE = Path("ideas_scored.xlsx")
REQUIRED_DASHBOARD_COLUMNS = {
    "idea_id", "title", "description", "category", "source", "advance",
    "expert_rank", "avg_overall", "final_rank", "ai_advance",
    "qwen_overall", "mistral_overall", "llama_overall",
    "qwen_ai_advance", "mistral_ai_advance", "llama_ai_advance",
    "human_review_required", "score_spread", "combined_feedback",
}


@st.cache_data
def load_dashboard_data(workbook_mtime):
    """Load one validated scored workbook for every dashboard page."""
    data = pd.read_excel(DATA_FILE)
    missing = REQUIRED_DASHBOARD_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"{DATA_FILE.name} is missing required scored columns: {', '.join(sorted(missing))}")
    if data.empty:
        raise ValueError(f"{DATA_FILE.name} contains no scored ideas.")
    return data


def load_metrics(data):
    truth = pd.to_numeric(data["advance"], errors="coerce").fillna(0).astype(int)
    model_predictions = {
        "Qwen": "qwen_ai_advance",
        "Mistral": "mistral_ai_advance",
        "Llama": "llama_ai_advance",
    }
    models = {
        name: round((pd.to_numeric(data[column], errors="coerce").fillna(0).astype(int) == truth).mean() * 100, 1)
        for name, column in model_predictions.items()
    }
    counts = data["category"].fillna("Others").astype(str).value_counts()
    categories = {name: round(count / len(data) * 100, 1) for name, count in counts.head(3).items()}
    categories["Others"] = round(max(0, 100 - sum(categories.values())), 1)
    return {
        "ideas": len(data),
        "advancements": int(pd.to_numeric(data["ai_advance"], errors="coerce").fillna(0).sum()),
        "accuracy": max(models.values()),
        "models": models,
        "categories": categories,
    }


def requested_page():
    value = st.query_params.get("page")
    if isinstance(value, list):
        value = value[-1] if value else None
    return str(value).lower() if value else None


def go_to(page):
    st.session_state.page = page
    st.query_params["page"] = page
    st.rerun()


try:
    dashboard_data = load_dashboard_data(DATA_FILE.stat().st_mtime)
except (OSError, ValueError) as error:
    st.error(f"Unable to load the scored dataset: {error}")
    st.stop()
metrics = load_metrics(dashboard_data)
valid_pages = {"home", "leaderboard", "comparison", "detail", "live"}
url_page = requested_page()
if "page" not in st.session_state:
    st.session_state.page = url_page if url_page in valid_pages else "home"
elif url_page in valid_pages and url_page != st.session_state.page:
    st.session_state.page = url_page

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
:root { --green:#08b879; --ink:#16312e; --muted:#66807a; --line:rgba(77,148,126,.18); }
html, body, [data-testid="stAppViewContainer"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
[data-testid="stAppViewContainer"] { background:radial-gradient(ellipse at 96% 12%,rgba(87,220,164,.38),transparent 27%),radial-gradient(ellipse at 5% 72%,rgba(151,238,205,.42),transparent 30%),linear-gradient(135deg,#f7fcfa 0%,#e3f5ee 52%,#f2faf7 100%); }
[data-testid="stHeader"] { background:transparent; }.block-container { max-width:1240px; padding:10px 24px 55px; }
.st-key-navbar { border-bottom:1px solid var(--line); padding:0 170px 10px 0; margin-bottom:30px; }
.brand { color:#087f59; font-size:18px; font-weight:700; letter-spacing:-.4px; padding:8px 0; }
.st-key-navbar .stButton > button { width:100%; min-height:38px; border:0; border-bottom:2px solid transparent; border-radius:0; background:transparent; color:var(--muted); box-shadow:none; font-size:13px; font-weight:600; padding:8px 3px; }
.st-key-navbar .stButton > button:hover { border:0; border-bottom:2px solid transparent; background:transparent; color:var(--ink); box-shadow:none; }
.st-key-navbar .stButton > button[kind="primary"], .st-key-navbar .stButton > button[kind="primary"]:hover { border:0; border-bottom:2px solid var(--green); background:transparent; color:#087f59; box-shadow:none; }
.hero { text-align:center; padding:105px 10px 90px; }.badge { display:inline-block; color:#087f59; background:rgba(8,184,121,.08); border:1px solid rgba(8,184,121,.32); padding:6px 15px; border-radius:20px; font-size:10px; font-weight:700; letter-spacing:1px; }
.hero h1 { max-width:820px; margin:32px auto 18px; font-size:48px; line-height:1.15; letter-spacing:-2.4px; }.hero p { max-width:690px; margin:auto; color:var(--muted); font-size:16px; line-height:1.55; }
.metrics { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; }.metric { padding:20px; min-height:100px; background:rgba(255,255,255,.78); border:1px solid var(--line); border-radius:14px; box-shadow:0 9px 25px rgba(62,126,105,.10); }.metric small { color:#087f59; font-weight:700; }.metric strong { display:block; margin:8px 0 3px; font-size:28px; }.metric span { color:var(--muted); font-size:12px; }
.comparison-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:18px; }.comparison-card { padding:26px 22px; background:rgba(255,255,255,.78); border:1px solid var(--line); border-radius:14px; box-shadow:0 9px 25px rgba(62,126,105,.10); }.comparison-card.featured { background:linear-gradient(145deg,rgba(218,250,236,.96),rgba(190,242,218,.84)); border-color:rgba(8,184,121,.55); }.comparison-card h3 { margin:0 0 14px; font-size:17px; }.comparison-card ul { margin:0; padding-left:19px; color:var(--muted); font-size:13px; line-height:1.9; }
.st-key-leaderboard_filters { margin:10px 0 34px; padding:18px; border:1px solid var(--line); border-radius:16px; background:rgba(255,255,255,.76); box-shadow:0 10px 28px rgba(62,126,105,.08); }.st-key-leaderboard_filters input, .st-key-leaderboard_filters [data-baseweb="select"] > div { background:#fff; border-color:rgba(77,148,126,.28); }.podium-card { min-height:145px; padding:23px; border:1px solid var(--line); border-radius:16px; box-shadow:0 14px 32px rgba(62,126,105,.14); }.podium-card .rank { font-size:13px; font-weight:700; letter-spacing:.8px; }.podium-card h3 { margin:12px 0 6px; font-size:20px; }.podium-card p { margin:0; color:var(--muted); font-size:13px; }.podium-card.rank-1 { background:linear-gradient(145deg,#fff8df,#ffedb0); border-color:#edbf49; }.podium-card.rank-1 .rank { color:#936400; }.podium-card.rank-2 { background:linear-gradient(145deg,#f3efff,#ded4ff); border-color:#9d87e6; }.podium-card.rank-2 .rank { color:#5b45a4; }.podium-card.rank-3 { background:linear-gradient(145deg,#fff0ea,#ffd6c6); border-color:#e49a78; }.podium-card.rank-3 .rank { color:#a94b29; }
.stButton > button { transition:transform .18s ease, box-shadow .18s ease; }.stButton > button:hover { transform:translateY(-1px); box-shadow:0 7px 16px rgba(8,127,89,.16); }
.section-title { margin:70px 0 24px; text-align:center; }.section-title h2 { margin:0 0 7px; font-size:26px; }.section-title p { color:var(--muted); margin:0; }
.chart-title { font-size:15px; font-weight:700; color:var(--ink); border-left:4px solid var(--green); padding-left:10px; }.stButton button[kind="primary"] { background:#087f59; border-color:#087f59; }
.comparison-hero { padding:24px 0 0; }.comparison-hero h1 { margin:0 0 8px; font-size:38px; letter-spacing:-1.7px; }.comparison-hero p { margin:0; color:var(--muted); }.comparison-section { margin-top:40px; padding-top:32px; border-top:1px solid var(--line); animation:fadeIn .4s ease both; }.comparison-section-title { margin:0 0 20px; font-size:20px; }.comparison-section-subtitle { margin:-12px 0 20px; color:var(--muted); font-size:13px; }.comparison-table-card, .chart-card, .scatter-card, .st-key-comparison_chart_left, .st-key-comparison_chart_right { width:100%; box-sizing:border-box; padding:20px 24px; border-radius:12px; background:rgba(255,255,255,.84); border:1px solid var(--line); box-shadow:0 10px 28px rgba(62,126,105,.08); }.comparison-table { width:100%; overflow-x:auto; border-radius:8px; }.comparison-table table { width:100%; border-collapse:separate; border-spacing:0; min-width:760px; }.comparison-table th { position:sticky; top:0; z-index:1; background:#e9f8f1; color:#37685c; text-align:left; padding:15px 17px; font-size:11px; text-transform:uppercase; letter-spacing:.7px; }.comparison-table td { padding:16px 17px; border-top:1px solid rgba(77,148,126,.12); font-size:13px; }.comparison-table tr { transition:background .2s ease; }.comparison-table tbody tr:hover { background:rgba(8,184,121,.06); }.comparison-table .best { background:linear-gradient(90deg,rgba(213,250,231,.9),rgba(255,255,255,.35)); box-shadow:inset 3px 0 0 #08b879; }.model-name { font-weight:700; color:#165b49; }.metric-cell { min-width:140px; }.metric-line { display:flex; justify-content:space-between; gap:10px; margin-bottom:7px; }.bar-track { height:7px; border-radius:8px; background:#dceee7; overflow:hidden; }.bar-fill { height:100%; border-radius:8px; }.metric-good { color:#087f59; }.metric-warn { color:#ad7800; }.metric-bad { color:#b42318; }.explain-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:24px; align-items:stretch; }.explain-card { min-height:112px; box-sizing:border-box; padding:20px; border-radius:12px; background:#f1f5f9; border:1px solid var(--line); animation:fadeIn .4s ease both; }.explain-card strong { display:block; margin:8px 0 4px; }.explain-card span { color:var(--muted); font-size:12px; line-height:1.45; }.explain-icon { color:#087f59; font-size:20px; }.st-key-comparison_chart_left, .st-key-comparison_chart_right { height:380px; overflow:hidden; }.st-key-scatter_plot { margin-top:40px; height:400px; overflow:hidden; padding:20px 24px; border-radius:12px; background:rgba(255,255,255,.84); border:1px solid var(--line); box-shadow:0 10px 28px rgba(62,126,105,.08); }.chart-card h3 { margin:0 0 3px; font-size:15px; }.subtle-note { color:var(--muted); font-size:12px; margin:0 0 8px; } @keyframes fadeIn { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
.detail-selector { margin-top:24px; padding:18px 20px; border-radius:12px; background:rgba(255,255,255,.8); border:1px solid var(--line); }.idea-header-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }.idea-info { padding:18px 20px; background:#fff; border:1px solid var(--line); border-radius:12px; min-height:74px; }.idea-info small { color:var(--muted); text-transform:uppercase; letter-spacing:.7px; font-size:10px; }.idea-info strong { display:block; margin-top:8px; font-size:19px; }.idea-info span { display:block; margin-top:8px; color:var(--muted); font-size:13px; }.tag { display:inline-block; padding:5px 10px; border-radius:999px; background:#e5f7ef; color:#087f59; font-size:12px; font-weight:700; }.status-agree { color:#087f59!important; }.status-disagree { color:#b42318!important; }.idea-description, .detail-card { padding:20px 24px; border-radius:12px; background:#fff; border:1px solid var(--line); box-shadow:0 10px 28px rgba(62,126,105,.08); line-height:1.6; }.idea-description h3 { margin:0 0 10px; font-size:16px; }.score-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; }.score-card { height:120px; box-sizing:border-box; padding:16px; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; background:#fff; border:1px solid var(--line); border-radius:12px; transition:transform .2s ease, box-shadow .2s ease; animation:fadeIn .3s ease both; }.score-card:hover { transform:translateY(-3px); box-shadow:0 6px 16px rgba(0,0,0,.08); }.score-card small { color:var(--muted); font-size:11px; }.score-number { margin:5px 0 7px; font-size:30px; line-height:1; font-weight:700; }.score-mini-track { width:70%; height:6px; border-radius:6px; background:#e3eee9; overflow:hidden; }.score-mini-fill { height:100%; border-radius:6px; animation:grow .5s ease both; }.score-max { margin-top:5px; color:var(--muted); font-size:10px; }.detail-card-title { margin:0 0 16px; font-size:18px; } @keyframes grow { from{width:0} } @media(max-width:900px){ .idea-header-grid{grid-template-columns:1fr} .score-grid{grid-template-columns:repeat(2,1fr)} }
@media(max-width:900px){ .st-key-navbar{padding-right:0}.metrics{grid-template-columns:repeat(2,1fr)} .explain-grid{grid-template-columns:repeat(2,1fr)} }.st-key-detail_radar,.st-key-detail_bars{height:380px;overflow:hidden;padding:20px 24px;border-radius:12px;background:rgba(255,255,255,.84);border:1px solid var(--line);box-shadow:0 10px 28px rgba(62,126,105,.08)}.stButton button { cursor:pointer; } @media(max-width:600px){ .metrics,.comparison-grid,.explain-grid{grid-template-columns:1fr}.hero h1{font-size:35px} .comparison-table-card,.chart-card,.scatter-card{padding:16px}.chart-card,.scatter-card,.st-key-detail_radar,.st-key-detail_bars{height:auto;min-height:350px} }
+

/* Responsive layout refinements: preserve the existing visual system at every viewport. */
*, *::before, *::after { box-sizing:border-box; }
img, svg, canvas { max-width:100%; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main { min-width:0; }
.block-container { width:100%; max-width:1240px; }
.st-key-navbar { width:100%; }
.st-key-navbar > div { flex-wrap:wrap; }
.st-key-navbar button { white-space:nowrap; }
.stMarkdown, .stDataFrame, [data-testid="stVerticalBlock"] { min-width:0; }

@media (max-width:1100px) {
  .block-container { padding-left:20px; padding-right:20px; }
  .hero { padding-top:76px; padding-bottom:64px; }
  .hero h1 { font-size:42px; }
  .comparison-grid, .metrics { grid-template-columns:repeat(2, minmax(0, 1fr)); }
  .comparison-table-card, .chart-card, .scatter-card,
  .st-key-comparison_chart_left, .st-key-comparison_chart_right,
  .st-key-detail_radar, .st-key-detail_bars { padding:18px; }
}

@media (max-width:768px) {
  .block-container { padding:8px 16px 40px; }
  .hero { padding:54px 4px 46px; }
  .hero h1 { margin-top:22px; font-size:clamp(30px, 6vw, 38px); letter-spacing:-1.4px; }
  .hero p { font-size:14px; }
  .comparison-hero h1 { font-size:clamp(28px, 6vw, 36px); }
  .section-title { margin:48px 0 18px; }
  .section-title h2, .comparison-section-title { font-size:22px; }
  .metrics { gap:12px; }
  .metric { padding:16px; min-height:92px; }
  .metric strong { font-size:24px; }
  .comparison-grid { gap:12px; }
  .comparison-card { padding:20px 18px; }
  .st-key-navbar > div { gap:4px; }
  .st-key-navbar button { font-size:12px; padding:6px 8px; }
  .st-key-comparison_chart_left, .st-key-comparison_chart_right,
  .st-key-detail_radar, .st-key-detail_bars { min-height:320px; height:auto; overflow:visible; }
  .stDataFrame { overflow-x:auto; }
}

@media (max-width:480px) {
  .block-container { padding-left:12px; padding-right:12px; }
  .hero { padding-top:38px; padding-bottom:34px; }
  .badge { font-size:9px; padding:5px 11px; }
  .hero h1 { font-size:29px; line-height:1.18; }
  .hero p { font-size:13px; }
  .metrics, .comparison-grid, .explain-grid, .score-grid { grid-template-columns:1fr; }
  .metric { min-height:84px; }
  .comparison-card ul { font-size:12px; line-height:1.7; }
  .comparison-section { margin-top:28px; padding-top:24px; }
  .comparison-section-subtitle, .subtle-note { font-size:12px; }
  .comparison-table-card, .chart-card, .scatter-card,
  .st-key-comparison_chart_left, .st-key-comparison_chart_right,
  .st-key-detail_radar, .st-key-detail_bars { padding:12px; border-radius:10px; }
  .st-key-comparison_chart_left, .st-key-comparison_chart_right,
  .st-key-detail_radar, .st-key-detail_bars { min-height:280px; }
  .idea-description, .detail-card { padding:16px; }
  .score-card { height:108px; }
  .st-key-navbar > div { justify-content:center; }
  .st-key-navbar button { width:100%; min-height:36px; }
}
/* Requested UI visibility rules. */
.brand { font-size:0; }
.brand::after { content:'✦ AIEval'; font-size:18px; }
.hidden-rank-comparison { display:none !important; }
.st-key-scatter_plot { display:none !important; }
.idea-header-grid > .idea-info:nth-child(n+4) { display:none; }

/* Premium product-demo visual system. */
:root { --navy:#101a2e; --navy-2:#17243d; --ink:#15233b; --muted:#728097; --accent:#6d5dfc; --accent-2:#8b7dff; --mint:#27c7a5; --surface:rgba(255,255,255,.9); --line:#e8ebf2; --shadow:0 18px 50px rgba(22,35,59,.07); }
html, body, [data-testid="stAppViewContainer"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
[data-testid="stAppViewContainer"] { background:#f5f7fb; }
[data-testid="stHeader"] { background:rgba(245,247,251,.86); }
.block-container { max-width:1400px; padding:36px 42px 72px; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#111b30 0%,#182741 100%); border-right:0; }
[data-testid="stSidebar"] > div:first-child { padding:26px 18px; }
[data-testid="stSidebar"] * { color:#dce5f4; }
[data-testid="stSidebar"] .brand { display:block; color:#fff; font-size:21px; letter-spacing:-.7px; padding:8px 12px 28px; }
[data-testid="stSidebar"] .brand::after { content:'✦ AIEval'; font-size:21px; }
[data-testid="stSidebar"] .sidebar-kicker { padding:0 12px 12px; color:#8191ad; font-size:10px; text-transform:uppercase; letter-spacing:1.5px; font-weight:700; }
[data-testid="stSidebar"] .stButton { margin:3px 0; }
[data-testid="stSidebar"] .stButton > button { justify-content:flex-start; width:100%; min-height:44px; padding:0 14px; border:1px solid transparent; border-radius:11px; background:transparent; color:#aebbd0; box-shadow:none; font-weight:600; text-align:left; transition:background .2s ease,color .2s ease,transform .2s ease; }
[data-testid="stSidebar"] .stButton > button:hover { background:rgba(255,255,255,.08); color:#fff; transform:translateX(3px); box-shadow:none; }
[data-testid="stSidebar"] .stButton > button[kind="primary"] { background:linear-gradient(135deg,#7061ff,#5a4cea); border-color:rgba(255,255,255,.08); color:#fff; box-shadow:0 10px 22px rgba(60,47,182,.3); }
[data-testid="stSidebar"] .sidebar-foot { margin:40px 12px 0; padding-top:18px; border-top:1px solid rgba(255,255,255,.1); color:#8594ad; font-size:11px; line-height:1.55; }
.navbar { display:none; }
.st-key-navbar { display:none !important; }
.hero { padding:76px 0 48px; text-align:left; max-width:900px; animation:riseIn .55s ease both; }
.hero h1 { max-width:850px; margin:20px 0 16px; font-size:clamp(38px,4.5vw,64px); line-height:1.05; letter-spacing:-3px; color:var(--navy); }
.hero p, .comparison-hero p { max-width:660px; color:var(--muted); font-size:15px; line-height:1.65; }
.badge { color:#5b4de0; background:#efedff; border:1px solid #dcd8ff; padding:7px 12px; border-radius:7px; font-size:10px; font-weight:800; letter-spacing:1.3px; }
.metrics { gap:18px; }
.metric { min-height:128px; padding:22px; background:var(--surface); border:1px solid var(--line); border-radius:16px; box-shadow:var(--shadow); animation:riseIn .45s ease both; transition:transform .25s ease,box-shadow .25s ease; }
.metric:hover, .comparison-card:hover, .podium-card:hover, .score-card:hover, .explain-card:hover { transform:translateY(-4px); box-shadow:0 22px 44px rgba(22,35,59,.12); }
.metric small { color:var(--muted); font-size:10px; letter-spacing:1.25px; }
.metric strong { margin:12px 0 4px; color:var(--navy); font-size:32px; letter-spacing:-1.2px; }
.metric span { color:var(--muted); font-size:12px; }
.section-title { margin:58px 0 20px; text-align:left; }
.section-title h2, .comparison-section-title { color:var(--navy); font-size:22px; letter-spacing:-.6px; }
.section-title p, .comparison-section-subtitle { color:var(--muted); }
.comparison-hero { padding:26px 0 8px; animation:riseIn .45s ease both; }
.comparison-hero h1 { margin:17px 0 8px; color:var(--navy); font-size:42px; letter-spacing:-2px; }
.comparison-section { margin-top:40px; padding-top:34px; border-top:1px solid var(--line); animation:riseIn .5s ease both; }
.comparison-card, .comparison-table-card, .chart-card, .scatter-card, .st-key-comparison_chart_left, .st-key-comparison_chart_right, .st-key-detail_radar, .st-key-detail_bars, .st-key-scatter_plot, .idea-description, .detail-card { background:var(--surface); border:1px solid var(--line); border-radius:16px; box-shadow:var(--shadow); }
.comparison-card { padding:26px 24px; transition:transform .25s ease,box-shadow .25s ease; }
.comparison-card.featured { background:linear-gradient(145deg,#f0eeff,#e4e0ff); border-color:#c9c2ff; }
.comparison-card h3, .detail-card-title { color:var(--navy); }
.comparison-card ul, .comparison-section-subtitle, .subtle-note { color:var(--muted); }
.st-key-leaderboard_filters, .detail-selector { padding:20px; background:var(--surface); border:1px solid var(--line); border-radius:16px; box-shadow:var(--shadow); }
.st-key-leaderboard_filters { margin:20px 0 34px; }
.stTextInput input, .stTextArea textarea, [data-baseweb="select"] > div { border-color:#dfe3ed !important; border-radius:10px !important; background:#fff !important; }
.stTextInput input:focus, .stTextArea textarea:focus { border-color:var(--accent) !important; box-shadow:0 0 0 3px rgba(109,93,252,.12) !important; }
.stButton > button { min-height:42px; border-radius:10px; border:1px solid #dfe3ed; background:#fff; color:var(--ink); font-weight:700; transition:transform .2s ease,box-shadow .2s ease,background .2s ease; }
.stButton > button:hover { transform:translateY(-2px); box-shadow:0 10px 22px rgba(22,35,59,.12); }
.stButton > button[kind="primary"] { background:linear-gradient(135deg,#7061ff,#5a4cea); border-color:#5a4cea; color:#fff; }
.podium-card { border-radius:16px; box-shadow:var(--shadow); transition:transform .25s ease,box-shadow .25s ease; }
.podium-card h3 { color:var(--navy); }
.comparison-table-card, .st-key-comparison_chart_left, .st-key-comparison_chart_right, .st-key-detail_radar, .st-key-detail_bars { padding:22px; }
.comparison-table th { background:#f3f4fb; color:#64718a; }
.comparison-table td { border-top-color:#edf0f5; }
.comparison-table tbody tr:hover { background:#f8f7ff; }
.comparison-table .best { background:#f3f1ff; box-shadow:inset 3px 0 #6d5dfc; }
.bar-track, .score-mini-track { background:#eceef5; }
.bar-fill { background:linear-gradient(90deg,#6d5dfc,#27c7a5) !important; }
.explain-card { background:#fbfbfe; border-color:var(--line); transition:transform .25s ease,box-shadow .25s ease; }
.explain-icon, .model-name, .metric-good { color:#6354ea !important; }
.idea-info, .idea-description, .score-card { background:#fff; border-color:var(--line); }
.tag { background:#e9fbf6; color:#138d76; }
.score-card { transition:transform .25s ease,box-shadow .25s ease; }
.score-mini-fill { background:linear-gradient(90deg,#6d5dfc,#27c7a5) !important; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:14px; overflow:hidden; box-shadow:var(--shadow); }
[data-testid="stPlotlyChart"] { animation:fadeIn .55s ease both; }
.stProgress > div > div > div { background:linear-gradient(90deg,#6d5dfc,#27c7a5); }
@keyframes riseIn { from { opacity:0; transform:translateY(14px); } to { opacity:1; transform:translateY(0); } }

/* Shared dashboard background: cover the available screen on every page. */
[data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main, .stApp {
    background-color:#f5f7fb;
    background-image:url("__BACKGROUND_IMAGE__");
    background-size:cover;
    background-position:center center;
    background-repeat:no-repeat;
    background-attachment:fixed;
}
[data-testid="stHeader"] { background:rgba(245,247,251,.72); }

@media (max-width:900px) { .block-container { padding:28px 24px 56px; } .hero { padding-top:48px; } }
@media (max-width:600px) { .block-container { padding:22px 15px 44px; } .hero { padding:34px 0 34px; } .hero h1 { font-size:34px; letter-spacing:-1.6px; } .comparison-hero h1 { font-size:34px; } .metrics, .comparison-grid, .explain-grid, .score-grid { grid-template-columns:1fr; } }

/* Home page composition based on the supplied reference image. */
.st-key-home_page { width:100%; max-width:1080px; margin:0 auto; }
.st-key-home_page .hero { padding:8px 0 48px; max-width:850px; }
.st-key-home_page .hero h1 { color:#121b32; font-size:clamp(38px,4vw,56px); line-height:1.08; letter-spacing:-2.6px; margin-bottom:14px; }
.st-key-home_page .hero p { color:#526486; font-size:15px; line-height:1.5; }
.st-key-home_page .badge { color:#6354ea; background:rgba(239,237,255,.78); border-color:#cbc5ff; }
.st-key-home_page .metrics { gap:16px; margin-bottom:38px; }
.st-key-home_page .metric { min-height:101px; padding:17px 15px; background:rgba(255,255,255,.91); border-color:rgba(225,229,243,.95); border-radius:10px; box-shadow:0 12px 26px rgba(76,91,157,.08); }
.st-key-home_page .metric { position:relative; padding-left:82px; }
.st-key-home_page .metric::before { content:'✦'; position:absolute; left:16px; top:20px; width:44px; height:44px; display:grid; place-items:center; border-radius:50%; color:#fff; font-size:23px; background:linear-gradient(145deg,#5790ff,#214ddd); box-shadow:0 8px 16px rgba(42,91,224,.25); }
.st-key-home_page .metric:nth-child(2)::before { content:'☆'; background:linear-gradient(145deg,#a77aff,#6731e9); box-shadow:0 8px 16px rgba(103,49,233,.22); }
.st-key-home_page .metric:nth-child(3)::before { content:'◎'; background:linear-gradient(145deg,#37dcb8,#04a996); box-shadow:0 8px 16px rgba(4,169,150,.22); }
.st-key-home_page .metric:nth-child(4)::before { content:'♧'; background:linear-gradient(145deg,#ffc36d,#f28d1e); box-shadow:0 8px 16px rgba(242,141,30,.22); }
.st-key-home_page .metric small { color:#64718a; font-size:10px; letter-spacing:1px; }
.st-key-home_page .metric strong { margin:13px 0 10px; color:#15233b; font-size:27px; letter-spacing:-.5px; }
.st-key-home_page .metric span { color:#71809a; font-size:10px; }
.st-key-home_page .section-title { margin:44px 0 14px; }
.st-key-home_page .section-title h2 { color:#15233b; font-size:18px; letter-spacing:-.25px; }
.st-key-home_page .section-title p { color:#71809a; font-size:12px; }
.st-key-home_page [data-testid="stPlotlyChart"] { margin-top:-4px; }
.st-key-home_page .st-key-home_accuracy_card, .st-key-home_page .st-key-home_category_card { min-height:446px; padding:24px 24px 16px; background:rgba(255,255,255,.93); border:1px solid rgba(225,229,243,.95); border-left:3px solid #2f68f0; border-radius:16px; box-shadow:0 15px 34px rgba(76,91,157,.10); }
.st-key-home_page .st-key-home_category_card { border-left-color:#11b89e; }
.st-key-home_page .st-key-home_accuracy_card .section-title, .st-key-home_page .st-key-home_category_card .section-title { margin:0 0 16px; }
.st-key-home_page .st-key-home_accuracy_card .section-title h2, .st-key-home_page .st-key-home_category_card .section-title h2 { font-size:19px; }
.st-key-home_page .comparison-grid { gap:12px; }
.st-key-home_page .comparison-card { min-height:142px; padding:18px 15px 18px 76px; position:relative; background:rgba(255,255,255,.91); border-radius:16px; box-shadow:0 12px 26px rgba(76,91,157,.08); }
.st-key-home_page .comparison-card::before { content:'▤'; position:absolute; left:18px; top:22px; width:40px; height:40px; display:grid; place-items:center; color:#3474f2; background:#e8f0ff; border-radius:10px; font-size:22px; }
.st-key-home_page .comparison-card:nth-child(2)::before { content:'♧'; color:#7041ee; background:#f0eaff; }
.st-key-home_page .comparison-card:nth-child(3)::before { content:'↗'; color:#10a888; background:#e2f7f1; }
.st-key-home_page .comparison-card.featured { background:rgba(239,237,255,.88); border-color:#cbc5ff; }
.st-key-home_page .comparison-card h3 { color:#15233b; font-size:16px; }
.st-key-home_page .comparison-card ul { color:#71809a; font-size:13px; line-height:1.9; }
.st-key-home_page > div:last-child { margin-top:12px; }
@media (max-width:900px) { .st-key-home_page .metrics { grid-template-columns:repeat(2,1fr); } .st-key-home_page .st-key-home_accuracy_card, .st-key-home_page .st-key-home_category_card { min-height:420px; } }
@media (max-width:600px) { .st-key-home_page .hero { padding-top:0; } .st-key-home_page .metrics { grid-template-columns:repeat(2,1fr); gap:12px; margin-bottom:28px; } .st-key-home_page .metric { padding-left:68px; } .st-key-home_page .metric::before { left:12px; width:38px; height:38px; } .st-key-home_page .metric strong { font-size:23px; } .st-key-home_page .metric small, .st-key-home_page .metric span { font-size:9px; } .st-key-home_page .st-key-home_accuracy_card, .st-key-home_page .st-key-home_category_card { min-height:380px; padding:18px 15px 10px; } }

/* Leaderboard page uses the same polished visual language as Home. */
.st-key-leaderboard_page { width:100%; max-width:1080px; margin:0 auto; }
.st-key-leaderboard_page .comparison-hero { padding:8px 0 26px; }
.st-key-leaderboard_page .comparison-hero h1 { color:#121b32; font-size:clamp(38px,4vw,56px); line-height:1.08; letter-spacing:-2.6px; }
.st-key-leaderboard_page .comparison-hero p { color:#526486; font-size:15px; }
.st-key-leaderboard_page .badge { color:#6354ea; background:rgba(239,237,255,.78); border-color:#cbc5ff; }
.st-key-leaderboard_page .st-key-leaderboard_filters { margin:0 0 34px; padding:20px; background:rgba(255,255,255,.93); border:1px solid rgba(225,229,243,.95); border-radius:16px; box-shadow:0 15px 34px rgba(76,91,157,.10); }
.st-key-leaderboard_page .section-title { margin:34px 0 16px; }
.st-key-leaderboard_page .section-title h2 { color:#15233b; font-size:20px; }
.st-key-leaderboard_page .podium-card { min-height:156px; padding:22px; background:rgba(255,255,255,.93); border:1px solid rgba(225,229,243,.95); border-radius:16px; box-shadow:0 15px 34px rgba(76,91,157,.10); }
.st-key-leaderboard_page .podium-card.rank-1 { background:linear-gradient(145deg,rgba(232,240,255,.97),rgba(255,255,255,.93)); border-color:#a9c3ff; }
.st-key-leaderboard_page .podium-card.rank-2 { background:linear-gradient(145deg,rgba(242,236,255,.97),rgba(255,255,255,.93)); border-color:#c9b8ff; }
.st-key-leaderboard_page .podium-card.rank-3 { background:linear-gradient(145deg,rgba(228,249,244,.97),rgba(255,255,255,.93)); border-color:#a7e3d3; }
.st-key-leaderboard_page .podium-card .rank { font-size:11px; letter-spacing:1px; }
.st-key-leaderboard_page .podium-card h3 { margin:14px 0 8px; color:#15233b; font-size:18px; }
.st-key-leaderboard_page .podium-card p { color:#71809a; font-size:13px; }
.st-key-leaderboard_page .stDataFrame { margin-top:34px; background:rgba(255,255,255,.93); }
.st-key-leaderboard_page .stExpander { background:rgba(255,255,255,.84); border:1px solid rgba(225,229,243,.95); border-radius:12px; margin:8px 0; }
@media (max-width:600px) { .st-key-leaderboard_page .comparison-hero { padding-top:0; } .st-key-leaderboard_page .st-key-leaderboard_filters { padding:14px; } .st-key-leaderboard_page .podium-card h3 { font-size:16px; } }

/* Keep every workspace page aligned with the Home composition. */
.st-key-model_comparison_page, .st-key-detail_page, .st-key-live_page { width:100%; max-width:1080px; margin:0 auto; }
.st-key-model_comparison_page .comparison-hero, .st-key-detail_page .comparison-hero, .st-key-live_page .comparison-hero { padding:8px 0 26px; }
.st-key-model_comparison_page .comparison-hero h1, .st-key-detail_page .comparison-hero h1, .st-key-live_page .comparison-hero h1 { color:#121b32; font-size:clamp(38px,4vw,56px); line-height:1.08; letter-spacing:-2.6px; }
.st-key-model_comparison_page .comparison-hero p, .st-key-detail_page .comparison-hero p, .st-key-live_page .comparison-hero p { color:#526486; font-size:15px; }
.st-key-model_comparison_page .badge, .st-key-detail_page .badge, .st-key-live_page .badge { color:#6354ea; background:rgba(239,237,255,.78); border-color:#cbc5ff; }
.st-key-model_comparison_page .comparison-section, .st-key-detail_page .comparison-section, .st-key-live_page .section-title { margin-top:34px; }
.st-key-model_comparison_page .metrics { gap:16px; margin:0 0 38px; }
.st-key-model_comparison_page .metric { min-height:101px; padding:17px 15px 17px 82px; background:rgba(255,255,255,.91); border-color:rgba(225,229,243,.95); border-radius:10px; box-shadow:0 12px 26px rgba(76,91,157,.08); position:relative; }
.st-key-model_comparison_page .metric::before { content:'✦'; position:absolute; left:16px; top:20px; width:44px; height:44px; display:grid; place-items:center; border-radius:50%; color:#fff; font-size:23px; background:linear-gradient(145deg,#5790ff,#214ddd); }
.st-key-model_comparison_page .metric:nth-child(2)::before { content:'☆'; background:linear-gradient(145deg,#a77aff,#6731e9); }
.st-key-model_comparison_page .metric:nth-child(3)::before { content:'◎'; background:linear-gradient(145deg,#37dcb8,#04a996); }
.st-key-model_comparison_page .metric:nth-child(4)::before { content:'♧'; background:linear-gradient(145deg,#ffc36d,#f28d1e); }
.st-key-model_comparison_page .metric small { color:#64718a; font-size:10px; letter-spacing:1px; }
.st-key-model_comparison_page .metric strong { margin:13px 0 10px; color:#15233b; font-size:27px; letter-spacing:-.5px; }
.st-key-model_comparison_page .metric span { color:#71809a; font-size:10px; }
.st-key-model_comparison_page .comparison-section-title, .st-key-detail_page .comparison-section-title { color:#15233b; }
.st-key-detail_page .detail-selector, .st-key-live_page .stTextInput, .st-key-live_page .stTextArea { background:rgba(255,255,255,.93); }
.st-key-live_page .stTextInput, .st-key-live_page .stTextArea { padding:0; }
.leaderboard-table-wrap { width:100%; overflow-x:auto; background:rgba(255,255,255,.93); border:1px solid rgba(225,229,243,.95); border-radius:14px; box-shadow:0 12px 26px rgba(76,91,157,.08); }
.leaderboard-table { width:100%; min-width:980px; border-collapse:separate; border-spacing:0; }
.leaderboard-table th { padding:14px 12px; background:#f3f4fb; color:#111b35; text-align:left; font-size:11px; font-weight:800; letter-spacing:.5px; }
.leaderboard-table td { padding:13px 12px; border-top:1px solid #edf0f5; color:#526486; font-size:12px; }
.leaderboard-table tbody tr:hover { background:#f8f7ff; }
.leaderboard-summary { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:0 0 26px; }
.leaderboard-summary article { padding:15px 17px; background:rgba(255,255,255,.78); border:1px solid rgba(225,229,243,.95); border-radius:12px; }
.leaderboard-summary small { display:block; color:#71809a; font-size:10px; font-weight:800; letter-spacing:.8px; }
.leaderboard-summary strong { display:block; margin-top:5px; color:#15233b; font-size:22px; }
.leaderboard-toolbar { display:flex; align-items:center; justify-content:space-between; gap:14px; margin:0 0 12px; }
.leaderboard-toolbar h2 { margin:0; color:#15233b; font-size:21px; }
.leaderboard-toolbar p { margin:3px 0 0; color:#71809a; font-size:12px; }
.rank-pill, .decision-pill, .review-pill, .category-pill { display:inline-flex; align-items:center; white-space:nowrap; border-radius:999px; font-size:11px; font-weight:750; }
.rank-pill { min-width:31px; justify-content:center; padding:5px 8px; background:#efedff; color:#5545c7; }
.decision-pill { padding:5px 9px; background:#def7eb; color:#087f59; }
.decision-pill.reject { background:#fff0ef; color:#bc3d37; }
.review-pill { padding:5px 9px; background:#fff4d6; color:#966300; }
.category-pill { padding:5px 9px; background:#f1f4fb; color:#526486; }
.idea-cell { min-width:220px; color:#15233b; font-weight:700; }
.idea-cell small { display:block; margin-top:3px; color:#71809a; font-size:11px; font-weight:500; }
.score-value { color:#15233b; font-variant-numeric:tabular-nums; font-weight:700; }
.score-muted { color:#9aa6ba; }
@media (max-width:600px) { .leaderboard-summary { grid-template-columns:1fr; } .leaderboard-toolbar { align-items:flex-start; flex-direction:column; } }
@media (max-width:600px) { .st-key-model_comparison_page .comparison-hero, .st-key-detail_page .comparison-hero, .st-key-live_page .comparison-hero { padding-top:0; } .st-key-model_comparison_page .metric { padding-left:68px; } .st-key-model_comparison_page .metric::before { left:12px; width:38px; height:38px; } }

/* Strong, high-contrast heading hierarchy. */
.section-title h1, .section-title h2, .comparison-section-title, .comparison-hero h1,
.comparison-card h3, .podium-card h3, .detail-card-title, .chart-title,
.comparison-table th, .st-key-home_page .hero h1, .st-key-home_page .section-title h2,
.st-key-leaderboard_page .section-title h2, .st-key-leaderboard_page .podium-card h3 {
    color:#111b35;
    font-weight:800;
}
.comparison-table th { font-weight:800; }
.st-key-home_page .section-title h1, .st-key-home_page .section-title h2,
.st-key-leaderboard_page .section-title h2 { text-shadow:0 1px 0 rgba(255,255,255,.7); }

/* Page-specific UI refinements. The leaderboard keeps its independent table system. */
.st-key-home_page .hero { margin:8px 0 24px; padding:82px 42px 74px; overflow:hidden; border:1px solid rgba(169,195,255,.7); border-radius:24px; background:radial-gradient(circle at 84% 14%,rgba(103,64,229,.2),transparent 27%),linear-gradient(135deg,#f9fbff,#eef2ff 55%,#f9fbff); text-align:left; }
.st-key-home_page .hero h1 { max-width:780px; margin:22px 0 14px; color:#15233b; font-size:clamp(34px,4.2vw,56px); line-height:1.06; }
.st-key-home_page .hero p { max-width:570px; margin:0; color:#526486; font-size:16px; }
.st-key-home_page .metrics { margin:0 0 34px; }
.st-key-home_page .metric { position:relative; overflow:hidden; padding:21px 18px 18px 76px; border-color:rgba(225,229,243,.95); background:rgba(255,255,255,.92); box-shadow:0 12px 28px rgba(76,91,157,.09); }
.st-key-home_page .metric::before { position:absolute; left:17px; top:20px; width:42px; height:42px; display:grid; place-items:center; border-radius:13px; background:#edf1ff; color:#3156c8; content:'01'; font-size:11px; font-weight:800; }
.st-key-home_page .metric:nth-child(2)::before { content:'02'; background:#eeeaff; color:#6b45d6; }.st-key-home_page .metric:nth-child(3)::before { content:'03'; background:#e2f8f0; color:#087f59; }.st-key-home_page .metric:nth-child(4)::before { content:'04'; background:#fff2dc; color:#b76a00; }
.st-key-home_page .metric strong { color:#15233b; }.st-key-home_page .metric span { color:#71809a; }
.st-key-home_page .section-title { margin:44px 0 18px; text-align:left; }.st-key-home_page .section-title h1, .st-key-home_page .section-title h2 { color:#15233b; }
.st-key-home_page .st-key-home_accuracy_card, .st-key-home_page .st-key-home_category_card { min-height:342px; padding:22px 24px; border:1px solid rgba(225,229,243,.95); border-radius:18px; background:rgba(255,255,255,.88); box-shadow:0 14px 32px rgba(76,91,157,.08); }
.st-key-home_page .st-key-home_accuracy_card .section-title, .st-key-home_page .st-key-home_category_card .section-title { margin:0 0 8px; }.st-key-home_page .st-key-home_accuracy_card .section-title h2, .st-key-home_page .st-key-home_category_card .section-title h2 { font-size:18px; }
.st-key-home_page .comparison-grid { gap:16px; }.st-key-home_page .comparison-card { min-height:218px; padding:25px; border-color:rgba(225,229,243,.95); background:rgba(255,255,255,.88); box-shadow:0 12px 28px rgba(76,91,157,.08); }.st-key-home_page .comparison-card.featured { border-color:#9baeff; background:linear-gradient(145deg,#f0f3ff,#fdfdff); }

.st-key-model_comparison_page .comparison-hero, .st-key-detail_page .comparison-hero, .st-key-live_page .comparison-hero { padding:30px 0 26px; border-bottom:1px solid rgba(225,229,243,.9); }
.st-key-model_comparison_page .comparison-section { margin-top:30px; padding-top:28px; }.st-key-model_comparison_page .comparison-table-card { padding:10px; border-radius:18px; background:rgba(255,255,255,.9); }.st-key-model_comparison_page .explain-card { min-height:126px; padding:22px; border-radius:16px; background:rgba(255,255,255,.86); border-color:rgba(225,229,243,.95); }.st-key-model_comparison_page .explain-card:hover { border-color:#b9b1ff; }.st-key-model_comparison_page .comparison-section-subtitle { max-width:620px; line-height:1.55; }

.st-key-detail_page .detail-selector { margin:24px 0 8px; padding:18px 20px; border-radius:16px; border-color:rgba(225,229,243,.95); box-shadow:0 12px 26px rgba(76,91,157,.08); }.st-key-detail_page .idea-header-grid { gap:12px; }.st-key-detail_page .idea-info { min-height:96px; padding:17px; border-color:rgba(225,229,243,.95); border-radius:14px; box-shadow:0 8px 18px rgba(76,91,157,.05); }.st-key-detail_page .idea-info strong { color:#15233b; font-size:17px; }.st-key-detail_page .idea-description { border-radius:16px; border-color:rgba(225,229,243,.95); background:rgba(255,255,255,.9); }.st-key-detail_page .comparison-section { margin-top:30px; padding-top:26px; }.st-key-detail_page .score-card { height:132px; border-color:rgba(225,229,243,.95); border-radius:16px; box-shadow:0 10px 22px rgba(76,91,157,.07); }

.st-key-live_form { margin:24px 0 30px; padding:24px; border:1px solid rgba(225,229,243,.95); border-radius:18px; background:rgba(255,255,255,.9); box-shadow:0 14px 32px rgba(76,91,157,.08); }.st-key-live_form label { color:#15233b!important; font-size:13px!important; font-weight:750!important; }.st-key-live_form input, .st-key-live_form textarea { border-radius:10px!important; border-color:#dce2f0!important; background:#fbfcff!important; }.st-key-live_form textarea { line-height:1.55; }.st-key-live_form .stButton > button { min-height:46px; border-radius:10px; font-weight:750; }
.st-key-live_page .section-title { margin:34px 0 16px; text-align:left; }.st-key-live_page .section-title h2 { color:#15233b; font-size:22px; }.st-key-live_page [data-testid="stMetric"] { min-height:108px; padding:17px; border:1px solid rgba(225,229,243,.95); border-radius:14px; background:rgba(255,255,255,.9); box-shadow:0 9px 20px rgba(76,91,157,.06); }.st-key-live_page [data-testid="stMetricLabel"] { color:#64718a; font-size:11px; }.st-key-live_page [data-testid="stMetricValue"] { color:#15233b; font-size:24px; }.st-key-live_page [data-testid="stDataFrame"] { border:1px solid rgba(225,229,243,.95); border-radius:14px; overflow:hidden; }.st-key-live_page .stProgress { margin:10px 0; }.st-key-live_feedback, [class*="st-key-live_feedback_"] { min-height:186px; padding:20px; border:1px solid rgba(225,229,243,.95); border-radius:16px; background:rgba(255,255,255,.88); box-shadow:0 10px 22px rgba(76,91,157,.06); }.st-key-live_feedback h4, [class*="st-key-live_feedback_"] h4 { color:#3156c8; font-size:13px; }.st-key-live_feedback p, .st-key-live_feedback li, [class*="st-key-live_feedback_"] p, [class*="st-key-live_feedback_"] li { color:#526486; line-height:1.6; }
@media (max-width:700px) { .st-key-home_page .hero { padding:50px 24px; border-radius:18px; }.st-key-home_page .metric { padding-left:70px; }.st-key-home_accuracy_card, .st-key-home_category_card { min-height:0; padding:18px; }.st-key-live_form { padding:18px; }.st-key-detail_page .idea-header-grid { grid-template-columns:1fr; } }
</style>
""".replace("__BACKGROUND_IMAGE__", background_data_uri), unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="brand"></div><div class="sidebar-kicker">AI evaluation workspace</div>', unsafe_allow_html=True)
    for page, label in (("home", "Home"), ("leaderboard", "Leaderboard"), ("comparison", "Model Comparison"), ("detail", "Idea Detail"), ("live", "Live Evaluation")):
        if st.button(label, key=f"{page}_sidebar_nav", type="primary" if st.session_state.page == page else "secondary"):
            go_to(page)
    st.markdown('<div class="sidebar-foot">Multi-model evaluation<br>for better product decisions.</div>', unsafe_allow_html=True)

with st.container(key="navbar"):
    brand, home, leaderboard, comparison, detail, live = st.columns([1.25, .85, 1.05, 1.25, 1.0, 1.1], vertical_alignment="center")
    with brand:
        st.markdown('<div class="brand"></div>', unsafe_allow_html=True)
    with home:
        if st.button("Home", key="home_nav", type="primary" if st.session_state.page == "home" else "secondary"):
            go_to("home")
    with leaderboard:
        if st.button("Leaderboard", key="leaderboard_nav", type="primary" if st.session_state.page == "leaderboard" else "secondary"):
            go_to("leaderboard")
    with comparison:
        if st.button("Model Comparison", key="comparison_nav", type="primary" if st.session_state.page == "comparison" else "secondary"):
            go_to("comparison")
    with detail:
        if st.button("Idea Detail", key="detail_nav", type="primary" if st.session_state.page == "detail" else "secondary"):
            go_to("detail")
    with live:
        if st.button("Live Evaluation", key="live_nav", type="primary" if st.session_state.page == "live" else "secondary"):
            go_to("live")


def _home_page_content():
    st.markdown("""
    <section class="hero"><div class="badge">NEXT-GEN AI EVALUATION</div><h1>Hackathon Idea Evaluation and Ranking System Using Multiple LLMs</h1><p>Multi-LLM evaluation. Smarter rankings. Better decisions.</p></section>
    """, unsafe_allow_html=True)
    cards = [("TOTAL IDEAS", metrics["ideas"], "Ideas processed from dataset"), ("AI ADVANCEMENTS", metrics["advancements"], "Shortlisted by consensus"), ("BEST MODEL ACCURACY", f'{metrics["accuracy"]}%', "Highest verification score"), ("MODELS COMPARED", "3", "State-of-the-art LLMs")]
    html = "".join(f'<article class="metric"><small>{label}</small><strong>{value}</strong><span>{description}</span></article>' for label, value, description in cards)
    st.markdown(f'<section class="metrics">{html}</section>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="medium")
    with left:
        with st.container(key="home_accuracy_card"):
            st.markdown('<div class="section-title"><h2>Model accuracy</h2></div>', unsafe_allow_html=True)
            figure = go.Figure(go.Bar(x=list(metrics["models"]), y=list(metrics["models"].values()), marker_color=["#2457d9", "#2479ee", "#18b9a4"], text=[f"{value}%" for value in metrics["models"].values()], textposition="outside"))
            figure.update_layout(height=270, margin=dict(l=8, r=8, t=20, b=30), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(range=[0, 100], ticksuffix="%", gridcolor="rgba(84,109,170,.16)"), showlegend=False)
            st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
    with right:
        with st.container(key="home_category_card"):
            st.markdown('<div class="section-title"><h2>Ideas by category</h2></div>', unsafe_allow_html=True)
            figure = go.Figure(go.Pie(labels=list(metrics["categories"]), values=list(metrics["categories"].values()), hole=.66, marker=dict(colors=["#08b879", "#26a9cf", "#2f68f0", "#8b4de8"], line=dict(color="#ffffff", width=2)), textinfo="none", domain=dict(x=[0, .62])))
            figure.update_layout(height=270, margin=dict(l=0, r=0, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="v", x=.60, y=.5, xanchor="left", font=dict(size=10, color="#304263")))
            st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
    st.markdown('<div class="section-title"><h1>Innovation Benchmarking</h1><p><b>How the system improves on earlier evaluation approaches.</b></p></div>', unsafe_allow_html=True)
    comparisons = [
        ("Base Paper", ["GPT-3.5 only", "Climate change domain", "No feedback", "70–80% accuracy"], ""),
        ("Our System", ["3 LLMs compared", "Hackathon domain", "AI feedback given", "80%+ accuracy"], "featured"),
        ("Our Novelty", ["Multi-LLM benchmark", "New domain", "Feedback generation", "Live dashboard"], ""),
    ]
    comparison_html = "".join(f'<article class="comparison-card {style}"><h3>{title}</h3><ul>{"".join(f"<li>{item}</li>" for item in items)}</ul></article>' for title, items, style in comparisons)
    st.markdown(f'<section class="comparison-grid">{comparison_html}</section>', unsafe_allow_html=True)


def home_page():
    with st.container(key="home_page"):
        _home_page_content()


def _leaderboard_page_content():
    data = dashboard_data.copy()
    # Rank directly from the combined AI score. Ideas with equal scores share
    # the same rank, rather than being forced into a 1–120 sequence.
    data["avg_overall"] = pd.to_numeric(data["avg_overall"], errors="coerce")
    data["ai_rank"] = pd.to_numeric(data["final_rank"], errors="coerce").astype("Int64")
    data = data.sort_values("ai_rank", na_position="last").reset_index(drop=True)
    st.markdown('<section class="comparison-hero"><div class="badge">RESEARCH LEADERBOARD</div><h1>Leaderboard</h1><p>Ranked hackathon ideas and AI advancement decisions.</p></section>', unsafe_allow_html=True)
    def focus_table():
        st.session_state.focus_leaderboard_table = True

    with st.container(key="leaderboard_filters"):
        search, category, decision, sort = st.columns([2.1, 1, 1.25, 1.2])
        query = search.text_input("Search idea by title...", key="idea_search", label_visibility="collapsed", placeholder="Search idea by title...", on_change=focus_table)
        category_value = category.selectbox("Category", ["All", *sorted(data["category"].dropna().unique())], key="idea_category", label_visibility="collapsed", on_change=focus_table)
        decision_value = decision.selectbox("Decision", ["Show All", "Show Advanced Only", "Show Rejected Only"], key="idea_decision", label_visibility="collapsed", on_change=focus_table)
        sort_value = sort.selectbox("Sort", ["AI Rank", "Highest score", "Lowest score"], key="idea_sort", label_visibility="collapsed", on_change=focus_table)
    filtered = data[data["title"].str.contains(query, case=False, na=False)].copy()
    if category_value != "All":
        filtered = filtered[filtered["category"] == category_value]
    if decision_value == "Show Advanced Only":
        filtered = filtered[filtered["ai_advance"] == 1]
    elif decision_value == "Show Rejected Only":
        filtered = filtered[filtered["ai_advance"] == 0]
    if sort_value == "Highest score":
        filtered = filtered.sort_values(["avg_overall", "ai_rank"], ascending=[False, True])
    elif sort_value == "Lowest score":
        filtered = filtered.sort_values(["avg_overall", "ai_rank"], ascending=[True, True])

    advanced_count = int(pd.to_numeric(filtered["ai_advance"], errors="coerce").fillna(0).sum())
    review_count = int(pd.to_numeric(filtered["human_review_required"], errors="coerce").fillna(0).sum())
    st.markdown(
        '<section class="leaderboard-summary">'
        f'<article><small>MATCHING IDEAS</small><strong>{len(filtered)}</strong></article>'
        f'<article><small>ADVANCE RECOMMENDATIONS</small><strong>{advanced_count}</strong></article>'
        f'<article><small>NEEDS HUMAN REVIEW</small><strong>{review_count}</strong></article>'
        '</section>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title"><h2>Top ideas</h2></div>', unsafe_allow_html=True)
    podium = filtered.nsmallest(3, "ai_rank")
    podium_columns = st.columns(3)
    medals = ["🥇", "🥈", "🥉"]
    for column, medal, (_, idea) in zip(podium_columns, medals, podium.iterrows()):
        with column:
            style = f'rank-{int(idea["ai_rank"])}'
            st.markdown(f'<article class="podium-card {style}"><div class="rank">{medal} RANK {int(idea["ai_rank"])}</div><h3>{idea["title"]}</h3><p>Avg: {idea["avg_overall"]:.2f} · {idea["category"]}</p></article>', unsafe_allow_html=True)
            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
            button_left, button_center, button_right = st.columns([1, 1.3, 1])
            with button_center:
                if st.button("View idea", key=f"podium_{idea['idea_id']}", type="secondary", use_container_width=True):
                    st.session_state.detail_idea_id = int(idea["idea_id"])
                    go_to("detail")

    st.markdown('<div style="height:34px"></div>', unsafe_allow_html=True)

    st.markdown('<div id="leaderboard-table"></div>', unsafe_allow_html=True)
    if st.session_state.pop("focus_leaderboard_table", False):
        components.html("<script>window.parent.document.getElementById('leaderboard-table')?.scrollIntoView({behavior:'smooth', block:'start'});</script>", height=0)
    page_size = st.selectbox("Rows per page", ["All", 10, 25, 50], index=2, key="leaderboard_page_size")
    visible = filtered if page_size == "All" else filtered.head(page_size)
    st.markdown(
        f'<div class="leaderboard-toolbar"><div><h2>Ranked ideas</h2>'
        f'<p>Showing {len(visible)} of {len(filtered)} matching ideas.</p></div></div>',
        unsafe_allow_html=True,
    )
    headers = ''.join(f'<th>{label}</th>' for label in ("Rank", "Idea", "Category", "Decision", "Average", "Range", "Qwen", "Mistral", "Llama", "Expert"))

    def score_cell(value):
        value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return '<span class="score-muted">—</span>' if pd.isna(value) else f'<span class="score-value">{value:.2f}</span>'

    def table_row(idea):
        rank = '—' if pd.isna(idea["ai_rank"]) else str(int(idea["ai_rank"]))
        category_value = idea.get("category")
        category_name = html.escape("Uncategorised" if pd.isna(category_value) else str(category_value))
        title = html.escape(str(idea["title"]))
        decision = int(pd.to_numeric(pd.Series([idea["ai_advance"]]), errors="coerce").fillna(0).iloc[0])
        decision_html = '<span class="decision-pill">Advance</span>' if decision else '<span class="decision-pill reject">Reject</span>'
        review = int(pd.to_numeric(pd.Series([idea["human_review_required"]]), errors="coerce").fillna(0).iloc[0])
        review_html = '<small><span class="review-pill">Review</span></small>' if review else ''
        expert = '—' if pd.isna(idea["expert_rank"]) else f'#{int(idea["expert_rank"])}'
        return (
            '<tr>'
            f'<td><span class="rank-pill">{rank}</span></td>'
            f'<td class="idea-cell">{title}{review_html}</td>'
            f'<td><span class="category-pill">{category_name}</span></td>'
            f'<td>{decision_html}</td><td>{score_cell(idea["avg_overall"])}</td>'
            f'<td>{score_cell(idea["score_spread"])}</td><td>{score_cell(idea["qwen_overall"])}</td>'
            f'<td>{score_cell(idea["mistral_overall"])}</td><td>{score_cell(idea["llama_overall"])}</td><td>{expert}</td>'
            '</tr>'
        )

    body = ''.join(table_row(idea) for _, idea in visible.iterrows())
    if not body:
        body = '<tr><td colspan="10">No ideas match the current filters.</td></tr>'
    st.markdown(
        f'<div class="leaderboard-table-wrap"><table class="leaderboard-table"><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table></div>',
        unsafe_allow_html=True,
    )

    flagged = filtered.loc[
        pd.to_numeric(filtered["human_review_required"], errors="coerce").fillna(0).eq(1)
    ]
    st.markdown(
        f'<div class="section-title"><h2>Flagged ideas for human review ({len(flagged)})</h2>'
        '<p>These ideas have incomplete model results or substantial disagreement between model scores.</p></div>',
        unsafe_allow_html=True,
    )
    if flagged.empty:
        st.info("No flagged ideas match the current filters.")
    else:
        flagged_page_size = st.selectbox(
            "Flagged rows per page", ["All", 10, 25, 50], index=1, key="flagged_page_size"
        )
        visible_flagged = flagged if flagged_page_size == "All" else flagged.head(flagged_page_size)
        st.markdown(
            f'<div class="leaderboard-toolbar"><div><h2>Review queue</h2>'
            f'<p>Showing {len(visible_flagged)} of {len(flagged)} ideas that need a closer look.</p></div></div>',
            unsafe_allow_html=True,
        )
        flagged_body = ''.join(table_row(idea) for _, idea in visible_flagged.iterrows())
        st.markdown(
            f'<div class="leaderboard-table-wrap"><table class="leaderboard-table"><thead><tr>{headers}</tr></thead><tbody>{flagged_body}</tbody></table></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title"><h2>Feedback for top 10 ideas</h2></div>', unsafe_allow_html=True)
    for _, idea in filtered.nsmallest(10, "ai_rank").iterrows():
        feedback = idea.get("combined_feedback") or idea.get("llama_feedback") or "Feedback was not generated for this idea."
        with st.expander(f'#{int(idea["ai_rank"])} · {idea["title"]}'):
            st.write(feedback)


def leaderboard_page():
    with st.container(key="leaderboard_page"):
        _leaderboard_page_content()


def _model_comparison_page_content():
    data = dashboard_data.copy()
    model_map = {"Qwen": "qwen_ai_advance", "Mistral": "mistral_ai_advance", "Llama": "llama_ai_advance"}
    rows, chart_rows = [], []
    truth = pd.to_numeric(data.get("advance"), errors="coerce").fillna(0).astype(int)
    for name, prediction_column in model_map.items():
        predicted = pd.to_numeric(data[prediction_column], errors="coerce").fillna(0).astype(int)
        tp, tn = int(((predicted == 1) & (truth == 1)).sum()), int(((predicted == 0) & (truth == 0)).sum())
        fp, fn = int(((predicted == 1) & (truth == 0)).sum()), int(((predicted == 0) & (truth == 1)).sum())
        sensitivity = tp / (tp + fn) if tp + fn else 0
        specificity = tn / (tn + fp) if tn + fp else 0
        balanced = round((sensitivity + specificity) * 50, 1)
        plain = round((tp + tn) / len(data) * 100, 1) if len(data) else 0
        rows.append((name, balanced, plain, tp, tn, fp, fn))
        chart_rows.append({"Model": name, "Balanced Accuracy": balanced, "Plain Accuracy": plain, "TP": tp, "TN": tn, "FP": fp, "FN": fn})
    best_name = max(rows, key=lambda row: row[2])[0]
    st.markdown('<section class="comparison-hero"><div class="badge">RESEARCH VALIDATION</div><h1>Model Comparison</h1><p>Transparent performance analysis across every scored idea in the evaluation dataset.</p></section>', unsafe_allow_html=True)
    st.markdown(f'<section class="metrics" style="margin-top:32px"><article class="metric"><small>BEST MODEL</small><strong>{best_name}</strong><span>Highest accuracy</span></article><article class="metric"><small>IDEAS EVALUATED</small><strong>{len(data)}</strong><span>Loaded from dataset</span></article><article class="metric"><small>LOWER BENCHMARK</small><strong>70%</strong><span>Minimum acceptable accuracy</span></article><article class="metric"><small>UPPER BENCHMARK</small><strong>80%</strong><span>Strong validation target</span></article></section>', unsafe_allow_html=True)
    def level(value): return "metric-good" if value >= 80 else "metric-warn" if value >= 70 else "metric-bad"
    def accuracy_cell(value):
        color = "#08b879" if value >= 80 else "#e3b341" if value >= 70 else "#e05252"
        return f'<td class="metric-cell" title="{value:.1f}% accuracy"><div class="metric-line"><span class="{level(value)}">{value:.1f}%</span><span>●</span></div><div class="bar-track"><div class="bar-fill" style="width:{min(value,100)}%;background:{color}"></div></div></td>'
    table = '<div class="comparison-table"><table><thead><tr><th>Model</th><th>Balanced Accuracy</th><th>Plain Accuracy</th><th>TP</th><th>TN</th><th>FP</th><th>FN</th></tr></thead><tbody>'
    for name, balanced, plain, tp, tn, fp, fn in rows:
        best = ' best' if name == best_name else ''
        table += f'<tr class="{best}"><td class="model-name">{name}{"  ✦" if name == best_name else ""}</td>{accuracy_cell(balanced)}{accuracy_cell(plain)}' + ''.join(f'<td title="{label}: {value} ideas">{value}</td>' for label, value in (("True positives",tp),("True negatives",tn),("False positives",fp),("False negatives",fn))) + '</tr>'
    st.markdown('<section class="comparison-section"><h2 class="comparison-section-title">Accuracy Table</h2><div class="comparison-table-card">' + table + '</tbody></table></div></div></section>', unsafe_allow_html=True)
    explanations = [("◌", "Balanced Accuracy", "Average of sensitivity and specificity"), ("＋", "TP · True Positive", "AI correctly said advance AND expert agreed"), ("−", "TN · True Negative", "AI correctly said reject AND expert agreed"), ("↗", "FP · False Positive", "AI said advance but expert rejected"), ("↘", "FN · False Negative", "AI said reject but expert accepted")]
    st.markdown('<section class="comparison-section"><h2 class="comparison-section-title">How to read the metrics</h2><p class="comparison-section-subtitle">These measures show why the evaluation is reliable and valid.</p><section class="explain-grid">' + ''.join(f'<article class="explain-card" title="{desc}"><div class="explain-icon">{icon}</div><strong>{title}</strong><span>{desc}</span></article>' for icon,title,desc in explanations) + '</section></section>', unsafe_allow_html=True)
    frame = pd.DataFrame(chart_rows)
    st.markdown('<section class="comparison-section"><h2 class="comparison-section-title">Performance Charts</h2><p class="comparison-section-subtitle">Compare accuracy and decision outcomes across the evaluated models.</p>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="medium")
    with left:
        fig = go.Figure([go.Bar(name="Balanced Accuracy", x=frame.Model, y=frame["Balanced Accuracy"], marker_color="#087f59"), go.Bar(name="Plain Accuracy", x=frame.Model, y=frame["Plain Accuracy"], marker_color="#83cbb2")])
        fig.add_hline(y=70, line_dash="dot", line_color="#df5656", annotation_text="70% benchmark"); fig.add_hline(y=80, line_dash="dot", line_color="#08a66d", annotation_text="80% benchmark")
        fig.update_layout(barmode="group", yaxis=dict(range=[0,100],ticksuffix="%"), height=330, margin=dict(l=10,r=10,t=12,b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h"))
        with st.container(key="comparison_chart_left"):
            st.markdown('<h3>Balanced vs Plain Accuracy per Model</h3><p class="subtle-note">Higher is better against the 70–80% validation band.</p>', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    with right:
        fig = go.Figure([go.Bar(name=key, x=frame.Model, y=frame[key], marker_color=color) for key,color in (("TP","#08b879"),("TN","#4a8ed8"),("FP","#e5a34b"),("FN","#df5656"))])
        fig.update_layout(barmode="group", height=330, margin=dict(l=10,r=10,t=12,b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h"))
        with st.container(key="comparison_chart_right"):
            st.markdown('<h3>TP vs TN vs FP vs FN per Model</h3><p class="subtle-note">Decision outcomes across the full dataset.</p>', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    st.markdown('</section>', unsafe_allow_html=True)
    st.markdown('<section class="comparison-section hidden-rank-comparison"><h2 class="comparison-section-title">AI Rank vs Expert Rank</h2><p class="comparison-section-subtitle">Closer to the diagonal means stronger agreement. Drag to pan or scroll to zoom.</p>', unsafe_allow_html=True)
    rank_data = data[["title", "expert_rank", "final_rank"]].copy()
    rank_data["expert_rank"] = pd.to_numeric(rank_data["expert_rank"], errors="coerce")
    rank_data["final_rank"] = pd.to_numeric(rank_data["final_rank"], errors="coerce")
    rank_data = rank_data.dropna(subset=["expert_rank", "final_rank"])
    scatter = go.Figure(go.Scatter(x=rank_data["expert_rank"], y=rank_data["final_rank"], mode="markers", text=rank_data["title"], hovertemplate="%{text}<br>Expert rank: %{x}<br>AI rank: %{y}<extra></extra>", marker=dict(size=9, color="#08b879", opacity=.62)))
    max_rank = max(rank_data["expert_rank"].max(), rank_data["final_rank"].max()) if not rank_data.empty else 1
    scatter.add_trace(go.Scatter(x=[1,max_rank], y=[1,max_rank], mode="lines", line=dict(color="#df5656", dash="dot"), name="Perfect agreement"))
    scatter.update_layout(height=390, margin=dict(l=10,r=10,t=10,b=25), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.55)", xaxis_title="Expert Rank", yaxis_title="AI Final Rank", legend=dict(orientation="h"))
    with st.container(key="scatter_plot"):
        st.plotly_chart(scatter, use_container_width=True, config={"scrollZoom":True, "displayModeBar":True})


def model_comparison_page():
    with st.container(key="model_comparison_page"):
        _model_comparison_page_content()


def _detail_page_content():
    data = dashboard_data.copy()
    st.markdown('<section class="comparison-hero"><div class="badge">RESEARCH EXPLORER</div><h1>Idea Detail</h1><p>Inspect one idea across every evaluation dimension and model.</p></section>', unsafe_allow_html=True)
    choices = data[["idea_id", "title"]].dropna().itertuples(index=False, name=None)
    choice_map = {f"#{int(idea_id)} · {title}": idea_id for idea_id, title in choices}
    current = st.session_state.get("detail_idea_id", int(data.iloc[0]["idea_id"]))
    current_label = next((label for label, idea_id in choice_map.items() if idea_id == current), next(iter(choice_map)))
    with st.container(key="detail-selector"):
        selected_label = st.selectbox("Select an idea to view", list(choice_map), index=list(choice_map).index(current_label))
    item = data.loc[data["idea_id"] == choice_map[selected_label]].iloc[0]
    st.session_state.detail_idea_id = int(item["idea_id"])
    agreement = int(item.get("agrees_with_expert", 0)) == 1
    status_text, status_class = ("✓ Agreement", "status-agree") if agreement else ("✕ Disagreement", "status-disagree")
    st.markdown('<section class="comparison-section"><div class="idea-header-grid">' + ''.join([
        f'<article class="idea-info"><small>Title</small><strong>{item["title"]}</strong></article>',
        f'<article class="idea-info"><small>Category</small><span class="tag">{item["category"]}</span></article>',
        f'<article class="idea-info"><small>Source</small><span>{item.get("source", "Dataset")}</span></article>',
        f'<article class="idea-info"><small>Expert Rank</small><strong>#{int(item["expert_rank"])}</strong></article>',
        f'<article class="idea-info"><small>AI Final Rank</small><strong>#{int(item["final_rank"])}</strong></article>',
        f'<article class="idea-info"><small>Status</small><strong class="{status_class}">{status_text}</strong></article>'
    ]) + '</div></section>', unsafe_allow_html=True)
    st.markdown(f'<section class="comparison-section"><article class="idea-description"><h3>Idea Description</h3>{item["description"]}</article></section>', unsafe_allow_html=True)
    model_info = [("Qwen", "qwen"), ("Mistral", "mistral"), ("Llama", "llama")]
    criteria = [("Novelty", "novelty"), ("Feasibility", "feasibility"), ("Impact", "impact"), ("Presentation", "presentation")]
    def criterion_value(model_key, suffix):
        column = f"{model_key}_{suffix}"
        if column in data.columns and pd.notna(item[column]):
            return float(item[column])
        available = [float(item[f"{model_key}_{candidate}"]) for _, candidate in criteria if f"{model_key}_{candidate}" in data.columns and pd.notna(item[f"{model_key}_{candidate}"])]
        return sum(available) / len(available) if available else float(item.get(f"{model_key}_overall", 0))
    chart_rows = [{"Model": name, **{label: criterion_value(key, suffix) for label, suffix in criteria}} for name, key in model_info]
    frame = pd.DataFrame(chart_rows)
    radar = go.Figure()
    for name, key in model_info:
        values = [criterion_value(key, suffix) for _, suffix in criteria]
        radar.add_trace(go.Scatterpolar(r=values + [values[0]], theta=[label for label, _ in criteria] + [criteria[0][0]], fill="toself", name=name, opacity=.65))
    radar.update_layout(polar=dict(radialaxis=dict(range=[0,4], dtick=1)), height=310, margin=dict(l=20,r=20,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h"))
    bars = go.Figure([go.Bar(name=name, x=frame.columns[1:], y=frame.iloc[index,1:]) for index, (name, _) in enumerate(model_info)])
    bars.update_layout(barmode="group", yaxis=dict(range=[0,4], dtick=1), height=310, margin=dict(l=20,r=20,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h"))
    st.markdown('<section class="comparison-section"><h2 class="comparison-section-title">Model Evaluation Breakdown</h2>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="medium")
    with left:
        with st.container(key="detail_radar"):
            st.markdown('<h3 class="detail-card-title">Model Evaluation Breakdown</h3>', unsafe_allow_html=True); st.plotly_chart(radar, use_container_width=True, config={"displayModeBar":False})
    with right:
        with st.container(key="detail_bars"):
            st.markdown('<h3 class="detail-card-title">Criteria Score Comparison</h3>', unsafe_allow_html=True); st.plotly_chart(bars, use_container_width=True, config={"displayModeBar":False})
    st.markdown('</section>', unsafe_allow_html=True)
    scores = [(name, float(item[f"{key}_overall"])) for name, key in model_info]
    scores.append(("Average", sum(value for _, value in scores) / len(scores)))
    progress = ''.join(f'<article class="score-card" title="{name}: {score:.2f} out of 4.0"><small>{name}</small><div class="score-number" style="color:{"#08b879" if score >= 3.7 else "#e3b341" if score >= 3.4 else "#df5656"}">{score:.2f}</div><div class="score-mini-track"><div class="score-mini-fill" style="width:{min(score/4*100,100):.1f}%;background:{"#08b879" if score >= 3.7 else "#e3b341" if score >= 3.4 else "#df5656"}"></div></div><span class="score-max">/ 4.0</span></article>' for name, score in scores)
    st.markdown(f'<section class="comparison-section"><h2 class="comparison-section-title">Overall Score</h2><section class="score-grid">{progress}</section></section>', unsafe_allow_html=True)


def detail_page():
    with st.container(key="detail_page"):
        _detail_page_content()


def _live_page_content():
    st.markdown('<section class="comparison-hero"><div class="badge">LIVE EVALUATION</div><h1>Live Evaluation Studio</h1><p>Evaluate a hackathon idea with the AI review panel.</p></section>', unsafe_allow_html=True)
    with st.container(key="live_form"):
        st.caption("Include the problem, intended users, solution, and expected impact for the most useful review.")
        title = st.text_input("Project Title", placeholder="e.g. MediScan AI")
        description = st.text_area("Project Description", placeholder="Describe your idea in 3-5 sentences...", height=170)
        evaluate_clicked = st.button("Evaluate This Idea", type="primary", use_container_width=True)
    if evaluate_clicked:
        if not title.strip() or not description.strip():
            st.warning("Please enter both a project title and description before evaluating.")
            return
        started = time.perf_counter()
        display_names = {key: MODEL_CONFIGS[key]["model"] for key in MODEL_CONFIGS}
        model_scores, feedback = {}, {}
        model_keys = ("qwen", "mistral", "llama")
        progress_slots = {key: st.empty() for key in model_keys}
        progress_bars = {key: st.progress(0, text=f"Waiting for {display_names[key]}...") for key in model_keys}

        with ThreadPoolExecutor(max_workers=len(model_keys)) as executor:
            futures = [executor.submit(evaluate_live_model, key, title, description) for key in model_keys]
            for completed, future in enumerate(as_completed(futures), start=1):
                model_key, criteria_scores, model_feedback = future.result()
                if criteria_scores is None:
                    progress_bars[model_key].progress(1.0, text=f"{display_names[model_key]} failed")
                    progress_slots[model_key].error(model_feedback)
                    continue
                model_scores[model_key] = criteria_scores
                feedback[model_key] = model_feedback
                progress_bars[model_key].progress(1.0, text=f"{display_names[model_key]} complete")

        if not model_scores:
            st.error("No model completed the evaluation.")
            return

        elapsed = time.perf_counter() - started
        overall = {key: calculate_overall(scores) for key, scores in model_scores.items()}
        combined = round(pd.Series(list(overall.values())).median() if ENSEMBLE_METHOD == "median" else sum(overall.values()) / len(overall), 2)
        st.markdown('<div class="section-title"><h2>Evaluation Results</h2></div>', unsafe_allow_html=True)
        columns = st.columns(4)
        for column, model_key in zip(columns[:3], ("qwen", "mistral", "llama")):
            value = f"{overall[model_key]:.2f} / 4.0" if model_key in overall else "Unavailable"
            column.metric(display_names[model_key], value)
        columns[3].metric("Combined", f"{combined:.2f} / 4.0")

        advances = combined >= 3.0 and all(scores[1] >= 3.0 and scores[2] >= 3.0 for scores in model_scores.values())
        color, background = ("#087f59", "#d9f7e9") if advances else ("#b42318", "#ffe2e0")
        decision = "THIS IDEA SHOULD ADVANCE" if advances else "THIS IDEA DOES NOT ADVANCE"
        # The advancement decision remains available to the business logic but is not shown in the UI.

        criteria = list(MODEL_WEIGHTS)
        rows = []
        for index, criterion in enumerate(criteria):
            values = {key: scores[index] for key, scores in model_scores.items()}
            rows.append({
                "Criterion": criterion.title(),
                "Qwen": values.get("qwen"), "Mistral": values.get("mistral"), "Llama": values.get("llama"),
                "Average": round(sum(values.values()) / len(values), 2),
            })
        st.markdown('<div class="section-title"><h2>Criteria Breakdown</h2></div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        for row in rows:
            st.progress(row["Average"] / 4, text=f'{row["Criterion"]}: {row["Average"]:.2f} / 4.0')

        st.markdown('<div class="section-title"><h2>Model Feedback</h2></div>', unsafe_allow_html=True)
        feedback_columns = st.columns(3)
        for column, model_key in zip(feedback_columns, ("qwen", "mistral", "llama")):
            with column:
                with st.container(key=f"live_feedback_{model_key}"):
                    st.markdown(f'#### {display_names[model_key]} says:')
                    st.markdown(feedback.get(model_key, "Feedback unavailable because this model did not complete."))

        rank = "N/A"
        try:
            rank = str(int((pd.to_numeric(dashboard_data["avg_overall"], errors="coerce") > combined).sum()) + 1)
        except Exception:
            pass
        st.caption(f"Estimated rank if added to dataset: #{rank} out of {metrics['ideas']} · Evaluation completed in {elapsed:.1f} seconds")


def live_page():
    with st.container(key="live_page"):
        _live_page_content()


if st.session_state.page == "live":
    live_page()
elif st.session_state.page == "leaderboard":
    leaderboard_page()
elif st.session_state.page == "comparison":
    model_comparison_page()
elif st.session_state.page == "detail":
    detail_page()
else:
    home_page()
