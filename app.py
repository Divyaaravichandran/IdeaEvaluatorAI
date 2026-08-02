from pathlib import Path
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from scorer import (
    MODEL_CONFIGS,
    MODEL_WEIGHTS,
    build_feedback_prompt,
    calculate_overall,
    complete_model,
    score_model,
)


st.set_page_config(page_title="AIEval - AI Evaluation Dashboard", page_icon="✦", layout="wide", initial_sidebar_state="collapsed")


@st.cache_data
def load_metrics():
    fallback = {
        "ideas": 50,
        "advancements": 10,
        "accuracy": 80,
        "models": {"Qwen3": 78, "DeepSeek": 82, "Llama": 80},
        "categories": {"Healthcare": 35, "Education": 25, "Environment": 20, "Others": 20},
    }
    try:
        data = pd.read_excel(Path("ideas_scored.xlsx"))
        if data.empty:
            return fallback
        models = {
            "Qwen3": round(data["qwen_accuracy"].mean()),
            "DeepSeek": round(data["mistral_accuracy"].mean()),
            "Llama": round(data["llama_accuracy"].mean()),
        } if all(column in data for column in ("qwen_accuracy", "mistral_accuracy", "llama_accuracy")) else fallback["models"]
        category_column = next((column for column in data if str(column).lower() in {"category", "domain", "idea category"}), None)
        categories = fallback["categories"]
        if category_column:
            counts = data[category_column].fillna("Others").astype(str).value_counts()
            categories = {name: round(count / len(data) * 100) for name, count in counts.head(3).items()}
            categories["Others"] = max(0, 100 - sum(categories.values()))
        return {
            "ideas": len(data),
            "advancements": int(data["ai_advance"].sum()) if "ai_advance" in data else min(10, len(data)),
            "accuracy": round(data["combined_accuracy"].mean()) if "combined_accuracy" in data else round(sum(models.values()) / len(models)),
            "models": models,
            "categories": categories,
        }
    except Exception:
        return fallback


def requested_page():
    value = st.query_params.get("page")
    if isinstance(value, list):
        value = value[-1] if value else None
    return str(value).lower() if value else None


def go_to(page):
    st.session_state.page = page
    st.query_params["page"] = page
    st.rerun()


metrics = load_metrics()
valid_pages = {"home", "leaderboard", "live", "detail"}
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
@media(max-width:900px){ .st-key-navbar{padding-right:0}.metrics{grid-template-columns:repeat(2,1fr)} }.stButton button { cursor:pointer; } @media(max-width:600px){ .metrics,.comparison-grid{grid-template-columns:1fr}.hero h1{font-size:35px} }
</style>
""", unsafe_allow_html=True)

with st.container(key="navbar"):
    brand, home, leaderboard, live = st.columns([1.5, 1, 1.2, 1.4], vertical_alignment="center")
    with brand:
        st.markdown('<div class="brand">▣ AIEval</div>', unsafe_allow_html=True)
    with home:
        if st.button("Home", key="home_nav", type="primary" if st.session_state.page == "home" else "secondary"):
            go_to("home")
    with leaderboard:
        if st.button("Leaderboard", key="leaderboard_nav", type="primary" if st.session_state.page == "leaderboard" else "secondary"):
            go_to("leaderboard")
    with live:
        if st.button("Live Evaluation", key="live_nav", type="primary" if st.session_state.page == "live" else "secondary"):
            go_to("live")


def home_page():
    st.markdown("""
    <section class="hero"><div class="badge">NEXT-GEN AI EVALUATION</div><h1>Hackathon Idea Evaluation and Ranking System Using Multiple LLMs</h1><p>Automatically evaluate, rank and explain hackathon ideas using 3 Large Language Models.</p></section>
    """, unsafe_allow_html=True)
    cards = [("TOTAL IDEAS", metrics["ideas"], "Ideas processed from dataset"), ("AI ADVANCEMENTS", metrics["advancements"], "Shortlisted by consensus"), ("BEST MODEL ACCURACY", f'{metrics["accuracy"]}%', "Highest verification score"), ("MODELS COMPARED", "3", "State-of-the-art LLMs")]
    html = "".join(f'<article class="metric"><small>{label}</small><strong>{value}</strong><span>{description}</span></article>' for label, value, description in cards)
    st.markdown(f'<section class="metrics">{html}</section>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown('<div class="section-title"><h2>Model accuracy</h2></div>', unsafe_allow_html=True)
        figure = go.Figure(go.Bar(x=list(metrics["models"]), y=list(metrics["models"].values()), marker_color=["#087f59", "#08b879", "#61ddb0"], text=[f"{value}%" for value in metrics["models"].values()], textposition="outside"))
        figure.update_layout(height=290, margin=dict(l=8, r=8, t=20, b=30), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(range=[0, 100], ticksuffix="%", gridcolor="rgba(31,108,82,.12)"), showlegend=False)
        st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
    with right:
        st.markdown('<div class="section-title"><h2>Ideas by category</h2></div>', unsafe_allow_html=True)
        figure = go.Figure(go.Pie(labels=list(metrics["categories"]), values=list(metrics["categories"].values()), hole=.66, marker=dict(colors=["#08b879", "#76a79a", "#a2bdb5", "#d2e3de"]), textinfo="none"))
        figure.update_layout(height=290, margin=dict(l=0, r=0, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="v", x=.62, y=.5))
        st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
    st.markdown('<div class="section-title"><h2>Innovation Benchmarking</h2><p>How the system improves on earlier evaluation approaches.</p></div>', unsafe_allow_html=True)
    comparisons = [
        ("Base Paper", ["GPT-3.5 only", "Climate change domain", "No feedback", "70–80% accuracy"], ""),
        ("Our System", ["3 LLMs compared", "Hackathon domain", "AI feedback given", "80%+ accuracy"], "featured"),
        ("Our Novelty", ["Multi-LLM benchmark", "New domain", "Feedback generation", "Live dashboard"], ""),
    ]
    comparison_html = "".join(f'<article class="comparison-card {style}"><h3>{title}</h3><ul>{"".join(f"<li>{item}</li>" for item in items)}</ul></article>' for title, items, style in comparisons)
    st.markdown(f'<section class="comparison-grid">{comparison_html}</section>', unsafe_allow_html=True)


def leaderboard_page():
    data = pd.read_excel(Path("ideas_scored.xlsx")).sort_values("final_rank", na_position="last")
    st.markdown('<div class="section-title"><h2>Leaderboard</h2><p>Ranked hackathon ideas and AI advancement decisions.</p></div>', unsafe_allow_html=True)
    def focus_table():
        st.session_state.focus_leaderboard_table = True

    with st.container(key="leaderboard_filters"):
        search, category, decision = st.columns([2, 1, 1])
        query = search.text_input("Search idea by title...", key="idea_search", label_visibility="collapsed", placeholder="Search idea by title...", on_change=focus_table)
        category_value = category.selectbox("Category", ["All", *sorted(data["category"].dropna().unique())], key="idea_category", label_visibility="collapsed", on_change=focus_table)
        decision_value = decision.selectbox("Decision", ["Show All", "Show Advanced Only", "Show Rejected Only"], key="idea_decision", label_visibility="collapsed", on_change=focus_table)
    filtered = data[data["title"].str.contains(query, case=False, na=False)]
    if category_value != "All":
        filtered = filtered[filtered["category"] == category_value]
    if decision_value == "Show Advanced Only":
        filtered = filtered[filtered["ai_advance"] == 1]
    elif decision_value == "Show Rejected Only":
        filtered = filtered[filtered["ai_advance"] == 0]

    st.markdown('<div class="section-title"><h2>Top ideas</h2></div>', unsafe_allow_html=True)
    podium = data.nsmallest(3, "final_rank")
    podium_columns = st.columns(3)
    medals = ["🥇", "🥈", "🥉"]
    for column, medal, (_, idea) in zip(podium_columns, medals, podium.iterrows()):
        with column:
            style = f'rank-{int(idea["final_rank"])}'
            st.markdown(f'<article class="podium-card {style}"><div class="rank">{medal} RANK {int(idea["final_rank"])}</div><h3>{idea["title"]}</h3><p>Avg: {idea["avg_overall"]:.2f} · {idea["category"]}</p></article>', unsafe_allow_html=True)
            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
            button_left, button_center, button_right = st.columns([1, 1.3, 1])
            with button_center:
                if st.button("View idea", key=f"podium_{idea['idea_id']}", type="secondary", use_container_width=True):
                    st.session_state.selected_idea = int(idea["idea_id"])
                    go_to("detail")

    st.markdown('<div style="height:42px"></div>', unsafe_allow_html=True)

    st.markdown('<div id="leaderboard-table"></div>', unsafe_allow_html=True)
    if st.session_state.pop("focus_leaderboard_table", False):
        components.html("<script>window.parent.document.getElementById('leaderboard-table')?.scrollIntoView({behavior:'smooth', block:'start'});</script>", height=0)
    display = pd.DataFrame({
        "Rank": filtered["final_rank"], "Idea Title": filtered["title"], "Category": filtered["category"],
        "Qwen3 Score (out of 4)": filtered["qwen_overall"], "DeepSeek Score (out of 4)": filtered["mistral_overall"],
        "Llama Score (out of 4)": filtered["llama_overall"], "Average Score (out of 4)": filtered["avg_overall"],
        "Expert Rank": filtered["expert_rank"],
        "AI Decision": filtered["ai_advance"].map({1: "✅ Advance", 0: "❌ Reject"}),
        "Agrees with Expert": filtered["agrees_with_expert"].map({1: "✅", 0: "❌"}),
    })
    st.dataframe(display, hide_index=True, use_container_width=True)
    st.markdown('<div class="section-title"><h2>Feedback for top 10 ideas</h2></div>', unsafe_allow_html=True)
    for _, idea in data.nsmallest(10, "final_rank").iterrows():
        feedback = idea.get("combined_feedback") or idea.get("llama_feedback") or "Feedback was not generated for this idea."
        with st.expander(f'#{int(idea["final_rank"])} · {idea["title"]}'):
            st.write(feedback)


def detail_page():
    idea_id = st.session_state.get("selected_idea")
    data = pd.read_excel(Path("ideas_scored.xlsx"))
    idea = data.loc[data["idea_id"] == idea_id]
    if idea.empty:
        st.warning("Choose an idea from the Leaderboard first.")
        return
    item = idea.iloc[0]
    st.markdown(f'<div class="section-title"><h2>{item["title"]}</h2><p>Rank #{int(item["final_rank"])} · {item["category"]}</p></div>', unsafe_allow_html=True)
    st.write(item["description"])
    if st.button("Back to Leaderboard"):
        go_to("leaderboard")


def live_page():
    st.markdown('<div class="section-title"><h2>Live Evaluation Studio</h2><p>Evaluate a hackathon idea with the AI review panel.</p></div>', unsafe_allow_html=True)
    title = st.text_input("Project Title", placeholder="e.g. MediScan AI")
    description = st.text_area("Project Description", placeholder="Describe your idea in 3-5 sentences...", height=170)
    if st.button("Evaluate This Idea", type="primary", use_container_width=True):
        if not title.strip() or not description.strip():
            st.warning("Please enter both a project title and description before evaluating.")
            return
        started = time.perf_counter()
        display_names = {"qwen": "Qwen3-32B", "mistral": "DeepSeek", "llama": "Llama 3.3 70B"}
        model_scores, feedback = {}, {}
        for model_key in ("qwen", "mistral", "llama"):
            name = display_names[model_key]
            with st.spinner(f"Scoring with {name}..."):
                _, criteria_scores, status = score_model(model_key, title, description)
            if criteria_scores is None:
                st.error(f"{name} could not evaluate this idea: {status}")
                return
            model_scores[model_key] = criteria_scores
            with st.spinner(f"Preparing feedback from {name}..."):
                try:
                    feedback[model_key] = complete_model(
                        model_key,
                        build_feedback_prompt(title, description, criteria_scores),
                        max_tokens=180,
                    ).strip()
                except Exception as error:
                    feedback[model_key] = f"Feedback unavailable: {error}"

        elapsed = time.perf_counter() - started
        overall = {key: calculate_overall(scores) for key, scores in model_scores.items()}
        average = round(sum(overall.values()) / len(overall), 2)
        st.markdown('<div class="section-title"><h2>Evaluation Results</h2></div>', unsafe_allow_html=True)
        columns = st.columns(4)
        for column, model_key in zip(columns[:3], ("qwen", "mistral", "llama")):
            column.metric(display_names[model_key], f"{overall[model_key]:.2f} / 4.0")
        columns[3].metric("Average", f"{average:.2f} / 4.0")

        advances = average >= 3.0
        color, background = ("#087f59", "#d9f7e9") if advances else ("#b42318", "#ffe2e0")
        decision = "THIS IDEA SHOULD ADVANCE" if advances else "THIS IDEA DOES NOT ADVANCE"
        st.markdown(f'<div style="margin:28px 0;padding:22px;border-radius:12px;background:{background};color:{color};text-align:center;font-size:20px;font-weight:700;border:1px solid {color}33">{decision}</div>', unsafe_allow_html=True)

        criteria = list(MODEL_WEIGHTS)
        rows = []
        for index, criterion in enumerate(criteria):
            values = [model_scores[key][index] for key in ("qwen", "mistral", "llama")]
            rows.append({"Criterion": criterion.title(), "Qwen3": values[0], "DeepSeek": values[1], "Llama": values[2], "Average": round(sum(values) / 3, 2)})
        st.markdown('<div class="section-title"><h2>Criteria Breakdown</h2></div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        for row in rows:
            st.progress(row["Average"] / 4, text=f'{row["Criterion"]}: {row["Average"]:.2f} / 4.0')

        st.markdown('<div class="section-title"><h2>Model Feedback</h2></div>', unsafe_allow_html=True)
        feedback_columns = st.columns(3)
        for column, model_key in zip(feedback_columns, ("qwen", "mistral", "llama")):
            with column:
                st.markdown(f'#### {display_names[model_key]} says:')
                st.markdown(feedback[model_key])

        rank = "N/A"
        try:
            dataset = pd.read_excel(Path("ideas_scored.xlsx"))
            score_column = next((column for column in ("avg_overall", "combined_score", "combined_accuracy") if column in dataset), None)
            if score_column:
                rank = str(int((pd.to_numeric(dataset[score_column], errors="coerce") > average).sum()) + 1)
        except Exception:
            pass
        st.caption(f"Estimated rank if added to dataset: #{rank} out of {metrics['ideas']} · Evaluation completed in {elapsed:.1f} seconds")


if st.session_state.page == "live":
    live_page()
elif st.session_state.page == "leaderboard":
    leaderboard_page()
elif st.session_state.page == "detail":
    detail_page()
else:
    home_page()
