from __future__ import annotations

import html
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# CrowdGuard is intentionally a single-file demo so it is easy to run at a hackathon.
ROOT = Path(os.environ.get("CROWDGUARD_ROOT", Path(__file__).resolve().parent)).resolve()
CSV_PATH = ROOT / "output" / "timeline.csv"
VIDEO_PATH = ROOT / "data" / "videos" / "crowd.mp4"
ANNOTATED_VIDEO_PATH = ROOT / "output" / "annotated.mp4"
ANNOTATED_WEBM_PATH = ROOT / "output" / "annotated.webm"
FRAME_DIR = ROOT / "output" / "frames"
DEFAULT_ZONE_NAMES = ["North Entry", "Queue", "West Walkway", "South Walkway", "Gate 4", "Concourse", "Exit Lane", "Gate 2", "East Walkway"]
ZONE_NAMES = DEFAULT_ZONE_NAMES.copy()
ALERT_TYPES = [
    "Crush Risk",
    "Abandoned Object",
    "Loitering",
    "Restricted Zone",
    "Panic Running",
    "Wrong-Way Movement",
    "Fallen Person",
]
SEVERITIES = ["Low", "Medium", "High"]
GUIDE = [
    ("This is the live camera feed. YOLOv8 detects and tracks every person.", 12, "feed"),
    ("The frame is split into zones. We measure density, speed, counter-flow and chaos in each.", 40, "zones"),
    ("The system learned what normal looks like in the first 60 seconds.", 55, "chart"),
    ("When behavior deviates from normal, risk rises. No 'stampede' label was needed.", 115, "chart"),
    ("At high risk, CrowdGuard raises an alert with evidence and a recommended action.", 150, "alerts"),
    ("Suspicious-behavior alerts show the observed evidence and why it was flagged.", 128, "alerts"),
    ("The crowd management panel compares occupancy, queue waits and safer exits.", 145, "management"),
    ("Limits: early-warning indicators for human operators, not a certain prediction.", 150, "footer"),
]

st.set_page_config(page_title="CrowdGuard | NEMESIS-X", page_icon="🛡️", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    :root {
        --cg-bg: #10191c;
        --cg-sidebar: #142226;
        --cg-card: #18272a;
        --cg-text: #e7f0ef;
        --cg-muted: #a7bcba;
        --cg-teal: #63d2c3;
        --cg-green: #65d6a5;
        --cg-yellow: #f0c65b;
        --cg-orange: #f0a35b;
        --cg-red: #ff7777;
    }
    html { scroll-behavior: smooth; }
    *, *::before, *::after { box-sizing: border-box; }
    ::selection { color: #10191c; background: #63d2c3; }
    ::-webkit-scrollbar { width: 9px; height: 9px; }
    ::-webkit-scrollbar-track { background: #10191c; }
    ::-webkit-scrollbar-thumb { background: #397f78; border: 2px solid #10191c; border-radius: 999px; }
    .stApp {
        background: #10191c;
        color: #e7f0ef;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        animation: cgFadeIn 320ms ease-out both;
    }
    @keyframes cgFadeIn { from { opacity: .78; } to { opacity: 1; } }
    [data-testid="stMainBlockContainer"] { max-width: 1920px; padding-top: 1rem; padding-bottom: 2rem; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #142226 0%, #10191c 100%);
        border-right: 1px solid rgba(231, 240, 239, .08);
    }
    header[data-testid="stHeader"] { background: #10191c; }
    header[data-testid="stHeader"] [data-testid="stToolbar"] { background: transparent; color: #e7f0ef; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.25rem; }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 1rem; }
    h1, h2, h3, p, label { color: #e7f0ef; font-family: 'Inter', 'Segoe UI', sans-serif; }
    .stApp [data-testid="stHeading"] h1,
    .stApp [data-testid="stHeading"] h2,
    .stApp [data-testid="stHeading"] h3,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stAlert"] *,
    .stApp .cg-card,
    .stApp .cg-card *,
    .stApp .cg-alert,
    .stApp .cg-alert *,
    .stApp .cg-guide,
    .stApp .cg-guide * { font-family: 'Inter', 'Segoe UI', sans-serif !important; }
    h1 { font-size: 2.15rem !important; line-height: 1.2 !important; margin-bottom: .25rem !important; font-weight: 750 !important; }
    h2 { font-size: 1.35rem !important; line-height: 1.35 !important; font-weight: 700 !important; }
    h3 { font-size: 1.15rem !important; line-height: 1.4 !important; font-weight: 700 !important; }
    [data-testid="stMarkdownContainer"] p, [data-testid="stCaptionContainer"] { font-size: 16px; line-height: 1.55; }
    .cg-subtitle { color: #a7bcba; font-size: 16px; line-height: 1.5; margin: 0 0 1.5rem; }
    .cg-card {
        position: relative;
        isolation: isolate;
        overflow: hidden;
        height: 100%;
        min-height: 120px;
        padding: 20px 24px;
        border: 1px solid rgba(231, 240, 239, .08);
        border-radius: 16px;
        background: linear-gradient(180deg, rgba(231, 240, 239, .045) 0%, rgba(231, 240, 239, .008) 100%), #18272a;
        box-shadow: 0 10px 28px rgba(0, 0, 0, .22), inset 0 1px 0 rgba(231, 240, 239, .035);
        transition: border-color 200ms ease, box-shadow 200ms ease, transform 200ms ease, background-color 200ms ease, color 200ms ease;
    }
    .cg-card::before {
        content: "";
        position: absolute;
        inset: 0 0 auto;
        height: 2px;
        background: #63d2c3;
        opacity: .85;
        transition: background-color 200ms ease, opacity 200ms ease;
    }
    .cg-card:has(.cg-value[style*="#65d6a5"])::before { background: #65d6a5; }
    .cg-card:has(.cg-value[style*="#f0c65b"])::before { background: #f0c65b; }
    .cg-card:has(.cg-value[style*="#ff7777"])::before { background: #ff7777; }
    .stHorizontalBlock:has(.cg-card) { align-items: stretch; }
    [data-testid="stMarkdownContainer"]:has(.cg-card) { height: 100%; }
    .cg-label { color: #b7c8c6; font-size: 12px; line-height: 1.4; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }
    .cg-value { color: #f0f7f5; font-size: 38px; line-height: 1.12; font-weight: 750; margin-top: 8px; font-variant-numeric: tabular-nums lining-nums; letter-spacing: 0; transition: color 200ms ease; }
    .cg-value > span { color: #afc0be; font-size: 16px !important; line-height: 1.2; font-weight: 600; }
    .cg-detail { color: #afc0be; font-size: 16px; line-height: 1.45; margin-top: 7px; font-variant-numeric: tabular-nums; }
    .cg-badge { display: inline-flex; align-items: center; min-height: 28px; color: #122225; background: #63d2c3; font-weight: 800; border-radius: 999px; padding: 5px 12px; font-size: 12px; letter-spacing: .06em; }
    .cg-guide { border-left: 4px solid #63d2c3; background: linear-gradient(180deg, rgba(231, 240, 239, .035), rgba(231, 240, 239, .008)), #18272a; border-radius: 12px; padding: 16px 20px; margin: 12px 0 18px; box-shadow: 0 8px 22px rgba(0, 0, 0, .18); }
    .cg-guide-step { color: #63d2c3; font-size: 12px; line-height: 1.4; font-weight: 800; text-transform: uppercase; letter-spacing: .08em; }
    .cg-guide-text { color: #e7f0ef; font-size: 16px; line-height: 1.55; margin-top: 7px; }
    .cg-focus { color: #63d2c3; font-size: 12px; line-height: 1.4; font-weight: 800; letter-spacing: .08em; }
    .cg-alert { position: relative; overflow: hidden; border-radius: 12px; padding: 14px 16px 14px 20px; margin: 4px 0 10px; color: #f4f6f5; transition: transform 200ms ease, box-shadow 200ms ease, border-color 200ms ease; }
    .cg-alert::before { content: ""; position: absolute; inset: 0 auto 0 0; width: 4px; background: #63d2c3; }
    .cg-alert-high { border: 1px solid rgba(255, 119, 119, .34); background: linear-gradient(180deg, rgba(255, 119, 119, .08), rgba(255, 119, 119, .025)), #3a2023; }
    .cg-alert-high::before { background: #ff7777; }
    .cg-alert-medium { border: 1px solid rgba(240, 163, 91, .34); background: linear-gradient(180deg, rgba(240, 163, 91, .08), rgba(240, 163, 91, .025)), #38291e; }
    .cg-alert-medium::before { background: #f0a35b; }
    .cg-alert-low { border: 1px solid rgba(240, 198, 91, .34); background: linear-gradient(180deg, rgba(240, 198, 91, .08), rgba(240, 198, 91, .025)), #342f1d; }
    .cg-alert-low::before { background: #f0c65b; }
    .cg-alert:hover { transform: translateY(-2px); box-shadow: 0 12px 28px rgba(0, 0, 0, .3); }
    .cg-alert-selected { outline: 2px solid rgba(99, 210, 195, .72); outline-offset: 2px; box-shadow: 0 0 0 4px rgba(99, 210, 195, .1); }
    .cg-alert-title { color: #f4f6f5; font-size: 16px; line-height: 1.45; font-weight: 800; margin-bottom: 8px; }
    .cg-severity-high { color: #ffaaaa; background: rgba(255, 119, 119, .14); border: 1px solid rgba(255, 119, 119, .26); }
    .cg-severity-medium { color: #ffd0a1; background: rgba(240, 163, 91, .14); border: 1px solid rgba(240, 163, 91, .26); }
    .cg-severity-low { color: #f5d77f; background: rgba(240, 198, 91, .14); border: 1px solid rgba(240, 198, 91, .26); }
    .cg-time, .cg-zone { color: #b7f0e9; background: rgba(99, 210, 195, .10); border: 1px solid rgba(99, 210, 195, .2); }
    .cg-source { display: inline-flex; align-items: center; min-height: 24px; margin-left: 6px; border: 1px solid rgba(99, 210, 195, .5); color: #b7f0e9; background: rgba(99, 210, 195, .12); border-radius: 999px; padding: 3px 9px; font-size: 12px; font-weight: 700; vertical-align: middle; }
    .cg-alert-line { font-size: 16px; line-height: 1.55; color: #f0f5f4; }
    .cg-chip { display: inline-flex; align-items: center; min-height: 26px; padding: 4px 10px; border-radius: 999px; font-weight: 700; font-size: 12px; line-height: 1.2; font-variant-numeric: tabular-nums; }
    .cg-normal { color: #9ce8c3; background: rgba(101, 214, 165, .14); border: 1px solid rgba(101, 214, 165, .25); }
    .cg-busy { color: #f5d77f; background: rgba(240, 198, 91, .14); border: 1px solid rgba(240, 198, 91, .25); }
    .cg-critical { color: #ffaaaa; background: rgba(255, 119, 119, .14); border: 1px solid rgba(255, 119, 119, .25); }
    .cg-footer { color: #afc0be; border-top: 1px solid rgba(231, 240, 239, .10); padding-top: 16px; font-size: 16px; line-height: 1.5; }
    .cg-notice { border: 1px solid rgba(99, 210, 195, .3); border-left: 4px solid #63d2c3; border-radius: 12px; background: linear-gradient(180deg, rgba(99, 210, 195, .07), rgba(99, 210, 195, .02)), #19302f; color: #dff4ef; padding: 14px 18px; margin: 0 0 16px; font-size: 16px; line-height: 1.5; font-weight: 600; }
    div[data-testid="stMetric"] { background: linear-gradient(180deg, rgba(231, 240, 239, .04), rgba(231, 240, 239, .008)), #18272a; border: 1px solid rgba(231, 240, 239, .08); padding: 16px; border-radius: 16px; box-shadow: 0 8px 22px rgba(0, 0, 0, .2); }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.cg-alert) { border: 1px solid rgba(231, 240, 239, .08); border-radius: 16px; background: linear-gradient(180deg, rgba(231, 240, 239, .025), rgba(231, 240, 239, .004)), #142226; box-shadow: 0 8px 22px rgba(0, 0, 0, .18); transition: border-color 200ms ease, box-shadow 200ms ease, transform 200ms ease; }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.cg-alert):hover { border-color: rgba(231, 240, 239, .14); box-shadow: 0 12px 28px rgba(0, 0, 0, .28); transform: translateY(-2px); }
    [data-testid="stVideo"], [data-testid="stImage"] { overflow: hidden; border: 1px solid rgba(231, 240, 239, .10); border-radius: 16px; background: #142226; box-shadow: 0 10px 26px rgba(0, 0, 0, .24), inset 0 1px 0 rgba(231, 240, 239, .04); }
    [data-testid="stVideo"] video, [data-testid="stImage"] img { border-radius: 14px; }
    [data-testid="stImage"] figcaption { color: #e7f0ef; font-size: 14px; line-height: 1.4; }
    [data-testid="stDataFrame"] { overflow: hidden; border: 1px solid rgba(231, 240, 239, .08); border-radius: 14px; background: #18272a; box-shadow: 0 8px 22px rgba(0, 0, 0, .2); }
    [data-testid="stDataFrame"] .stDataFrameGlideDataEditor {
        --gdg-link-color: #63d2c3 !important;
        --gdg-font-family: 'Inter', 'Segoe UI', sans-serif !important;
        --gdg-base-font-style: 400 16px !important;
        --gdg-header-font-style: 600 14px !important;
        --gdg-marker-font-style: 12px !important;
        --gdg-editor-font-size: 1rem !important;
        --gdg-bg-cell: #18272a !important;
        --gdg-bg-cell-medium: #142226 !important;
        --gdg-bg-header: #142226 !important;
        --gdg-bg-header-hover: #18272a !important;
        --gdg-bg-header-hovered: #18272a !important;
        --gdg-bg-header-has-focus: #18272a !important;
        --gdg-bg-group-header: #142226 !important;
        --gdg-bg-group-header-hovered: #18272a !important;
        --gdg-bg-bubble: #142226 !important;
        --gdg-bg-bubble-selected: rgba(99, 210, 195, .16) !important;
        --gdg-bg-search-result: rgba(255, 119, 119, .14) !important;
        --gdg-bg-icon-header: rgba(231, 240, 239, .72) !important;
        --gdg-text-dark: #e7f0ef !important;
        --gdg-text-medium: #afc0be !important;
        --gdg-text-light: #91a5a4 !important;
        --gdg-text-header: #e7f0ef !important;
        --gdg-text-header-selected: #e7f0ef !important;
        --gdg-text-group-header: #afc0be !important;
        --gdg-text-bubble: #e7f0ef !important;
        --gdg-border-color: rgba(231, 240, 239, .10) !important;
        --gdg-horizontal-border-color: rgba(231, 240, 239, .10) !important;
        --gdg-drilldown-border: rgba(231, 240, 239, .14) !important;
        --gdg-accent-color: #63d2c3 !important;
        --gdg-accent-light: rgba(99, 210, 195, .2) !important;
        --gdg-accent-fg: #10191c !important;
        --gdg-fg-icon-header: #e7f0ef !important;
        --gdg-resize-indicator-color: #ff7777 !important;
        --gdg-cell-vertical-padding: 6px !important;
        border-radius: 13px;
    }
    [data-testid="stPlotlyChart"] { border: 1px solid rgba(231, 240, 239, .07); border-radius: 16px; background: linear-gradient(180deg, rgba(231, 240, 239, .018), transparent); padding: 4px; }
    [data-testid="stPlotlyChart"] .main-svg { font-family: 'Inter', 'Segoe UI', sans-serif !important; }
    [data-testid="stPlotlyChart"] .legend { filter: drop-shadow(0 4px 10px rgba(0, 0, 0, .22)); }
    .js-plotly-plot .heatmaplayer .hm { stroke-linejoin: round; transition: filter 180ms ease; }
    .js-plotly-plot .heatmaplayer .hm:hover { filter: brightness(1.12); }
    [data-testid="stButton"] button, [data-testid="stDownloadButton"] button { min-height: 40px; border-radius: 10px; font-family: 'Inter', 'Segoe UI', sans-serif !important; font-size: 16px; font-weight: 650; transition: background-color 180ms ease, border-color 180ms ease, box-shadow 180ms ease, transform 180ms ease; }
    [data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover { border-color: rgba(99, 210, 195, .65); box-shadow: 0 4px 14px rgba(0, 0, 0, .22); transform: translateY(-1px); }
    button:focus-visible, input:focus-visible, [role="slider"]:focus-visible { outline: 2px solid #63d2c3 !important; outline-offset: 2px !important; box-shadow: 0 0 0 4px rgba(99, 210, 195, .18) !important; }
    [data-baseweb="input"] > div, [data-baseweb="select"] > div, [data-baseweb="textarea"] { border-radius: 10px !important; border-color: rgba(231, 240, 239, .14) !important; background-color: #142226 !important; }
    [data-testid="stSlider"] [role="slider"] { background: #63d2c3 !important; border-color: #63d2c3 !important; box-shadow: 0 0 0 4px rgba(99, 210, 195, .16); }
    [data-testid="stSlider"] [data-baseweb="slider"] > div:first-child { background: rgba(99, 210, 195, .24); }
    [data-testid="stProgress"] > div { border-radius: 999px; overflow: hidden; background: #142226; }
    [data-testid="stProgress"] [data-testid="stProgressBar"] { border-radius: 999px; background: #63d2c3; }
    [data-testid="stExpander"] { border-color: rgba(231, 240, 239, .09) !important; border-radius: 12px !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(231, 240, 239, .10); margin: 16px 0; }
    #MainMenu, footer { visibility: hidden; }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; scroll-behavior: auto !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def make_demo_data() -> pd.DataFrame:
    """Create deterministic 180-second, nine-zone simulated crowd observations."""
    rng = np.random.default_rng(20261125)
    rows = []
    for sec in range(180):
        rise = float(np.clip((sec - 84) / 66, 0, 1))
        if sec > 150:
            rise *= max(0.35, 1 - (sec - 150) / 65)
        for zone in range(9):
            gate = zone in (4, 7)
            base_count = 28 + (zone * 5 % 22)
            count = max(0, int(base_count + rise * (102 if gate else 9) + rng.normal(0, 3.5)))
            speed = float(np.clip(1.34 - rise * (1.03 if gate else 0.12) + rng.normal(0, 0.07), 0.2, 1.8))
            counterflow = float(np.clip(0.045 + rise * (0.46 if gate else 0.035) + rng.normal(0, 0.018), 0, 0.85))
            chaos = float(np.clip(12 + rise * (75 if gate else 10) + rng.normal(0, 4), 0, 100))
            density_score = np.clip(count / 140 * 100, 0, 100)
            speed_score = np.clip((1.4 - speed) / 1.2 * 100, 0, 100)
            risk = np.clip(0.32 * density_score + 0.22 * speed_score + 0.20 * counterflow * 100 + 0.26 * chaos, 0, 100)
            rows.append((sec, zone, count, speed, counterflow, chaos, risk))
    return pd.DataFrame(rows, columns=["sec", "zone", "count", "speed", "counterflow", "chaos", "risk"])


@st.cache_data(show_spinner=False)
def load_real_data(csv_path: str) -> pd.DataFrame:
    required = ["sec", "zone", "count", "speed", "counterflow", "chaos", "risk"]
    frame = pd.read_csv(csv_path)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    frame = frame[required].copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=required)
    frame["sec"] = frame["sec"].astype(int)
    frame["zone"] = frame["zone"].astype(int)
    frame = frame[frame["zone"].between(0, 8)]
    if frame.empty:
        raise ValueError("The timeline contains no valid zone observations.")
    if frame["risk"].max() <= 1:
        frame["risk"] *= 100
    frame["risk"] = frame["risk"].clip(0, 100)
    return frame.sort_values(["sec", "zone"])


@st.cache_data(show_spinner=False)
def load_config(config_path: str) -> tuple[dict, str | None]:
    if not Path(config_path).is_file():
        return {}, None
    try:
        with Path(config_path).open(encoding="utf-8") as config_file:
            config = json.load(config_file)
        if not isinstance(config, dict):
            return {}, "config.json must contain a JSON object; using built-in defaults."
        return config, None
    except (OSError, json.JSONDecodeError) as error:
        return {}, f"Could not read config.json ({error}); using built-in defaults."


def zone_config_value(config: dict, zone: int, field: str, aliases: tuple[str, ...] = ()):
    sources = [config.get("zones")]
    if field == "capacity":
        sources.extend([config.get("zone_capacities"), config.get("capacities")])
    elif field == "name":
        sources.extend([config.get("zone_names")])
    for source in sources:
        if isinstance(source, list) and zone < len(source):
            value = source[zone]
            if isinstance(value, dict):
                for key in (field, *aliases):
                    if key in value:
                        return value[key]
            elif field == "name" and isinstance(value, str):
                return value
            elif field == "capacity" and isinstance(value, (int, float)):
                return value
        if isinstance(source, dict):
            if "0" in source:
                keys = (str(zone), f"Zone {zone + 1}", str(zone + 1))
            elif "9" in source:
                keys = (str(zone + 1), f"Zone {zone + 1}", str(zone))
            else:
                keys = (f"Zone {zone + 1}", str(zone), str(zone + 1))
            value = next((source[key] for key in keys if key in source), None)
            if isinstance(value, dict):
                for key in (field, *aliases):
                    if key in value:
                        return value[key]
            elif value is not None:
                return value
    return None


def apply_config_defaults(config: dict) -> tuple[list[int], float, float]:
    ZONE_NAMES[:] = DEFAULT_ZONE_NAMES
    default_capacities = []
    for zone in range(9):
        configured_name = zone_config_value(config, zone, "name", ("zone_name",))
        if configured_name:
            ZONE_NAMES[zone] = str(configured_name)
        configured_capacity = zone_config_value(config, zone, "capacity", ("people_capacity",))
        try:
            default_capacities.append(max(10, int(configured_capacity)))
        except (TypeError, ValueError):
            default_capacities.append(150)
    thresholds = config.get("thresholds", {})
    if not isinstance(thresholds, dict):
        thresholds = {}
    try:
        low_threshold = float(thresholds.get("low", thresholds.get("low_risk", thresholds.get("risk_low", 40))))
        high_threshold = float(thresholds.get("high", thresholds.get("high_risk", thresholds.get("risk_high", 70))))
    except (TypeError, ValueError):
        low_threshold, high_threshold = 40.0, 70.0
    low_threshold = float(np.clip(low_threshold, 0, 100))
    high_threshold = float(np.clip(high_threshold, low_threshold, 100))
    return default_capacities, low_threshold, high_threshold


def percent_counterflow(value: float) -> float:
    return float(value * 100 if value <= 1 else value)


def fmt_time(seconds: int | float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def nearest_frame(seconds: int) -> Path | None:
    exact = FRAME_DIR / f"sec_{seconds}.jpg"
    if exact.is_file():
        return exact
    if not FRAME_DIR.is_dir():
        return None
    candidates = []
    for path in FRAME_DIR.glob("sec_*.jpg"):
        try:
            candidates.append((abs(int(path.stem.removeprefix("sec_")) - seconds), path))
        except ValueError:
            continue
    return min(candidates, default=(None, None), key=lambda item: item[0])[1]


def explain_with_ai(frame: str | None, metrics: dict) -> str:
    """Template placeholder for a future vision model explanation call."""
    return (
        f"At {fmt_time(metrics['sec'])}, {metrics['zone_name']} reached {metrics['risk']:.0f} risk "
        f"with {metrics['count']:.0f} people, speed {metrics['speed']:.2f} px/s, and "
        f"{metrics['counterflow']:.0f}% counter-flow. Recommended action: {metrics['action']}"
    )


def make_alert(alert_id: str, alert_type: str, severity: str, sec: int, zone: int, what: str, why: str, action: str, evidence_frames: list[str], metrics: dict) -> dict:
    return {
        "id": alert_id,
        "type": alert_type,
        "severity": severity,
        "sec": int(sec),
        "zone": int(zone),
        "what": what,
        "why_flagged": why,
        "evidence_frames": evidence_frames,
        "recommended_action": action,
        "metrics": metrics,
        "explanation_source": "template",
    }


def demo_alerts(data: pd.DataFrame) -> list[dict]:
    # The seven synthetic incidents intentionally span the requested alert taxonomy.
    plan = [
        (96, 4, "Crush Risk", "Medium"),
        (109, 7, "Wrong-Way Movement", "Medium"),
        (123, 4, "Abandoned Object", "Low"),
        (137, 7, "Restricted Zone", "Medium"),
        (149, 4, "Panic Running", "High"),
        (160, 7, "Fallen Person", "High"),
        (171, 4, "Loitering", "Low"),
    ]
    examples = {
        "Crush Risk": ("Queue density surged", "Pause entry and open Gate 4"),
        "Wrong-Way Movement": ("Counter-flow built against the queue", "Redirect the opposing flow at Gate 2"),
        "Abandoned Object": ("A stationary item remained in a busy walkway", "Dispatch staff to inspect the item"),
        "Restricted Zone": ("Movement entered a staff-only boundary", "Verify access and secure the boundary"),
        "Panic Running": ("Running and elevated motion chaos detected", "Deploy stewards and slow incoming arrivals"),
        "Fallen Person": ("A person appears down in a dense zone", "Send a response team and create space"),
        "Loitering": ("A person remained in the gate approach", "Ask a steward to check the obstruction"),
    }
    alerts = []
    for sec, zone, alert_type, severity in plan:
        row = data[(data.sec == sec) & (data.zone == zone)].iloc[0]
        previous = data[(data.sec == max(0, sec - 20)) & (data.zone == zone)].iloc[0]
        count_change = row["count"] / max(1, previous["count"])
        speed_drop = max(0, (previous["speed"] - row["speed"]) / max(previous["speed"], 0.1) * 100)
        flow_pct = percent_counterflow(float(row["counterflow"]))
        what_base, action = examples[alert_type]
        what = (
            f"{what_base}: {row['count']:.0f} people, {row['speed']:.2f} px/s, "
            f"{flow_pct:.0f}% counter-flow (risk {row['risk']:.0f})."
        )
        why = (
            f"Density is {count_change:.1f}x its 20-second baseline; speed changed by "
            f"-{speed_drop:.0f}%; counter-flow is {flow_pct:.0f}% and chaos is {row['chaos']:.0f}/100."
        )
        keyframe = FRAME_DIR / f"sec_{sec}.jpg"
        evidence_frames = [str(keyframe)] if keyframe.is_file() else []
        metrics = {
            "count": int(row["count"]),
            "speed": round(float(row["speed"]), 2),
            "counterflow_percent": round(flow_pct, 1),
            "chaos": round(float(row["chaos"]), 1),
            "risk": round(float(row["risk"]), 1),
            "density_change_20s": round(float(count_change), 2),
            "speed_drop_percent_20s": round(float(speed_drop), 1),
        }
        alerts.append(make_alert(f"demo-{sec}-{zone}", alert_type, severity, sec, zone, what, why, action, evidence_frames, metrics))
    return alerts


def normalize_alerts(raw_alerts: list) -> tuple[list[dict], int]:
    alerts = []
    skipped = 0
    for index, item in enumerate(raw_alerts):
        if not isinstance(item, dict):
            skipped += 1
            continue
        try:
            sec = int(float(item["sec"]))
            zone = int(float(item["zone"]))
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        if not 0 <= zone < 9:
            skipped += 1
            continue
        severity = str(item.get("severity", "Low")).title()
        if severity not in SEVERITIES:
            severity = "Low"
        evidence_frames = item.get("evidence_frames", [])
        if isinstance(evidence_frames, str):
            evidence_frames = [evidence_frames]
        if not isinstance(evidence_frames, list):
            evidence_frames = []
        metrics = item.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        source = "vision_ai" if item.get("explanation_source") == "vision_ai" else "template"
        alerts.append({
            "id": str(item.get("id", f"alert-{index}")),
            "type": str(item.get("type", "Alert")),
            "severity": severity,
            "sec": sec,
            "zone": zone,
            "what": str(item.get("what", "No description provided.")),
            "why_flagged": str(item.get("why_flagged", "No additional explanation provided.")),
            "evidence_frames": [str(path) for path in evidence_frames if path],
            "recommended_action": str(item.get("recommended_action", "Review with an on-site operator.")),
            "metrics": metrics,
            "explanation_source": source,
        })
    return alerts, skipped


@st.cache_data(show_spinner=False)
def load_real_alerts(alerts_path: str) -> tuple[list[dict], str | None, int]:
    path = Path(alerts_path)
    if not path.is_file():
        return [], None, 0
    try:
        with path.open(encoding="utf-8") as alerts_file:
            raw_alerts = json.load(alerts_file)
        if not isinstance(raw_alerts, list):
            return [], "output/alerts.json must contain a JSON list.", 0
        alerts, skipped = normalize_alerts(raw_alerts)
        return alerts, None, skipped
    except (OSError, json.JSONDecodeError) as error:
        return [], f"Could not read output/alerts.json ({error}).", 0


@st.cache_data(show_spinner=False)
def load_real_forecast(forecast_path: str) -> tuple[dict | None, str | None]:
    path = Path(forecast_path)
    if not path.is_file():
        return None, None
    try:
        with path.open(encoding="utf-8") as forecast_file:
            payload = json.load(forecast_file)
        if not isinstance(payload, dict):
            return None, "output/forecast.json must contain a JSON object."
        text = payload.get("text", payload.get("forecast_text", payload.get("risk_forecast_text", payload.get("risk_forecast", payload.get("forecast")))))
        seconds = payload.get("seconds_to_high", payload.get("seconds_to_HIGH"))
        try:
            seconds = None if seconds is None else max(0, int(seconds))
        except (TypeError, ValueError):
            seconds = None
        return {"text": str(text) if text else None, "seconds_to_high": seconds}, None
    except (OSError, json.JSONDecodeError) as error:
        return None, f"Could not read output/forecast.json ({error})."


@st.cache_data(show_spinner=False, max_entries=2)
def load_video_bytes(video_path: str, modified_ns: int, file_size: int) -> bytes:
    return Path(video_path).read_bytes()


def resolve_asset(path_value: str) -> Path | None:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path if path.is_file() else None


def placeholder_frame() -> np.ndarray:
    """Create a small synthetic camera-view image without external image assets."""
    image = np.zeros((300, 520, 3), dtype=np.uint8)
    image[:] = (20, 37, 40)
    image[18:282, 18:502] = (28, 48, 51)
    image[26:274:3, 24:496] = (32, 54, 57)
    image[26:274, 24:496:3] = (32, 54, 57)
    image[146:150, 24:496] = (66, 195, 177)
    image[258:262, 24:496] = (66, 195, 177)
    return image


def focus_label(target: str, current: str) -> None:
    if target == current:
        st.markdown('<div class="cg-focus">GUIDE FOCUS</div>', unsafe_allow_html=True)


# Load backend configuration once per Streamlit rerun and use it to seed controls.
config, config_warning = load_config(str(ROOT / "config.json"))
CAPACITY_DEFAULTS, LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD = apply_config_defaults(config)

# Sidebar mode selection. A failed real-data load is an error, never a demo fallback.
st.sidebar.title("CrowdGuard")
real_available = CSV_PATH.is_file()
pending_mode = st.session_state.pop("pending_mode", None)
if pending_mode:
    st.session_state.mode_select = pending_mode
mode_default = 0 if real_available else 1
mode = st.sidebar.selectbox("Data mode", ["Real", "Demo data"], index=mode_default, key="mode_select")
if not real_available:
    st.markdown('<div class="cg-notice">No data yet, run the pipeline or upload a video.</div>', unsafe_allow_html=True)

uploaded_video = st.sidebar.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])
if st.sidebar.button("Run pipeline", disabled=uploaded_video is None, width="stretch"):
    try:
        from pipeline import run_pipeline
    except ImportError:
        st.sidebar.error("pipeline.py with run_pipeline(video_path, config) is not available in this workspace.")
    else:
        temp_path = None
        progress = st.sidebar.progress(0)
        try:
            suffix = Path(uploaded_video.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_video:
                temp_video.write(uploaded_video.getvalue())
                temp_path = Path(temp_video.name)
            with st.spinner("Processing video and generating CrowdGuard outputs..."):
                progress.progress(0.02, text="Saving uploaded video")
                run_pipeline(
                    temp_path,
                    config,
                    progress_callback=lambda fraction, message: progress.progress(
                        fraction, text=message
                    ),
                )
                progress.progress(1.0, text="Outputs ready")
            st.cache_data.clear()
            st.session_state.pending_mode = "Real"
            st.rerun()
        except Exception as error:
            st.sidebar.error(f"Pipeline failed: {error}")
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink()

if config_warning:
    st.sidebar.warning(config_warning)

demo_mode = mode == "Demo data"
if demo_mode:
    data = make_demo_data()
    alerts = demo_alerts(data)
    alert_error = None
    skipped_alerts = 0
    backend_forecast = None
    forecast_error = None
else:
    try:
        data = load_real_data(str(CSV_PATH))
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Real mode is unavailable: {error}. Check output/timeline.csv and its required columns.")
        st.markdown('<div class="cg-footer">Behavior-based detection only. No face recognition. Assists human operators.</div>', unsafe_allow_html=True)
        st.stop()
    alerts, alert_error, skipped_alerts = load_real_alerts(str(ROOT / "output" / "alerts.json"))
    backend_forecast, forecast_error = load_real_forecast(str(ROOT / "output" / "forecast.json"))

max_sec = int(data.sec.max())
min_sec = 0
if "time_slider" not in st.session_state:
    st.session_state.time_slider = min_sec
st.session_state.time_slider = int(np.clip(st.session_state.time_slider, min_sec, max_sec))
st.session_state.setdefault("playing", False)
st.session_state.setdefault("guide_step", 0)
st.session_state.setdefault("pending_time", None)

guided = st.sidebar.toggle("Guided walkthrough", value=False, key="guided_toggle")
if guided and not st.session_state.get("guide_was_on", False):
    st.session_state.pending_time = GUIDE[st.session_state.guide_step][1]
st.session_state.guide_was_on = guided

with st.sidebar.expander("Zone capacities", expanded=False):
    capacity_by_zone = {}
    for zone, name in enumerate(ZONE_NAMES):
        capacity_by_zone[zone] = st.number_input(
            f"{zone + 1}. {name}", min_value=10, max_value=5000, value=CAPACITY_DEFAULTS[zone],
            step=10, key=f"capacity_{zone}",
        )

if demo_mode:
    st.markdown('<span class="cg-badge">DEMO DATA</span>', unsafe_allow_html=True)
if alert_error:
    st.sidebar.warning(alert_error)
if skipped_alerts:
    st.sidebar.warning(f"Skipped {skipped_alerts} malformed alert record(s) in output/alerts.json.")
if forecast_error:
    st.sidebar.warning(forecast_error)

@st.fragment(run_every=0.45 if st.session_state.playing else None)
def render_dashboard() -> None:
    if st.session_state.pending_time is not None:
        st.session_state.time_slider = int(np.clip(st.session_state.pending_time, min_sec, max_sec))
        st.session_state.pending_time = None
    elif st.session_state.playing:
        if st.session_state.time_slider < max_sec:
            st.session_state.time_slider += 1
        else:
            st.session_state.playing = False
            st.rerun(scope="app")

    st.sidebar.markdown("### Playback")
    if max_sec == min_sec:
        selected_sec = st.sidebar.select_slider(
            "Selected second", options=[min_sec], value=min_sec, key="time_slider_single"
        )
    else:
        selected_sec = st.sidebar.slider("Selected second", min_value=min_sec, max_value=max_sec, key="time_slider")
    play_label = "Pause" if st.session_state.playing else "Play"
    if st.sidebar.button(play_label, key="playback_toggle", disabled=max_sec == min_sec, width="stretch"):
        st.session_state.playing = not st.session_state.playing
        st.rerun(scope="app")

    st.title("CrowdGuard: detect the warning signs before the stampede")
    st.markdown('<div class="cg-subtitle">NEMESIS-X (A03): Open-World Video Intelligence</div>', unsafe_allow_html=True)

    if demo_mode:
        st.markdown('<span class="cg-badge">DEMO DATA</span>', unsafe_allow_html=True)
    if guided:
        st.session_state.guide_step = int(np.clip(st.session_state.guide_step, 0, len(GUIDE) - 1))
        guide_text, guide_time, guide_target = GUIDE[st.session_state.guide_step]
        guide_col, back_col, next_col = st.columns([7, 1, 1])
        with guide_col:
            st.markdown(
                f'<div class="cg-guide"><div class="cg-guide-step">Step {st.session_state.guide_step + 1} of {len(GUIDE)}</div>'
                f'<div class="cg-guide-text">{html.escape(guide_text)}</div></div>',
                unsafe_allow_html=True,
            )
        with back_col:
            st.write("")
            if st.button("Back", key="guide_back", disabled=st.session_state.guide_step == 0, width="stretch"):
                st.session_state.guide_step -= 1
                st.session_state.pending_time = GUIDE[st.session_state.guide_step][1]
                st.rerun(scope="app")
        with next_col:
            st.write("")
            next_label = "Restart" if st.session_state.guide_step == len(GUIDE) - 1 else "Next"
            if st.button(next_label, key="guide_next", width="stretch"):
                st.session_state.guide_step = 0 if st.session_state.guide_step == len(GUIDE) - 1 else st.session_state.guide_step + 1
                st.session_state.pending_time = GUIDE[st.session_state.guide_step][1]
                st.rerun(scope="app")

    current = data[data.sec == selected_sec].copy()
    if current.empty:
        current = data.iloc[(data.sec - selected_sec).abs().argsort()[:9]].copy()
    observed_current = current.copy()
    current = current.set_index("zone").reindex(range(9)).reset_index()
    current["zone"] = current["zone"].astype(int)
    for column in ["count", "speed", "counterflow", "chaos", "risk"]:
        current[column] = current[column].fillna(0)
    current_risk = float(observed_current.risk.max())
    people_total = int(observed_current["count"].sum())
    avg_speed = float(observed_current["speed"].mean())
    avg_counterflow = float(np.mean([percent_counterflow(value) for value in observed_current.counterflow]))
    if current_risk < LOW_RISK_THRESHOLD:
        risk_color, risk_word = "#65d6a5", "LOW"
    elif current_risk <= HIGH_RISK_THRESHOLD:
        risk_color, risk_word = "#f0c65b", "ELEVATED"
    else:
        risk_color, risk_word = "#ff7777", "HIGH"

    risk_card, people_card, speed_card, flow_card = st.columns(4)
    with risk_card:
        st.markdown(f'<div class="cg-card"><div class="cg-label">Overall Risk · {risk_word}</div><div class="cg-value" style="color:{risk_color}">{current_risk:.0f}<span style="font-size:1rem"> / 100</span></div><div class="cg-detail">Peak zone risk at {fmt_time(selected_sec)}</div></div>', unsafe_allow_html=True)
    with people_card:
        st.markdown(f'<div class="cg-card"><div class="cg-label">People Count</div><div class="cg-value">{people_total:,}</div><div class="cg-detail">Across all 9 monitored zones</div></div>', unsafe_allow_html=True)
    with speed_card:
        st.markdown(f'<div class="cg-card"><div class="cg-label">Average Speed</div><div class="cg-value">{avg_speed:.2f}<span style="font-size:1rem"> px/s</span></div><div class="cg-detail">Mean zone movement speed</div></div>', unsafe_allow_html=True)
    with flow_card:
        st.markdown(f'<div class="cg-card"><div class="cg-label">Counter-flow</div><div class="cg-value">{avg_counterflow:.0f}<span style="font-size:1rem">%</span></div><div class="cg-detail">Average opposing movement</div></div>', unsafe_allow_html=True)

    risk_by_second = data.groupby("sec", as_index=False).risk.max().sort_values("sec")
    peak_row = risk_by_second.loc[risk_by_second.risk.idxmax()]
    peak_sec = int(peak_row.sec)
    peak_zone_row = data[(data.sec == peak_sec) & (data.risk == peak_row.risk)].iloc[0]
    peak_zone = int(peak_zone_row.zone)
    severity_counts = {severity: sum(alert["severity"] == severity for alert in alerts) for severity in SEVERITIES}

    recent = risk_by_second[risk_by_second.sec.between(max(min_sec, selected_sec - 29), selected_sec)]
    forecast_text = "Stable"
    forecast_is_available = bool(backend_forecast and (backend_forecast.get("text") or backend_forecast.get("seconds_to_high") is not None))
    if forecast_is_available:
        forecast_text = backend_forecast.get("text") or ""
        seconds_to_high = backend_forecast.get("seconds_to_high")
        if seconds_to_high is not None and str(seconds_to_high) not in forecast_text:
            forecast_text = f"{forecast_text} · ~{seconds_to_high} seconds to HIGH" if forecast_text else f"May reach HIGH in ~{seconds_to_high} seconds"
    elif len(recent) >= 30 and recent.sec.max() - recent.sec.min() >= 29:
        slope, intercept = np.polyfit(recent.sec.to_numpy(), recent.risk.to_numpy(), 1)
        predicted_now = slope * selected_sec + intercept
        if current_risk >= HIGH_RISK_THRESHOLD:
            forecast_text = "HIGH now"
        elif slope > 0.05:
            seconds_to_high = max(0, int(np.ceil((HIGH_RISK_THRESHOLD - predicted_now) / slope)))
            forecast_text = f"May reach HIGH in ~{seconds_to_high} seconds"

    st.markdown("### Situation summary")
    summary_cols = st.columns(4)
    summary_values = [
        ("High alerts", severity_counts["High"], "#ff7777"),
        ("Medium alerts", severity_counts["Medium"], "#f0a35b"),
        ("Low alerts", severity_counts["Low"], "#f0c65b"),
        ("Peak risk", f"Zone {peak_zone + 1} · {fmt_time(peak_sec)}", "#63d2c3"),
    ]
    for column, (label, value, color) in zip(summary_cols, summary_values):
        with column:
            st.markdown(f'<div class="cg-card" style="min-height:84px"><div class="cg-label">{label}</div><div class="cg-value" style="font-size:1.35rem;color:{color}">{value}</div></div>', unsafe_allow_html=True)
    st.caption(f"Risk forecast from the last 30 seconds: **{forecast_text}**")

    feed_col, zone_col, alert_col = st.columns([1.05, 1.0, 1.1], gap="medium")
    with feed_col:
        focus_label("feed", guide_target if guided else "")
        st.subheader("Camera feed")
        selected_video = next(
            (path for path in (ANNOTATED_VIDEO_PATH, ANNOTATED_WEBM_PATH) if path.is_file()),
            None,
        )
        frame_path = nearest_frame(selected_sec)
        if selected_video is not None:
            video_stat = selected_video.stat()
            video_bytes = load_video_bytes(str(selected_video), video_stat.st_mtime_ns, video_stat.st_size)
            video_format = "video/webm" if selected_video.suffix.lower() == ".webm" else "video/mp4"
            st.video(video_bytes, format=video_format)
        elif frame_path is not None:
            st.image(str(frame_path), caption=f"Keyframe at {fmt_time(selected_sec)}", width="stretch")
        elif VIDEO_PATH.is_file():
            video_stat = VIDEO_PATH.stat()
            video_bytes = load_video_bytes(str(VIDEO_PATH), video_stat.st_mtime_ns, video_stat.st_size)
            st.video(video_bytes, format="video/mp4")
        else:
            if demo_mode:
                st.image(placeholder_frame(), caption=f"Simulated camera view · {fmt_time(selected_sec)}", width="stretch")
            else:
                st.info(f"No video or keyframe available for {fmt_time(selected_sec)}.")

    with zone_col:
        focus_label("zones", guide_target if guided else "")
        st.subheader("Zone risk map")
        heat_values = current.set_index("zone").reindex(range(9)).risk.to_numpy().reshape(3, 3)
        zone_labels = np.array([f"Zone {zone + 1}<br>{ZONE_NAMES[zone]}" for zone in range(9)]).reshape(3, 3)
        heat = go.Figure(go.Heatmap(
            z=heat_values,
            x=[0, 1, 2],
            y=[0, 1, 2],
            text=zone_labels,
            customdata=heat_values,
            texttemplate="%{text}<br>%{z:.0f}",
            textfont={"size": 11, "color": "#f5fbf9"},
            hovertemplate="%{text}<br>Risk: %{z:.1f}<extra></extra>",
            colorscale=[
                [0, "#1c9b73"],
                [LOW_RISK_THRESHOLD / 100, "#1c9b73"],
                [LOW_RISK_THRESHOLD / 100, "#d8b63e"],
                [HIGH_RISK_THRESHOLD / 100, "#d8b63e"],
                [HIGH_RISK_THRESHOLD / 100, "#c4484d"],
                [1, "#c4484d"],
            ],
            zmin=0,
            zmax=100,
            showscale=True,
            colorbar={"title": {"text": "Risk", "font": {"color": "#cbd8d6"}}, "tickfont": {"color": "#cbd8d6"}},
            xgap=5,
            ygap=5,
        ))
        selected_zone = int(current.loc[current.risk.idxmax(), "zone"])
        selected_row, selected_column = divmod(selected_zone, 3)
        heat.add_shape(
            type="rect",
            x0=selected_column - 0.5,
            x1=selected_column + 0.5,
            y0=selected_row - 0.5,
            y1=selected_row + 0.5,
            line={"color": "#e7f0ef", "width": 2},
            fillcolor="rgba(0,0,0,0)",
            layer="above",
        )
        heat.update_layout(
            height=315,
            margin={"l": 3, "r": 3, "t": 4, "b": 3},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter, Segoe UI, sans-serif", "color": "#e7f0ef", "size": 13},
            hoverlabel={"bgcolor": "#18272a", "bordercolor": "#397f78", "font": {"family": "Inter, Segoe UI, sans-serif", "color": "#e7f0ef", "size": 14}},
            xaxis={"visible": False, "range": [-0.5, 2.5]},
            yaxis={"visible": False, "range": [2.5, -0.5]},
        )
        st.plotly_chart(heat, width="stretch", config={"displayModeBar": False})

    with alert_col:
        focus_label("alerts", guide_target if guided else "")
        st.subheader("Alert feed")
        filter_a, filter_b = st.columns(2)
        available_alert_types = sorted(set(ALERT_TYPES + [alert["type"] for alert in alerts]))
        with filter_a:
            type_filter = st.multiselect("Type", available_alert_types, default=available_alert_types, key="alert_type_filter")
        with filter_b:
            severity_filter = st.multiselect("Severity", SEVERITIES, default=SEVERITIES, key="alert_severity_filter")
        visible_alerts = [item for item in alerts if item["type"] in type_filter and item["severity"] in severity_filter]
        visible_alerts.sort(key=lambda item: item["sec"], reverse=True)
        if visible_alerts:
            for alert_index, alert in enumerate(visible_alerts[:5]):
                severity_class = alert["severity"].lower()
                is_selected = st.session_state.get("selected_alert_id") == alert["id"]
                source_label = "Vision AI" if alert["explanation_source"] == "vision_ai" else "Template"
                selected_alert_class = " cg-alert-selected" if is_selected else ""
                with st.container(border=True):
                    st.markdown(
                        f'<div class="cg-alert cg-alert-{severity_class}{selected_alert_class}"><div class="cg-alert-title"><span class="cg-chip cg-severity-{severity_class}">{html.escape(alert["severity"])}</span> · {html.escape(alert["type"])} '
                        f'<span class="cg-source">{source_label}</span></div></div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**When/Where:** <span class=\"cg-chip cg-time\">{fmt_time(alert['sec'])}</span> · <span class=\"cg-chip cg-zone\">Zone {alert['zone'] + 1}</span> ({ZONE_NAMES[alert['zone']]})", unsafe_allow_html=True)
                    st.markdown(f"**What:** {alert['what']}")
                    st.markdown(f"**Why flagged:** {alert['why_flagged']}")
                    st.markdown(f"**Evidence:** {len(alert['evidence_frames'])} frame(s)")
                    st.markdown(f"**Action:** {alert['recommended_action']}")
                    if alert["evidence_frames"]:
                        first_frame = resolve_asset(alert["evidence_frames"][0])
                        if first_frame:
                            st.image(str(first_frame), caption="Evidence frame 1", width=260)
                        else:
                            st.caption(f"Evidence frame not found: {alert['evidence_frames'][0]}")
                        if len(alert["evidence_frames"]) > 1:
                            with st.expander(f"Additional evidence frames ({len(alert['evidence_frames']) - 1})", expanded=is_selected):
                                for frame_index, frame_path in enumerate(alert["evidence_frames"][1:], start=2):
                                    resolved_frame = resolve_asset(frame_path)
                                    if resolved_frame:
                                        st.image(str(resolved_frame), caption=f"Evidence frame {frame_index}", width="stretch")
                                    else:
                                        st.caption(f"Evidence frame not found: {frame_path}")
                    if st.button("Jump to alert", key=f"jump_alert_{alert['id']}_{alert_index}", width="stretch"):
                        st.session_state.pending_time = alert["sec"]
                        st.session_state.selected_alert_id = alert["id"]
                        st.rerun(scope="app")
                    with st.expander("Metrics", expanded=is_selected):
                        if alert["metrics"]:
                            st.json(alert["metrics"])
                        else:
                            st.caption("No metrics supplied for this alert.")
        elif not demo_mode and not alerts:
            st.info("No alerts detected in this video")
        else:
            st.info("No alerts match these filters.")

    if current_risk > HIGH_RISK_THRESHOLD:
        riskiest = current.loc[current.risk.idxmax()]
        zone = int(riskiest.zone)
        previous_sec = max(min_sec, selected_sec - 20)
        prior_rows = data[(data.sec == previous_sec) & (data.zone == zone)]
        prior = prior_rows.iloc[0] if not prior_rows.empty else riskiest
        density_multiple = float(riskiest["count"] / max(1, prior["count"]))
        speed_drop_pct = max(0, float((prior["speed"] - riskiest["speed"]) / max(prior["speed"], 0.1) * 100))
        counter_pct = percent_counterflow(float(riskiest["counterflow"]))
        action = "Open Gate 4 and pause entry at Gate 2" if zone in (4, 7) else "Hold entry briefly and open the nearest alternate exit"
        explanation = explain_with_ai(None, {
            "sec": selected_sec,
            "zone_name": ZONE_NAMES[zone],
            "risk": float(riskiest["risk"]),
            "count": float(riskiest["count"]),
            "speed": float(riskiest["speed"]),
            "counterflow": counter_pct,
            "action": action,
        })
        st.markdown(
            f'<div class="cg-alert cg-alert-high"><div class="cg-alert-title">HIGH RISK · {fmt_time(selected_sec)} · Zone {zone + 1} ({html.escape(ZONE_NAMES[zone])})</div>'
            f'<div class="cg-alert-line"><b>What changed:</b> density up {density_multiple:.1f}x in 20 s, speed down {speed_drop_pct:.0f}%, {counter_pct:.0f}% moving against the flow.</div>'
            f'<div class="cg-alert-line"><b>Evidence:</b> {html.escape(explanation)}</div>'
            f'<div class="cg-alert-line"><b>Recommended action:</b> {html.escape(action)}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Risk over time · event timeline")
    focus_label("chart", guide_target if guided else "")
    chart = go.Figure()
    chart.add_hrect(y0=0, y1=LOW_RISK_THRESHOLD, fillcolor="rgba(36, 166, 112, 0.10)", line_width=0, layer="below")
    chart.add_hrect(y0=LOW_RISK_THRESHOLD, y1=HIGH_RISK_THRESHOLD, fillcolor="rgba(224, 185, 63, 0.11)", line_width=0, layer="below")
    chart.add_hrect(y0=HIGH_RISK_THRESHOLD, y1=100, fillcolor="rgba(215, 72, 76, 0.12)", line_width=0, layer="below")
    chart.add_trace(go.Scatter(x=risk_by_second.sec, y=risk_by_second.risk, mode="lines", name="Overall risk", line={"color": "#63d2c3", "width": 2.5, "shape": "spline", "smoothing": 0.65}, hovertemplate="%{x}s · risk %{y:.1f}<extra></extra>"))
    marker_colors = {"High": "#ff7777", "Medium": "#f0a35b", "Low": "#f0c65b"}
    if alerts:
        marker_rows = []
        for alert in alerts:
            point = risk_by_second.iloc[(risk_by_second.sec - alert["sec"]).abs().argsort()[:1]].iloc[0]
            marker_rows.append((alert, point))
        chart.add_trace(go.Scatter(
            x=[item[1].sec for item in marker_rows],
            y=[item[1].risk for item in marker_rows],
            mode="markers",
            name="Alerts",
            marker={"size": 10, "color": [marker_colors[item[0]["severity"]] for item in marker_rows], "line": {"color": "#10191c", "width": 1}},
            text=[f"{item[0]['type']} · {item[0]['severity']}<br>Zone {item[0]['zone'] + 1}<br>{item[0]['what']}" for item in marker_rows],
            hovertemplate="%{text}<extra></extra>",
        ))
    chart.add_vline(x=selected_sec, line_color="#f3f6f3", line_dash="dash", line_width=1.5, annotation_text=f"{fmt_time(selected_sec)}", annotation_font_color="#f3f6f3")
    chart.update_layout(height=300, margin={"l": 10, "r": 10, "t": 12, "b": 10}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"family": "Inter, Segoe UI, sans-serif", "color": "#d7e3e1", "size": 14}, hoverlabel={"bgcolor": "#18272a", "bordercolor": "#397f78", "font": {"family": "Inter, Segoe UI, sans-serif", "color": "#e7f0ef", "size": 14}}, xaxis_title="Time (seconds)", yaxis_title="Risk", yaxis={"range": [0, 100], "gridcolor": "rgba(231,240,239,0.10)", "zerolinecolor": "rgba(231,240,239,0.12)"}, xaxis={"gridcolor": "rgba(231,240,239,0.10)", "zeroline": False}, legend={"orientation": "h", "y": 1.12, "bgcolor": "rgba(20,34,38,0.72)", "bordercolor": "rgba(231,240,239,0.08)", "borderwidth": 1}, hovermode="x unified")
    st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})

    st.markdown("### Crowd management")
    focus_label("management", guide_target if guided else "")
    least_dense = observed_current.loc[observed_current["count"].idxmin()]
    safe_zone = int(least_dense.zone)
    st.markdown(f"**Safest exit suggestion:** Zone {safe_zone + 1} · {ZONE_NAMES[safe_zone]} ({int(least_dense['count'])} people, lowest current density).")
    management_cols = st.columns(3)
    for zone in range(9):
        row = current[current.zone == zone].iloc[0]
        capacity = max(1, capacity_by_zone[zone])
        occupancy = float(row["count"] / capacity * 100)
        if occupancy >= 85:
            status, status_class = "Critical", "critical"
        elif occupancy >= 60:
            status, status_class = "Busy", "busy"
        else:
            status, status_class = "Normal", "normal"
        with management_cols[zone % 3]:
            st.markdown(f'<div class="cg-card" style="min-height:105px;margin-bottom:10px"><div style="display:flex;justify-content:space-between;gap:8px"><b>Zone {zone + 1} · {html.escape(ZONE_NAMES[zone])}</b><span class="cg-chip cg-{status_class}">{status}</span></div><div class="cg-detail">Capacity: {occupancy:.0f}% · {int(row["count"])} / {capacity} people</div><div class="cg-detail">{("Estimated gate wait: ~" + str(max(1, int(np.ceil(row["count"] / max(1.0, row["speed"] * 12))))) + " min") if zone in (4, 7) else ("Risk: " + str(int(row["risk"])) + " / 100")}</div></div>', unsafe_allow_html=True)

    st.markdown("### Top 5 riskiest moments")
    top_moments = data.nlargest(5, "risk")[["sec", "zone", "risk"]].copy()
    top_moments["time"] = top_moments.sec.map(fmt_time)
    top_moments["zone"] = top_moments.zone.map(lambda zone: f"Zone {zone + 1} · {ZONE_NAMES[int(zone)]}")
    top_moments["risk"] = top_moments.risk.round(1)
    st.dataframe(top_moments[["time", "zone", "risk"]].rename(columns={"time": "Time", "zone": "Zone", "risk": "Risk"}), width="stretch", hide_index=True)

    focus_label("footer", guide_target if guided else "")
    st.markdown('<div class="cg-footer">Behavior-based detection only. No face recognition. Assists human operators.</div>', unsafe_allow_html=True)


render_dashboard()
