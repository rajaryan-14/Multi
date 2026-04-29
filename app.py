import streamlit as st

st.set_page_config(page_title="EXPLORE", layout="wide")

from emotion_module import emotion_app
from music_module import music_app

def apply_theme():
    st.markdown(
        """
        <style>
        :root {
            --olive-900: #1f2716;
            --olive-800: #2e3a1e;
            --olive-700: #465528;
            --olive-600: #667635;
            --olive-500: #7e8f42;
            --sage-300: #c8d4a3;
            --sage-200: #dfe8bf;
            --ivory-100: #f8f4e8;
            --ivory-200: #efe8d3;
            --brass-400: #c9a84f;
            --ink: #18200f;
        }

        .stApp {
            background:
                linear-gradient(90deg, rgba(31, 39, 22, 0.05) 1px, transparent 1px),
                linear-gradient(180deg, rgba(31, 39, 22, 0.04) 1px, transparent 1px),
                var(--ivory-100);
            background-size: 42px 42px;
            color: var(--ink);
        }

        .block-container {
            max-width: 1240px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: var(--olive-900);
            letter-spacing: 0;
        }

        .explore-hero {
            position: relative;
            overflow: hidden;
            min-height: 330px;
            display: grid;
            grid-template-columns: minmax(0, 1.1fr) minmax(300px, 0.9fr);
            gap: 2rem;
            align-items: center;
            padding: clamp(1.6rem, 4.5vw, 3.5rem);
            background: var(--olive-900);
            color: var(--ivory-100);
            border: 1px solid rgba(248, 244, 232, 0.18);
            box-shadow: 0 28px 70px rgba(31, 39, 22, 0.28);
            margin-bottom: 1.35rem;
        }

        .explore-hero::before {
            content: "";
            position: absolute;
            inset: 0;
            background-image:
                repeating-linear-gradient(90deg, rgba(248, 244, 232, 0.08) 0 1px, transparent 1px 70px),
                repeating-linear-gradient(0deg, rgba(201, 168, 79, 0.08) 0 1px, transparent 1px 54px);
            pointer-events: none;
        }

        .hero-copy,
        .music-scene {
            position: relative;
            z-index: 1;
        }

        .hero-kicker {
            color: var(--sage-200);
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.16rem;
            text-transform: uppercase;
            margin-bottom: 0.85rem;
        }

        .hero-title,
        .hero-title-display {
            color: var(--ivory-100);
            font-size: clamp(4.2rem, 11vw, 9.4rem);
            line-height: 0.86;
            font-weight: 950;
            margin: 0;
            text-shadow: 0 4px 18px rgba(0, 0, 0, 0.22);
            display: block;
        }

        .hero-title-display {
            color: var(--ivory-100) !important;
        }

        .hero-line {
            color: var(--sage-200);
            font-size: clamp(1.25rem, 2.5vw, 2rem);
            margin: 1.2rem 0 0;
            max-width: 560px;
        }

        .music-scene {
            min-height: 300px;
            display: grid;
            grid-template-columns: 1fr 0.85fr;
            gap: 1.5rem;
            align-items: end;
        }

        .record {
            width: min(220px, 30vw);
            aspect-ratio: 1;
            border-radius: 50%;
            margin-left: auto;
            background:
                radial-gradient(circle at center, var(--brass-400) 0 11%, var(--olive-900) 12% 17%, transparent 18%),
                repeating-radial-gradient(circle at center, #101608 0 9px, #2a351a 10px 17px);
            border: 12px solid rgba(248, 244, 232, 0.18);
            box-shadow: 0 22px 45px rgba(0, 0, 0, 0.38);
            animation: record-spin 12s linear infinite;
        }

        .bars {
            height: 190px;
            display: flex;
            align-items: end;
            gap: 0.65rem;
            padding-bottom: 0.3rem;
        }

        .bars span {
            display: block;
            width: 1.05rem;
            border-radius: 999px 999px 0 0;
            background: linear-gradient(180deg, var(--sage-200), var(--olive-500));
            box-shadow: 0 0 24px rgba(201, 168, 79, 0.24);
            animation: bar-pulse 1.3s ease-in-out infinite;
        }

        .bars span:nth-child(1) { height: 42%; animation-delay: 0s; }
        .bars span:nth-child(2) { height: 75%; animation-delay: 0.12s; }
        .bars span:nth-child(3) { height: 58%; animation-delay: 0.24s; }
        .bars span:nth-child(4) { height: 92%; animation-delay: 0.36s; }
        .bars span:nth-child(5) { height: 64%; animation-delay: 0.48s; }
        .bars span:nth-child(6) { height: 82%; animation-delay: 0.6s; }

        @keyframes record-spin {
            to { transform: rotate(360deg); }
        }

        @keyframes bar-pulse {
            0%, 100% { transform: scaleY(0.72); opacity: 0.76; }
            50% { transform: scaleY(1); opacity: 1; }
        }

        .section-kicker {
            color: var(--olive-700);
            font-size: 0.86rem;
            font-weight: 800;
            letter-spacing: 0.12rem;
            text-transform: uppercase;
            margin: 1.2rem 0 0.25rem;
        }

        .panel {
            background: rgba(255, 252, 242, 0.82);
            border: 1px solid rgba(70, 85, 40, 0.16);
            box-shadow: 0 18px 45px rgba(70, 85, 40, 0.11);
            padding: 1.25rem;
        }

        .music-header {
            display: flex;
            align-items: end;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 1.2rem;
        }

        .music-title {
            margin: 0;
            color: var(--olive-900);
            font-size: clamp(2.2rem, 5vw, 4.8rem);
            line-height: 0.95;
            font-weight: 950;
        }

        .music-subtitle {
            margin: 0.55rem 0 0;
            color: var(--olive-700);
            font-size: 1.08rem;
        }

        .mood-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            background: var(--olive-900);
            color: var(--ivory-100);
            border: 1px solid rgba(248, 244, 232, 0.2);
            padding: 0.55rem 0.75rem;
            font-weight: 800;
            white-space: nowrap;
        }

        .control-heading {
            color: var(--olive-900);
            font-weight: 900;
            margin: 0.4rem 0 0.55rem;
        }

        div.stButton > button,
        div.stDownloadButton > button {
            background: var(--olive-700);
            border: 1px solid var(--olive-700);
            color: var(--ivory-100);
            font-weight: 800;
            border-radius: 8px;
            min-height: 2.8rem;
            transition: transform 120ms ease, box-shadow 120ms ease, background 120ms ease;
        }

        div.stButton > button:hover,
        div.stDownloadButton > button:hover {
            background: var(--olive-600);
            border-color: var(--olive-600);
            color: var(--ivory-100);
            box-shadow: 0 10px 24px rgba(70, 85, 40, 0.2);
            transform: translateY(-1px);
        }

        div[data-testid="stRadio"] label,
        div[data-testid="stSelectbox"] label,
        div[data-testid="stSlider"] label,
        div[data-testid="stTextArea"] label,
        div[data-testid="stFileUploader"] label {
            color: var(--olive-900);
            font-weight: 800;
        }

        div[data-testid="stRadio"] [role="radiogroup"] {
            gap: 0.65rem;
        }

        div[data-testid="stRadio"] [role="radiogroup"] label {
            background: rgba(255, 252, 242, 0.9);
            border: 1px solid rgba(70, 85, 40, 0.2);
            padding: 0.7rem 0.95rem;
            border-radius: 8px;
        }

        div[data-testid="stRadio"] [role="radiogroup"] label *,
        div[data-testid="stRadio"] p,
        div[data-baseweb="popover"] * {
            color: var(--olive-900) !important;
            opacity: 1 !important;
        }

        div[data-baseweb="select"] > div {
            background: var(--ivory-100) !important;
            border: 1px solid rgba(70, 85, 40, 0.34) !important;
            box-shadow: inset 0 0 0 1px rgba(255, 252, 242, 0.7);
        }

        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div,
        div[data-baseweb="select"] input {
            color: var(--olive-900) !important;
            -webkit-text-fill-color: var(--olive-900) !important;
            opacity: 1 !important;
        }

        div[data-baseweb="select"] svg {
            fill: var(--olive-700) !important;
        }

        div[data-baseweb="popover"] ul,
        div[data-baseweb="popover"] li {
            background: var(--ivory-100) !important;
            color: var(--olive-900) !important;
        }

        div[data-baseweb="popover"] li:hover {
            background: var(--sage-200) !important;
        }

        .song-meta {
            background: rgba(255, 252, 242, 0.92);
            border: 1px solid rgba(70, 85, 40, 0.16);
            border-bottom: 0;
            padding: 0.85rem 0.95rem;
            margin-top: 1rem;
            border-radius: 8px 8px 0 0;
        }

        .song-title {
            color: var(--olive-900);
            font-weight: 900;
            font-size: 1.02rem;
            line-height: 1.25;
        }

        .song-subtitle {
            color: var(--olive-700);
            font-size: 0.9rem;
            margin-top: 0.2rem;
        }

        .showing-counter {
            background: rgba(255, 252, 242, 0.92);
            border: 1px solid rgba(70, 85, 40, 0.18);
            padding: 0.85rem 1rem;
            border-radius: 8px;
            min-height: 2.8rem;
        }

        .showing-label {
            color: var(--olive-700);
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.08rem;
            text-transform: uppercase;
        }

        .showing-value {
            color: var(--olive-900);
            font-size: 1.35rem;
            font-weight: 950;
            line-height: 1.15;
            margin-top: 0.15rem;
        }

        .showing-total {
            color: #16883a;
            font-weight: 800;
            margin-top: 0.2rem;
        }

        .live-frame-wrap {
            background: rgba(255, 252, 242, 0.92);
            border: 1px solid rgba(70, 85, 40, 0.18);
            border-radius: 8px;
            overflow: hidden;
            min-height: 260px;
            margin: 0.45rem 0 0.85rem;
        }

        .live-frame-image {
            display: block;
            width: 100%;
            height: auto;
            min-height: 260px;
            object-fit: cover;
            background: #161a14;
        }

        .live-status {
            border-radius: 8px;
            padding: 0.9rem 1rem;
            margin: 0.35rem 0 0.9rem;
            border: 1px solid rgba(70, 85, 40, 0.18);
            font-weight: 800;
        }

        .live-status--ok {
            background: rgba(209, 235, 196, 0.88);
            color: #203017;
        }

        .live-status--idle {
            background: rgba(255, 252, 242, 0.96);
            color: var(--olive-900);
        }

        .live-status--error {
            background: rgba(247, 221, 215, 0.94);
            color: #5d2018;
        }

        .mode-row {
            margin: 0.6rem 0 1.35rem;
        }

        .mode-row div.stButton > button {
            min-height: 4.5rem;
            width: 100%;
            background: rgba(255, 252, 242, 0.94);
            border: 1px solid rgba(70, 85, 40, 0.28);
            color: var(--olive-900);
            font-size: 1.05rem;
        }

        .mode-row div.stButton > button:hover {
            background: var(--sage-200);
            border-color: var(--olive-600);
            color: var(--olive-900);
        }

        div[data-testid="stMetric"] {
            background: rgba(255, 252, 242, 0.8);
            border: 1px solid rgba(70, 85, 40, 0.14);
            padding: 1rem;
        }

        iframe {
            border-radius: 8px;
        }

        @media (max-width: 900px) {
            .explore-hero {
                grid-template-columns: 1fr;
                min-height: auto;
            }

            .music-scene {
                grid-template-columns: 1fr;
                min-height: 210px;
            }

            .record {
                width: min(220px, 60vw);
                margin: 0;
            }

            .bars {
                height: 130px;
            }

            .music-header {
                align-items: start;
                flex-direction: column;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_theme()

if "page" not in st.session_state:
    st.session_state.page = "emotion"

# if emotion is detected and user clicked recommend
if st.session_state.page == "emotion":
    emotion_app.run()

elif st.session_state.page == "music":
    music_app.run()
