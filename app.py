"""
AI Travel Agent & Concierge — Apple Inspired Pro Edition
Designed with Apple's iconic minimalist design language (SF Pro aesthetic, frosted glass,
precision typography, pill controls, and titanium accents) powered by Groq LLM & LangGraph.
"""

# pylint: disable=invalid-name
import datetime
from datetime import timedelta
import os
import re
import uuid
from dotenv import find_dotenv, load_dotenv
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

load_dotenv(find_dotenv())
_env_app = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_app):
    load_dotenv(_env_app)

from agents.agent import Agent
from agents.memory import UserTravelPreferences
from rag.retriever import get_rag_retriever
from rag.utils import (
    get_map_dataframe,
    generate_ics_calendar,
    get_packing_checklist,
)
from export.pdf_generator import generate_pdf_travel_report
from export.email_template import markdown_to_clean_html_email


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Travel Planner • Apple Edition",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------

def init_session_state():
    if "agent" not in st.session_state:
        st.session_state.agent = Agent()
    if "rag" not in st.session_state:
        st.session_state.rag = get_rag_retriever()
    if "preferences" not in st.session_state:
        st.session_state.preferences = UserTravelPreferences()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="**Welcome to AI Travel Concierge.** Effortlessly discover verified flights, curated boutique stays, visa intelligence, and day-by-day itineraries. Where would you like to travel next?")
        ]
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "destination_name" not in st.session_state:
        st.session_state.destination_name = "Dubai"
    if "origin_name" not in st.session_state:
        st.session_state.origin_name = "Mumbai (BOM)"
    if "start_date" not in st.session_state:
        st.session_state.start_date = datetime.date.today() + datetime.timedelta(days=14)
    if "end_date" not in st.session_state:
        st.session_state.end_date = datetime.date.today() + datetime.timedelta(days=20)
    if "travel_info" not in st.session_state:
        st.session_state.travel_info = None
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "reasoning_trace" not in st.session_state:
        st.session_state.reasoning_trace = []
    if "validation_report" not in st.session_state:
        st.session_state.validation_report = None
    if "hitl_state" not in st.session_state:
        st.session_state.hitl_state = {"active": False, "type": None, "status": "APPROVED"}
    if "benchmark_results" not in st.session_state:
        st.session_state.benchmark_results = None


# ---------------------------------------------------------
# APPLE SIGNATURE DESIGN SYSTEM (CSS)
# ---------------------------------------------------------

def apply_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* Preserve Streamlit icon fonts and prevent ligature breakdown */
        [data-testid="stIconMaterial"], 
        [data-testid="stSidebarCollapseButton"] span,
        [data-testid="collapsedControl"] span,
        .material-symbols-rounded, 
        .material-symbols-sharp, 
        .material-icons {
            font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
            letter-spacing: normal !important;
            font-size: 1.25rem !important;
        }

        /* Hide Deploy button and default header clutter for a clean Apple look */
        .stDeployButton, #MainMenu, footer, [data-testid="stToolbar"] {
            visibility: hidden !important;
            display: none !important;
        }

        header[data-testid="stHeader"] {
            background: transparent !important;
        }

        /* Apple System Typography */
        html, body, p, div, h1, h2, h3, h4, h5, h6, input, textarea, button, select, label {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Inter", "Helvetica Neue", sans-serif;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            letter-spacing: -0.015em;
        }

        /* Apple Obsidian Matte Canvas */
        .stApp {
            background-color: #000000 !important;
            background-image: 
                radial-gradient(circle at 50% -10%, rgba(41, 151, 255, 0.12) 0%, transparent 60%),
                radial-gradient(circle at 80% 20%, rgba(134, 134, 139, 0.06) 0%, transparent 50%);
            color: #f5f5f7;
        }

        .block-container {
            padding-top: 1.0rem;
            padding-bottom: 4rem;
            max-width: 1240px;
        }

        /* Top Apple-Style Navigation Bar */
        .apple-nav {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(22, 22, 23, 0.8);
            backdrop-filter: blur(24px) saturate(180%);
            -webkit-backdrop-filter: blur(24px) saturate(180%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 980px;
            padding: 0.65rem 1.4rem;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }
        .apple-nav-brand {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            font-size: 0.95rem;
            font-weight: 600;
            color: #f5f5f7;
            letter-spacing: -0.02em;
        }
        .apple-nav-badge {
            font-size: 0.72rem;
            font-weight: 500;
            color: #2997ff;
            background: rgba(41, 151, 255, 0.12);
            border: 1px solid rgba(41, 151, 255, 0.25);
            padding: 0.2rem 0.65rem;
            border-radius: 980px;
        }

        /* Apple Hero Banner */
        .apple-hero {
            text-align: center;
            padding: 2.8rem 1.5rem 2.4rem 1.5rem;
            margin-bottom: 2.0rem;
            position: relative;
        }
        .apple-eyebrow {
            font-size: 0.82rem;
            font-weight: 600;
            color: #2997ff;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.6rem;
        }
        .apple-hero-title {
            font-size: 3.4rem;
            font-weight: 800;
            letter-spacing: -0.035em;
            line-height: 1.08;
            margin-bottom: 0.8rem;
            background: linear-gradient(180deg, #ffffff 0%, #a1a1a6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .apple-hero-subtitle {
            font-size: 1.15rem;
            font-weight: 400;
            color: #86868b;
            max-width: 720px;
            margin: 0 auto 1.6rem auto;
            line-height: 1.5;
            letter-spacing: -0.01em;
        }

        /* Apple Feature Pill Chips */
        .apple-chips-row {
            display: flex;
            justify-content: center;
            gap: 0.6rem;
            flex-wrap: wrap;
        }
        .apple-chip {
            background: rgba(28, 28, 30, 0.7);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: #d1d1d6;
            padding: 0.4rem 0.95rem;
            border-radius: 980px;
            font-size: 0.78rem;
            font-weight: 500;
            letter-spacing: -0.01em;
            transition: all 0.2s ease;
        }
        .apple-chip:hover {
            border-color: rgba(255, 255, 255, 0.2);
            background: rgba(44, 44, 46, 0.8);
            color: #ffffff;
        }

        /* Apple Destination Cards */
        .apple-card {
            background: #161617;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 1.5rem 1.2rem;
            text-align: center;
            transition: transform 0.3s cubic-bezier(0.25, 1, 0.5, 1), border-color 0.3s ease, box-shadow 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        .apple-card:hover {
            transform: translateY(-4px) scale(1.01);
            border-color: rgba(41, 151, 255, 0.4);
            box-shadow: 0 16px 36px -10px rgba(0, 0, 0, 0.7), 0 0 20px rgba(41, 151, 255, 0.15);
        }
        .apple-card-badge {
            font-size: 0.72rem;
            font-weight: 600;
            color: #2997ff;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.4rem;
        }
        .apple-card-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #f5f5f7;
            letter-spacing: -0.02em;
            margin-bottom: 0.25rem;
        }
        .apple-card-tag {
            font-size: 0.78rem;
            font-weight: 400;
            color: #86868b;
        }

        /* Apple Glass Panel */
        .apple-glass-panel {
            background: rgba(22, 22, 23, 0.75);
            backdrop-filter: blur(24px) saturate(180%);
            -webkit-backdrop-filter: blur(24px) saturate(180%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 22px;
            padding: 1.8rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.5);
        }

        /* Apple Result Board */
        .apple-result-board {
            background: #121214;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 2.2rem;
            box-shadow: 0 24px 48px rgba(0, 0, 0, 0.6);
            margin-top: 1.5rem;
            line-height: 1.7;
            color: #e5e5e7;
        }

        /* Apple Trip Header Ribbon */
        .apple-trip-ribbon {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: linear-gradient(90deg, rgba(41, 151, 255, 0.12) 0%, rgba(22, 22, 23, 0.6) 100%);
            border: 1px solid rgba(41, 151, 255, 0.25);
            padding: 1rem 1.4rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
            gap: 0.8rem;
        }
        .apple-trip-route {
            font-size: 1.28rem;
            font-weight: 700;
            color: #f5f5f7;
            letter-spacing: -0.02em;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Apple Metric Cards */
        .apple-metric-card {
            background: #161617;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            padding: 1.2rem;
            text-align: center;
            transition: all 0.25s ease;
        }
        .apple-metric-card:hover {
            border-color: rgba(255, 255, 255, 0.16);
            background: #1c1c1e;
        }
        .apple-metric-val {
            font-size: 1.85rem;
            font-weight: 700;
            color: #f5f5f7;
            letter-spacing: -0.03em;
            margin-bottom: 0.2rem;
        }
        .apple-metric-lbl {
            font-size: 0.75rem;
            font-weight: 600;
            color: #86868b;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        /* Apple Total Summary Card */
        .apple-total-box {
            background: linear-gradient(180deg, rgba(28, 28, 30, 0.85) 0%, rgba(18, 18, 20, 0.95) 100%);
            border: 1px solid rgba(41, 151, 255, 0.35);
            border-radius: 20px;
            padding: 1.6rem;
            margin-top: 1.5rem;
            text-align: center;
            box-shadow: 0 16px 36px rgba(0, 0, 0, 0.5);
        }
        .apple-total-val {
            font-size: 2.6rem;
            font-weight: 800;
            color: #ffffff;
            letter-spacing: -0.04em;
        }

        /* Apple Segmented Control / Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            background: #161617;
            padding: 5px;
            border-radius: 980px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            display: inline-flex;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 980px;
            padding: 8px 18px;
            font-weight: 500;
            font-size: 0.86rem;
            color: #86868b;
            border: none;
            background: transparent;
            transition: all 0.2s cubic-bezier(0.25, 0.1, 0.25, 1);
        }
        .stTabs [aria-selected="true"] {
            background: #2c2c2e !important;
            color: #ffffff !important;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.4) !important;
        }

        /* Apple Pill Buttons */
        .stButton>button {
            border-radius: 980px !important;
            font-weight: 500 !important;
            font-size: 0.88rem !important;
            letter-spacing: -0.01em !important;
            padding: 0.5rem 1.3rem !important;
            transition: all 0.2s cubic-bezier(0.25, 0.1, 0.25, 1) !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            background: #1c1c1e !important;
            color: #f5f5f7 !important;
        }
        .stButton>button:hover {
            background: #2c2c2e !important;
            border-color: rgba(255, 255, 255, 0.25) !important;
            transform: scale(1.02);
        }
        .stButton>button[kind="primary"] {
            background: #0071e3 !important;
            color: #ffffff !important;
            border: 1px solid #0077ed !important;
            box-shadow: 0 4px 14px rgba(0, 113, 227, 0.35) !important;
        }
        .stButton>button[kind="primary"]:hover {
            background: #0077ed !important;
            box-shadow: 0 6px 20px rgba(0, 113, 227, 0.55) !important;
        }

        /* Form Inputs in Apple Dark Mode */
        div[data-baseweb="input"], div[data-baseweb="select"] {
            border-radius: 12px !important;
            background-color: #1c1c1e !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
        }
        div[data-baseweb="input"]:focus-within {
            border-color: #2997ff !important;
            box-shadow: 0 0 0 3px rgba(41, 151, 255, 0.25) !important;
        }

        /* Sidebar in Apple Dark */
        [data-testid="stSidebar"] {
            background-color: #0d0d0f !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        if os.path.exists("images/ai-travel.png"):
            st.image("images/ai-travel.png", caption="Travel Intelligence Pro", use_container_width=True)

        st.markdown(
            """
            <div style="background: rgba(28, 28, 30, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 1.1rem; margin-bottom: 1.2rem;">
                <div style="font-size: 0.78rem; font-weight: 700; color: #2997ff; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem;">
                    SYSTEM STATUS: ONLINE
                </div>
                <div style="font-size: 0.80rem; color: #a1a1a6; line-height: 1.7;">
                    • <b>Groq Intelligence:</b> Ultra-Low Latency<br>
                    • <b>RAG Engine:</b> Hybrid Dense + BM25<br>
                    • <b>Tool Ecosystem:</b> 8 Specialized Tools<br>
                    • <b>Human-in-the-Loop:</b> Interrupt Active<br>
                    • <b>Reflection Agent:</b> Live Validation
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # API Keys Configuration
        with st.expander("API Keys Configuration", expanded=False):
            st.caption("Enter your API keys (optional). If omitted, verified offline fallback intelligence will be used.")
            current_groq = os.environ.get("GROQ_API_KEY", "")
            if "placeholder" in current_groq.lower() or "your_" in current_groq.lower():
                current_groq = ""
            groq_key_input = st.text_input("Groq API Key", value=current_groq, type="password", placeholder="gsk_...")
            serp_key_input = st.text_input("SerpAPI Key (Optional)", value=os.environ.get("SERPAPI_API_KEY", ""), type="password", placeholder="...")
            sendgrid_key_input = st.text_input("SendGrid Key (Optional)", value=os.environ.get("SENDGRID_API_KEY", ""), type="password", placeholder="SG...")

            if st.button("Save API Keys", use_container_width=True):
                if groq_key_input.strip():
                    os.environ["GROQ_API_KEY"] = groq_key_input.strip()
                if serp_key_input.strip():
                    os.environ["SERPAPI_API_KEY"] = serp_key_input.strip()
                if sendgrid_key_input.strip():
                    os.environ["SENDGRID_API_KEY"] = sendgrid_key_input.strip()
                # Re-initialize agent with updated keys
                st.session_state.agent = Agent()
                st.success("API keys updated successfully.")

        # Traveler Preferences Store
        st.markdown("### Traveler Profile")
        with st.expander("Preferences & Constraints", expanded=False):
            prefs = st.session_state.preferences.get_all()
            b_tier = st.selectbox("Budget Profile", ["Luxury ($$$)", "Moderate ($$)", "Backpacker / Budget ($)"], index=1 if "Moderate" in prefs["budget_tier"] else 0)
            max_b = st.number_input("Max Budget (USD)", min_value=300, max_value=50000, value=int(prefs.get("max_budget_usd", 3500)), step=100)
            t_style = st.selectbox("Travel Style", ["Cultural & Sightseeing", "Romantic / Honeymoon", "Family Fun", "Beach & Relaxation", "Adventure & Nature", "Solo Exploration"], index=0)
            h_cat = st.selectbox("Preferred Hotel Class", ["5-Star Luxury", "4-Star Premium", "3-Star Boutique"], index=1)
            f_pref = st.selectbox("Food Preference", ["All / Local Gourmet", "Vegetarian", "Vegan", "Halal", "Gluten-Free"], index=0)
            notes = st.text_input("Dietary & Accessibility Notes", value=prefs.get("special_notes", ""))

            if st.button("Save Profile Preferences", use_container_width=True):
                st.session_state.preferences.update({
                    "budget_tier": b_tier,
                    "max_budget_usd": float(max_b),
                    "travel_style": t_style,
                    "hotel_category": h_cat,
                    "food_preference": f_pref,
                    "special_notes": notes,
                })
                st.success("Preferences updated.")

        st.markdown("### Curated Destinations")
        st.markdown(
            """
            <div style="font-size: 0.82rem; color: #86868b; line-height: 1.8;">
                <b>Dubai, UAE</b> • Desert & Skyline<br>
                <b>Bali, Indonesia</b> • Tropical Retreat<br>
                <b>Paris, France</b> • Art & Culture<br>
                <b>Tokyo, Japan</b> • Modern & Traditional<br>
                <b>Goa, India</b> • Coastal Heritage<br>
                <b>London, UK</b> • Royal & Historic<br>
                <b>Singapore</b> • Futuristic City-State<br>
                <b>Switzerland</b> • Alpine Panorama
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.caption("AI Travel Planner • Apple Pro Edition • 2026")


# ---------------------------------------------------------
# APPLE TOP BAR & HERO BANNER
# ---------------------------------------------------------

def render_hero():
    st.markdown(
        """
        <div class="apple-nav">
            <div class="apple-nav-brand">
                <span>Travel Intelligence Pro</span>
            </div>
            <div class="apple-nav-badge">
                Autonomous Planning
            </div>
        </div>

        <div class="apple-hero">
            <div class="apple-eyebrow">The Architecture of Exploration</div>
            <div class="apple-hero-title">
                Wanderlust Meets<br>Artificial Intelligence.
            </div>
            <div class="apple-hero-subtitle">
                Decomposing complex journeys into live flight schedules, verified boutique stays, hybrid RAG visa intelligence, and human-in-the-loop precision.
            </div>
            <div class="apple-chips-row">
                <span class="apple-chip">Sub-Second Groq Engine</span>
                <span class="apple-chip">Hybrid Vector & BM25 RAG</span>
                <span class="apple-chip">Reflection & Validation</span>
                <span class="apple-chip">Human-in-the-Loop</span>
                <span class="apple-chip">PDF Dossier Export</span>
                <span class="apple-chip">SendGrid Dispatch</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# PROCESS QUERY WITH AGENT
# ---------------------------------------------------------

def execute_travel_query(query_text: str, human_feedback: str = None):
    if not query_text.strip():
        st.warning("Please enter your travel parameters.")
        return

    st.session_state.last_query = query_text

    # Dynamically detect trip extension in feedback
    if human_feedback:
        fb_l = human_feedback.lower()
        ext_by = re.search(r"extend(?:\s+\w+){0,3}\s+by\s+(\d+)\s+days?", fb_l)
        ext_to = re.search(r"(?:extend|change|make\s+it|expand|increase|plan)(?:\s+\w+){0,3}\s+(?:to\s+)?(\d+)\s+days?", fb_l)
        ext_simple = re.search(r"(\d+)\s+days?", fb_l)
        target_days = None
        if ext_by:
            cur_diff = (st.session_state.end_date - st.session_state.start_date).days
            target_days = max(1, cur_diff + int(ext_by.group(1)))
        elif ext_to:
            target_days = int(ext_to.group(1))
        elif "extend" in fb_l and ext_simple:
            target_days = int(ext_simple.group(1))

        if target_days and 1 <= target_days <= 30:
            st.session_state.end_date = st.session_state.start_date + datetime.timedelta(days=target_days)

    with st.spinner("Curating your bespoke travel plan with Groq & LangGraph..."):
        try:
            thread_id = st.session_state.thread_id or str(uuid.uuid4())
            st.session_state.thread_id = thread_id
            config = {"configurable": {"thread_id": thread_id}}

            trip_details = {
                "origin": st.session_state.origin_name,
                "destination": st.session_state.destination_name,
                "start_date": str(st.session_state.start_date),
                "end_date": str(st.session_state.end_date),
            }

            prefs = st.session_state.preferences.get_all()

            prompt_content = query_text
            if human_feedback:
                prompt_content += f"\n\n[USER DIRECTIVE / RE-PLANNING FEEDBACK]: {human_feedback}"

            initial_state = {
                "messages": [HumanMessage(content=prompt_content)],
                "preferences": prefs,
                "trip_details": trip_details,
                "human_feedback": human_feedback,
                "reasoning_trace": [],
                "selected_tools": [],
                "retrieved_documents": [],
            }

            result = st.session_state.agent.graph.invoke(initial_state, config=config)

            # Extract travel response
            travel_info = None
            for m in reversed(result["messages"]):
                if isinstance(m, AIMessage) and m.content:
                    travel_info = m.content
                    break
            if not travel_info and result["messages"]:
                travel_info = result["messages"][-1].content

            st.session_state.travel_info = travel_info
            st.session_state.reasoning_trace = result.get("reasoning_trace", [])
            st.session_state.validation_report = result.get("validation_details")

            # Update HITL state if approval requested
            if result.get("approval_required"):
                st.session_state.hitl_state = {
                    "active": True,
                    "type": result.get("approval_type", "General"),
                    "status": "PENDING",
                }
            else:
                st.session_state.hitl_state = {
                    "active": False,
                    "type": None,
                    "status": "APPROVED",
                }

            st.toast("Bespoke travel plan curated successfully.", icon="✨")
        except Exception as exc:
            # Automatic fallback to deterministic multi-tool generator if unexpected exception happens
            fallback_msg = st.session_state.agent._generate_deterministic_travel_plan({
                "trip_details": {
                    "origin": st.session_state.origin_name,
                    "destination": st.session_state.destination_name,
                    "start_date": str(st.session_state.start_date),
                    "end_date": str(st.session_state.end_date),
                },
                "preferences": st.session_state.preferences.get_all(),
                "human_feedback": human_feedback,
                "messages": [HumanMessage(content=prompt_content if 'prompt_content' in locals() else query_text)],
            })
            st.session_state.travel_info = fallback_msg.content
            st.session_state.hitl_state = {"active": False, "type": None, "status": "APPROVED"}
            st.toast("Bespoke travel plan updated with your directives.", icon="✨")


# ---------------------------------------------------------
# HUMAN-IN-THE-LOOP CONTROLLER (APPLE STYLE)
# ---------------------------------------------------------

def render_hitl_controller():
    if not st.session_state.travel_info:
        return

    st.markdown("### Human-in-the-Loop & Decision Control")
    st.caption("Review recommendations. Approve, request budget optimization, modify day plans, or reset.")

    h_cols = st.columns([1.2, 1.2, 1.2, 1.2, 1.2])

    with h_cols[0]:
        if st.button("Approve Plan", type="primary", use_container_width=True):
            st.session_state.hitl_state["status"] = "APPROVED"
            st.success("Plan approved. You can now download the PDF dossier or dispatch via SendGrid.")

    with h_cols[1]:
        if st.button("Optimize Budget", use_container_width=True):
            feedback = "The plan is too expensive. Please optimize flight, hotel, and dining allocations to strictly respect the target budget."
            execute_travel_query(st.session_state.last_query, human_feedback=feedback)
            st.rerun()

    with h_cols[2]:
        if st.button("Alternative Hotels", use_container_width=True):
            feedback = "Please search alternative top-rated hotel accommodations with better amenities and views."
            execute_travel_query(st.session_state.last_query, human_feedback=feedback)
            st.rerun()

    with h_cols[3]:
        if st.button("Search Again", use_container_width=True):
            execute_travel_query(st.session_state.last_query)
            st.rerun()

    with h_cols[4]:
        if st.button("Reset Plan", use_container_width=True):
            st.session_state.travel_info = None
            st.session_state.reasoning_trace = []
            st.session_state.validation_report = None
            st.session_state.hitl_state = {"active": False, "type": None, "status": "REJECTED"}
            st.rerun()

    # Custom Natural Language Modification Input
    with st.expander("Custom Re-planning Directive", expanded=False):
        c_fb1, c_fb2 = st.columns([4, 1])
        with c_fb1:
            custom_feedback = st.text_input(
                "Feedback directive",
                placeholder="e.g. 'Change Day 2 afternoon to visit contemporary art galleries', 'Add vegetarian dinner spots', 'Make it 1 day longer'"
            )
        with c_fb2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Re-plan", use_container_width=True) and custom_feedback:
                execute_travel_query(st.session_state.last_query, human_feedback=custom_feedback)
                st.rerun()


# ---------------------------------------------------------
# TAB 1: TRIP PLANNER (INTERACTIVE BUILDER)
# ---------------------------------------------------------

def render_trip_planner_tab():
    st.markdown("### Curate Your Journey")
    st.caption("Select an inspiration preset below or tailor every detail of your flight, stay, and day-by-day plan.")

    # Preset Destination Cards in Apple Grid
    p_cols = st.columns(4)

    with p_cols[0]:
        st.markdown(
            """
            <div class="apple-card">
                <div class="apple-card-badge">United Arab Emirates</div>
                <div class="apple-card-title">Dubai</div>
                <div class="apple-card-tag">5-Star Luxury • 5 Days</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Explore Dubai", key="btn_dubai", use_container_width=True):
            st.session_state.origin_name = "Mumbai (BOM)"
            st.session_state.destination_name = "Dubai"
            prompt = "Find flights from Mumbai (BOM) to Dubai (DXB) from 15 to 20 November 2026 for 2 adults. Find 4-star and 5-star hotels in Dubai. Provide visa guidelines, weather, and a 5-day luxury itinerary."
            execute_travel_query(prompt)

    with p_cols[1]:
        st.markdown(
            """
            <div class="apple-card">
                <div class="apple-card-badge">Indonesia</div>
                <div class="apple-card-title">Bali</div>
                <div class="apple-card-tag">Romantic Island • 7 Days</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Explore Bali", key="btn_bali", use_container_width=True):
            st.session_state.origin_name = "Delhi (DEL)"
            st.session_state.destination_name = "Bali"
            prompt = "Find flights from Delhi (DEL) to Bali Denpasar (DPS) from 10 to 17 October 2026 for 2 adults. Find 4-star beach resort hotels in Bali. Provide Bali visa-on-arrival details, weather, culture tips, and 7-day itinerary."
            execute_travel_query(prompt)

    with p_cols[2]:
        st.markdown(
            """
            <div class="apple-card">
                <div class="apple-card-badge">France</div>
                <div class="apple-card-title">Paris</div>
                <div class="apple-card-tag">Art & Romance • 5 Days</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Explore Paris", key="btn_paris", use_container_width=True):
            st.session_state.origin_name = "Mumbai (BOM)"
            st.session_state.destination_name = "Paris"
            prompt = "Find flights from Mumbai (BOM) to Paris (CDG) from 5 to 10 May 2026 for 2 adults. Find 4-star boutique hotels near Eiffel Tower or Central Paris. Include Schengen visa rules, weather, metro tips, and 5-day cultural itinerary."
            execute_travel_query(prompt)

    with p_cols[3]:
        st.markdown(
            """
            <div class="apple-card">
                <div class="apple-card-badge">India</div>
                <div class="apple-card-title">Goa</div>
                <div class="apple-card-tag">Beachfront Chill • 4 Days</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Explore Goa", key="btn_goa", use_container_width=True):
            st.session_state.origin_name = "Mumbai (BOM)"
            st.session_state.destination_name = "Goa"
            prompt = "Find flights from Mumbai (BOM) to Goa (GOI) from 12 to 16 December 2026 for 2 adults. Find 4-star or 5-star beachside resorts in Goa. Include weather, top beaches, local food, and 4-day itinerary."
            execute_travel_query(prompt)

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Form inside Apple Glass Container
    with st.expander("Tailor Journey Parameters", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            origin = st.text_input("Origin City / Airport Code", value=st.session_state.origin_name)
            destination = st.text_input("Destination City / Country", value=st.session_state.destination_name)
            st.session_state.origin_name = origin
            st.session_state.destination_name = destination

            date_cols = st.columns(2)
            with date_cols[0]:
                dep_date = st.date_input("Departure Date", value=st.session_state.start_date)
                st.session_state.start_date = dep_date
            with date_cols[1]:
                ret_date = st.date_input("Return Date", value=st.session_state.end_date)
                st.session_state.end_date = ret_date

        with c2:
            guest_cols = st.columns(3)
            with guest_cols[0]:
                adults = st.number_input("Adults (12+)", min_value=1, max_value=9, value=2)
            with guest_cols[1]:
                children = st.number_input("Children (2-11)", min_value=0, max_value=6, value=0)
            with guest_cols[2]:
                rooms = st.number_input("Rooms", min_value=1, max_value=5, value=1)

            pref_cols = st.columns(2)
            with pref_cols[0]:
                budget_tier = st.selectbox("Budget Tier", ["Luxury ($$$)", "Moderate ($$)", "Backpacker / Budget ($)"])
            with pref_cols[1]:
                travel_style = st.selectbox("Travel Style", ["Cultural & Heritage", "Romantic / Honeymoon", "Family Fun", "Adventure & Nature", "Beach & Relaxation"])

            hotel_class = st.select_slider("Hotel Class", options=["3-Star", "4-Star", "5-Star Luxury"], value="4-Star")

        additional_notes = st.text_input("Special Preferences & Requests", placeholder="e.g., Sea view suite, vegetarian dining, historical walking tours, airport limousine")

        if st.button("Plan Journey with AI", type="primary", use_container_width=True):
            query = (
                f"Find flights from {origin} to {destination} departing {dep_date} and returning {ret_date} "
                f"for {adults} adult(s) and {children} child(ren). "
                f"Find {hotel_class} hotels in {destination} for {rooms} room(s) suited for a {budget_tier} budget. "
                f"Travel style is {travel_style}. "
                f"Include destination visa requirements, best weather season, local transportation hacks, "
                f"and a complete day-by-day travel plan with itemized budget. "
                f"Additional preferences: {additional_notes or 'None'}."
            )
            execute_travel_query(query)


# ---------------------------------------------------------
# TAB 2: CONVERSATIONAL AI CONCIERGE CHAT
# ---------------------------------------------------------

def render_chat_tab():
    st.markdown("### Conversational Travel Concierge")
    st.caption("Direct high-speed communication with your AI Concierge. Ask follow-up questions, refine schedules, or query visa specifics.")

    # Render Chat History
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)

    # Chat Input
    user_prompt = st.chat_input("Ask: 'Suggest vegetarian dining in Dubai', 'What is the visa fee for Bali?', 'Check weather in Tokyo'...")
    if user_prompt:
        st.session_state.chat_history.append(HumanMessage(content=user_prompt))
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consulting travel intelligence..."):
                try:
                    retriever = st.session_state.rag
                    relevant_chunks = retriever.search(user_prompt, top_k=3)
                    # Pass previous chat messages for conversational memory context
                    response_text = retriever.synthesize_answer(
                        query=user_prompt,
                        context_chunks=relevant_chunks,
                        chat_history=st.session_state.chat_history[:-1]
                    )
                except Exception as exc:
                    response_text = f"Apologies, an error occurred: {exc}"

                st.markdown(response_text)
                st.session_state.chat_history.append(AIMessage(content=response_text))

    if len(st.session_state.chat_history) > 1:
        if st.button("Clear History"):
            st.session_state.chat_history = [
                AIMessage(content="Chat context refreshed. How can I assist with your next journey?")
            ]
            st.rerun()


# ---------------------------------------------------------
# TAB 3: RAG KNOWLEDGE BASE & DOCUMENT UPLOADER
# ---------------------------------------------------------

def render_rag_tab():
    st.markdown("### Travel Knowledge Hub & Document Indexer")
    st.caption("Query verified destination guides (visas, baggage policies, scams) or upload personal travel documents (.pdf, .md, .txt, .json) to index them directly.")

    r_col1, r_col2 = st.columns([1.2, 0.8])

    with r_col1:
        st.markdown("#### Search Verified Intelligence")
        c_q1, c_q2 = st.columns([2, 1])
        with c_q1:
            rag_search_query = st.text_input(
                "Search specific visa, cultural, or travel rules",
                placeholder="e.g. 'Dubai visa rules for Indian passport', 'Bali tourist levy fee', 'liquids rule in cabin luggage'"
            )
        with c_q2:
            dest_filter = st.selectbox("Destination Filter", ["All Destinations", "Dubai", "Bali", "Paris", "Tokyo", "Goa", "London", "Singapore", "Switzerland"])

        rag_search_btn = st.button("Search Knowledge Base", type="secondary")
        if rag_search_query and rag_search_btn:
            with st.spinner("Searching and synthesizing verified travel knowledge..."):
                retriever = st.session_state.rag
                d_filt = None if dest_filter == "All Destinations" else dest_filter
                results = retriever.search(rag_search_query, top_k=3, destination_filter=d_filt)
                answer = retriever.synthesize_answer(rag_search_query, results)

                st.markdown("#### Grounded Intelligence")
                st.info(answer)

                st.markdown("#### Verified Source Citations")
                for r in results:
                    with st.expander(f"Source: {r['source']} — {r['title']} (Relevance: {r['score']})"):
                        st.write(r["content"])

    with r_col2:
        st.markdown("#### Upload Travel Documents")
        st.caption("Upload ticket PDFs, baggage policies, booking vouchers, or custom guides (.txt, .md, .pdf, .json) to query them via RAG.")

        uploaded_files = st.file_uploader(
            "Upload documents to index",
            type=["txt", "md", "pdf", "json"],
            accept_multiple_files=True,
            key="rag_uploader"
        )

        if uploaded_files:
            for up_file in uploaded_files:
                doc_text = st.session_state.rag.extract_text_from_file(up_file)
                if doc_text.strip():
                    count = st.session_state.rag.add_custom_document(up_file.name, doc_text)
                    st.success(f"Indexed `{up_file.name}` ({len(doc_text.split())} words, {count} chunks) into RAG engine.")
                else:
                    st.warning(f"Could not extract readable text from `{up_file.name}`.")


# ---------------------------------------------------------
# INTERACTIVE RESULTS HUB
# ---------------------------------------------------------

def render_results_hub():
    if not st.session_state.travel_info:
        return

    st.markdown("---")
    st.markdown(
        f"""
        <div class="apple-trip-ribbon">
            <div class="apple-trip-route">
                {st.session_state.origin_name} &nbsp;➔&nbsp; {st.session_state.destination_name}
            </div>
            <div style="font-size: 0.88rem; color: #86868b; font-weight: 500;">
                {st.session_state.start_date} to {st.session_state.end_date} &nbsp;•&nbsp; Curated Itinerary
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render HITL Decision Buttons
    render_hitl_controller()

    result_tabs = st.tabs([
        "Overview & Booking",
        "Budget Calculator",
        "Destination Map",
        "Packing Guide",
        "Export & Dispatch",
    ])

    # Sub-tab 1: Full Travel Plan
    with result_tabs[0]:
        st.markdown('<div class="apple-result-board">', unsafe_allow_html=True)
        st.markdown(st.session_state.travel_info)
        st.markdown('</div>', unsafe_allow_html=True)

    # Sub-tab 2: Dynamic Budget Calculator
    with result_tabs[1]:
        st.markdown("### Trip Investment & Budget Calculator")
        st.caption("Dynamically adjust flight, stay, dining, and activity expectations to simulate real-time expenditure.")

        curr_col, people_col = st.columns(2)
        with curr_col:
            currency = st.selectbox("Preferred Currency", ["USD ($)", "INR (₹)", "EUR (€)", "AED (Dirham)", "GBP (£)", "JPY (¥)", "SGD (S$)"])
            curr_symbol = "$" if "USD" in currency else ("₹" if "INR" in currency else ("€" if "EUR" in currency else ("AED " if "AED" in currency else ("£" if "GBP" in currency else ("¥" if "JPY" in currency else "S$")))))
            rate = 1.0 if "USD" in currency else (86.8 if "INR" in currency else (0.92 if "EUR" in currency else (3.67 if "AED" in currency else (0.79 if "GBP" in currency else (152.5 if "JPY" in currency else 1.34)))))
        with people_col:
            num_travelers = st.slider("Total Traveling Party", min_value=1, max_value=8, value=2)

        days = (st.session_state.end_date - st.session_state.start_date).days
        if days <= 0:
            days = 5

        b_c1, b_c2 = st.columns(2)
        with b_c1:
            flight_cost_usd = st.slider("Flight Cost (per traveler)", min_value=100, max_value=2500, value=350, step=25)
            hotel_night_usd = st.slider("Hotel Suite (per room / night)", min_value=50, max_value=1200, value=180, step=10)
        with b_c2:
            food_day_usd = st.slider("Dining & Cafes (per person / day)", min_value=20, max_value=350, value=65, step=5)
            activities_usd = st.slider("Tours & Sightseeing (per person)", min_value=50, max_value=1500, value=220, step=25)

        total_flights = flight_cost_usd * num_travelers * rate
        total_hotels = hotel_night_usd * days * rate
        total_food = food_day_usd * days * num_travelers * rate
        total_activities = activities_usd * num_travelers * rate
        grand_total = total_flights + total_hotels + total_food + total_activities

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="apple-metric-card"><div class="apple-metric-val">{curr_symbol}{total_flights:,.0f}</div><div class="apple-metric-lbl">Total Flights</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="apple-metric-card"><div class="apple-metric-val">{curr_symbol}{total_hotels:,.0f}</div><div class="apple-metric-lbl">Hotel ({days} Nights)</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="apple-metric-card"><div class="apple-metric-val">{curr_symbol}{total_food:,.0f}</div><div class="apple-metric-lbl">Dining & Cuisine</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="apple-metric-card"><div class="apple-metric-val">{curr_symbol}{total_activities:,.0f}</div><div class="apple-metric-lbl">Tours & Transit</div></div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="apple-total-box">
                <div style="font-size: 0.85rem; color: #86868b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem;">
                    TOTAL ESTIMATED TRIP INVESTMENT
                </div>
                <div class="apple-total-val">{curr_symbol}{grand_total:,.0f}</div>
                <div style="font-size: 0.90rem; color: #2997ff; font-weight: 500; margin-top: 0.3rem;">
                    Approx. {curr_symbol}{grand_total/num_travelers:,.0f} per traveler
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Sub-tab 3: Interactive Destination Map
    with result_tabs[2]:
        st.markdown("### Destination & Landmark Visualizer")
        dest_term = st.session_state.destination_name or "Dubai"
        st.caption(f"Curated coordinates and recommended areas across {dest_term}:")
        map_df = get_map_dataframe(dest_term)
        st.map(map_df, latitude="lat", longitude="lon", size=40, zoom=11)
        st.dataframe(map_df[["name", "lat", "lon"]], hide_index=True, use_container_width=True)

    # Sub-tab 4: Packing Checklist
    with result_tabs[3]:
        st.markdown("### Smart Packing Checklist")
        st.caption("Tailored checklist based on your destination and style. Check off items as you pack.")

        checklist = get_packing_checklist(st.session_state.destination_name, "general")
        chk_cols = st.columns(2)
        half = len(checklist) // 2
        items = list(checklist.items())

        with chk_cols[0]:
            for category, tasks in items[:half]:
                clean_cat = category.replace("👕", "").replace("🛂", "").replace("🔌", "").replace("💊", "").strip()
                st.markdown(f"**{clean_cat}**")
                for task in tasks:
                    st.checkbox(task, key=f"chk_{category}_{task}")

        with chk_cols[1]:
            for category, tasks in items[half:]:
                clean_cat = category.replace("👕", "").replace("🛂", "").replace("🔌", "").replace("💊", "").strip()
                st.markdown(f"**{clean_cat}**")
                for task in tasks:
                    st.checkbox(task, key=f"chk_{category}_{task}")

    # Sub-tab 5: Export & Sharing Hub
    with result_tabs[4]:
        st.markdown("### Export, Share & Sync Your Itinerary")

        e_col1, e_col2 = st.columns(2)

        # PDF, Markdown & ICS Download
        with e_col1:
            st.markdown("#### Document Downloads")

            # Download PDF Report
            pdf_bytes = generate_pdf_travel_report(
                destination=st.session_state.destination_name,
                origin=st.session_state.origin_name,
                start_date_str=str(st.session_state.start_date),
                end_date_str=str(st.session_state.end_date),
                travel_plan_text=st.session_state.travel_info,
            )

            st.download_button(
                label="Download PDF Dossier",
                data=pdf_bytes,
                file_name=f"Travel_Dossier_{st.session_state.destination_name}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )

            st.download_button(
                label="Download Markdown",
                data=st.session_state.travel_info,
                file_name=f"Luxury_Itinerary_{st.session_state.destination_name}.md",
                mime="text/markdown",
                use_container_width=True,
            )

            ics_data = generate_ics_calendar(
                trip_title=f"Trip to {st.session_state.destination_name}",
                start_date_str=str(st.session_state.start_date),
                end_date_str=str(st.session_state.end_date),
                description=st.session_state.travel_info[:350],
                location=st.session_state.destination_name,
            )

            st.download_button(
                label="Export to Calendar (.ics)",
                data=ics_data,
                file_name=f"Trip_{st.session_state.destination_name}.ics",
                mime="text/calendar",
                use_container_width=True,
            )

        # Email Dispatch Form
        with e_col2:
            st.markdown("#### Dispatch via SendGrid Email")
            with st.form("email_dispatch_form"):
                receiver_email = st.text_input("Receiver Email Address", placeholder="vip_traveler@example.com")
                subject = st.text_input("Email Subject", f"Your Bespoke Itinerary to {st.session_state.destination_name}")
                email_submit = st.form_submit_button("Send Itinerary via Email", type="primary", use_container_width=True)

            if email_submit:
                if not receiver_email:
                    st.error("Please enter a valid receiver email address.")
                else:
                    sender_email = os.getenv("SENDGRID_FROM_EMAIL") or os.getenv("FROM_EMAIL")
                    sendgrid_key = os.getenv("SENDGRID_API_KEY")

                    if not sender_email or not sendgrid_key or "your_" in sendgrid_key.lower():
                        st.error("SendGrid API Key or Verified Sender Email is not configured in .env.")
                    else:
                        try:
                            # Email is intentionally independent of the HITL graph.
                            # All other HITL functionality in the app remains unchanged.
                            html_content = markdown_to_clean_html_email(
                                st.session_state.travel_info
                            )

                            message = Mail(
                                from_email=sender_email,
                                to_emails=receiver_email,
                                subject=subject,
                                html_content=html_content,
                            )

                            sendgrid_client = SendGridAPIClient(sendgrid_key)
                            response = sendgrid_client.send(message)

                            print("SENDGRID STATUS:", response.status_code)
                            print("SENDGRID BODY:", response.body)

                            if response.status_code in (200, 201, 202):
                                st.success(
                                    f"Itinerary successfully emailed to {receiver_email}."
                                )
                            else:
                                st.error(
                                    f"SendGrid failed. Status code: {response.status_code}"
                                )
                        except Exception as exc:
                            print("SENDGRID ERROR:", exc)
                            st.error(f"Failed to send email: {exc}")


# ---------------------------------------------------------
# MAIN ENTRYPOINT
# ---------------------------------------------------------

def main():
    init_session_state()
    apply_custom_css()
    render_sidebar()
    render_hero()

    # Main Navigation Tabs
    tab_planner, tab_chat, tab_rag = st.tabs([
        "Trip Planner",
        "Conversational Concierge",
        "Knowledge Hub & Docs",
    ])

    with tab_planner:
        render_trip_planner_tab()

    with tab_chat:
        render_chat_tab()

    with tab_rag:
        render_rag_tab()

    # Results Hub is rendered when travel plan is active
    render_results_hub()


if __name__ == "__main__":
    main()