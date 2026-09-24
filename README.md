# Agentic AI Travel Planner & Concierge — Production Edition ✈️ 🌍

A production-ready **Agentic AI Travel Planner** built with **LangGraph, Groq LLM Engine, Hybrid RAG (Dense Vector + BM25 + Reciprocal Rank Fusion), Multi-Tool Orchestration, Human-in-the-Loop (HITL) Interrupts, Reflection & Validation, and Automated Evaluation Benchmarking**.

---

## 🌟 Key Architecture & Capabilities

```
+---------------------------------------------------------------------------------------------------+
|                                  Streamlit Web Interface                                          |
|  [Trip Planner Form]  [HITL Actions]  [Concierge Chat]  [RAG Hub]  [Observability]  [Evaluation]  |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+───────────────────────────────────────────────────────────────────────────────────────────────────+
|                                    LangGraph Agentic StateGraph                                   |
|                                                                                                   |
|  1. [Planner & Router] ──► Classify intent (Single-Tool vs Multi-Tool Orchestrated Plan)          |
|  2. [Groq LLM Engine] ───► Sub-second reasoning across gpt-oss-120b, gpt-oss-20b, llama-3.3     |
|  3. [Tool Execution]  ───► Flights, Hotels, RAG Knowledge, Weather, Forex, Maps, Budget Optimizer|
|  4. [Reflection Agent]───► Date validation, price math, budget compliance, RAG grounding checks   |
|  5. [HITL Interrupt]  ───► Human approvals (Approve, Modify, Reject, Search Again, Rebalance)     |
|  6. [SendGrid / Export]──► Mobile-responsive HTML email dispatch & ReportLab PDF travel dossier   |
+───────────────────────────────────────────────────────────────────────────────────────────────────+
                                                  │
                                                  ▼
+───────────────────────────────────────────────────────────────────────────────────────────────────+
|                                      RAG Knowledge Pipeline                                       |
|  - Ingestion: PDF, TXT, MD, JSON with metadata extraction                                         |
|  - Semantic Chunker: Sliding-window with sentence boundary preservation & token budgets           |
|  - Vector Database: Dense cosine similarity embedding + BM25 keyword inverted index               |
|  - Hybrid Reranker: Reciprocal Rank Fusion (RRF) with metadata filtering (destination, category)  |
|  - Grounding & Citations: Source references with relevance metrics & hallucination detection     |
+───────────────────────────────────────────────────────────────────────────────────────────────────+
```

---

## 🛠️ Specialized Tool Ecosystem

| Tool | Purpose | Data Source & Fallback |
| :--- | :--- | :--- |
| `flights_finder` | Search real-time flights, schedules, durations, stops, and direct booking links | Google Flights via SerpAPI + Verified Caching + Realistic Route Fallbacks |
| `hotels_finder` | Find stays, ratings, prices per night, total stay costs, and amenities | Google Hotels via SerpAPI + Verified Caching + Fallback Database |
| `travel_knowledge_search` | Dedicated RAG retrieval for visa regulations, entry rules, customs, baggage, safety | In-Memory Vector Store + BM25 Hybrid Index + Metadata Filter |
| `search_travel_guide` | Exact backward-compatible alias for `travel_knowledge_search` | Shared RAG Engine |
| `weather_search` | Live temperatures, rain probability, seasonal trends, packing advice | Live Meteorological Sensor (Open-Meteo / wttr.in) + Climate Database |
| `currency_conversion` | Real-time forex conversion across USD, EUR, GBP, INR, AED, JPY, SGD, etc. | Live Open Exchange Rates API + Fixed Forex Matrix |
| `calculate_trip_budget` | Itemized calculation across flights, hotels, dining, tours, and 10% contingency | Multi-currency cost calculator |
| `maps_location_search` | Spatial geocoding, central coordinates, landmark clusters, transit hubs | Geo-coordinates Database + Leaflet `st.map` dataframe |
| `budget_optimizer` | Automatic expense reallocation and actionable cost-saving substitutions | Budget Rebalancing Algorithm |

---

## 🤝 Human-in-the-Loop (HITL) Workflow

The system uses **LangGraph Checkpointing (`MemorySaver`)** and interruption boundaries to keep the traveler in control of critical decisions:
- **Approval Actions**:
  - `✅ Approve Plan`: Finalize and unlock instant PDF and SendGrid dispatch.
  - `💰 Optimize Budget`: Feeds back instructions to rebalance stays and flight tiers.
  - `🏨 Find Better Hotel`: Requests alternate boutique/luxury accommodations.
  - `🔄 Search Again`: Re-queries live sources with adjusted parameters.
  - `✍️ Natural Language Feedback`: Supports custom revisions like *"Make Day 2 vegetarian"* or *"Add art museum in the afternoon"*.

---

## 🛡️ Reflection & Validation Agent

Before delivering any travel plan, the **Validator Node** executes 5 validation checks:
1. **Date Chronology**: Verifies departure date precedes return date and calculates exact stay duration.
2. **Section Completeness**: Confirms Flights, Hotels, Visa, and Day-by-Day itinerary are present.
3. **Link Validity**: Ensures clickable markdown links (`[🔗 Book Flight](...)`, `[🏨 View Hotel](...)`) are properly formatted.
4. **Budget & Pricing Math**: Confirms numerical prices match total calculations and stay within user budget limits.
5. **Destination Grounding**: Validates recommendations align with requested destination without hallucinations.

*If validation fails, the agent self-corrects using reflective prompt loops.*

---

## 📊 Automated Evaluation Suite

Built-in quantitative evaluation benchmark (`eval/evaluator.py`) measuring:
- **Retrieval Relevance (Precision@K & Recall@K)**: Evaluates semantic overlap between query and retrieved chunks.
- **Answer Faithfulness**: Factual grounding score verifying claims against context.
- **Citation Correctness**: Valid source attribution and booking link presence.
- **Tool-Selection Accuracy**: Precision, Recall, and F1 score against expected tool routes.
- **Hallucination Detection**: Detects unverified pricing outliers and placeholder tokens.

---

## 📑 Export & Dispatch Suite

- **📑 Downloadable PDF Travel Dossier**: Clean multi-page document generated via `reportlab` with flight cards, hotel details, day-by-day table, budget summary, and source citations.
- **📧 SendGrid Responsive HTML Email**: High-converting, mobile-friendly email template with inline CSS.
- **📅 iCalendar (.ics)**: One-click export to Google Calendar, Apple Calendar, or Outlook.
- **📄 Markdown (.md)**: Full text export.

---

## 🚀 Quick Start & Setup

### 1. Installation
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Environment Variables (`.env`)
```env
GROQ_API_KEY=your_groq_api_key
SERPAPI_API_KEY=your_serpapi_api_key
SENDGRID_API_KEY=your_sendgrid_api_key
SENDGRID_FROM_EMAIL=your_verified_sender@example.com
```
*(Note: All external APIs feature graceful mock/simulators so the application functions seamlessly even without live API keys).*

### 3. Run Automated Tests
```powershell
python -m pytest tests/ -v
```

### 4. Launch Streamlit Application
```powershell
streamlit run app.py
```

---

## ☁️ Deployment on Streamlit Community Cloud

The application is engineered to deploy seamlessly on **Streamlit Community Cloud**:
1. Push this repository to GitHub.
2. Connect your repo in [share.streamlit.io](https://share.streamlit.io).
3. Set your Secrets under **App Settings > Secrets**:
   ```toml
   GROQ_API_KEY = "gsk_..."
   SERPAPI_API_KEY = "..."
   SENDGRID_API_KEY = "SG...."
   SENDGRID_FROM_EMAIL = "verified@domain.com"
   ```
4. Set main file path to `app.py`.
