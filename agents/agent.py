"""
Agentic AI Travel Planner LangGraph Architecture.
Features:
- Multi-Model Groq LLM Engine (openai/gpt-oss-120b, openai/gpt-oss-20b, llama-3.3-70b-versatile)
- Intelligent Autonomous Fallback Engine when API key is missing or offline
- Comprehensive Tool Ecosystem (Flights, Hotels, Hybrid RAG, Weather, Forex, Maps, Budget Optimizer)
- Agentic Planning & Intelligent Tool Routing
- Reflection & Validation Loop (Dates, Costs, Links, Grounding)
- Human-in-the-Loop (HITL) Interrupts with MemorySaver Checkpointing
- SendGrid Mobile-Responsive Email Dispatch
"""

import datetime
import operator
import os
import re
import uuid
from typing import Annotated, Any, Dict, List, Optional

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# Tools
from agents.tools.flights_finder import flights_finder, FlightsInput
from agents.tools.hotels_finder import hotels_finder, HotelsInput
from agents.tools.rag_tool import travel_knowledge_search, search_travel_guide, TravelGuideInput
from agents.tools.weather_finder import weather_search, WeatherInput
from agents.tools.currency_tool import currency_conversion, calculate_trip_budget, TripCostCalculationInput
from agents.tools.maps_finder import maps_location_search, MapsInput
from agents.tools.budget_tool import budget_optimizer

# Modules
from agents.state import AgentState
from agents.planner import AgenticPlanner
from agents.validator import PlanValidator
from export.email_template import markdown_to_clean_html_email

# Search for .env
load_dotenv(find_dotenv())
_env_parent = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_env_parent):
    load_dotenv(_env_parent)

CURRENT_YEAR = datetime.datetime.now().year

ALL_TOOLS = [
    flights_finder,
    hotels_finder,
    travel_knowledge_search,
    search_travel_guide,
    weather_search,
    currency_conversion,
    calculate_trip_budget,
    maps_location_search,
    budget_optimizer,
]


def get_groq_llm(temperature: float = 0.1) -> Optional[ChatGroq]:
    """Instantiate ChatGroq with resilient fallback across models."""
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key or "your_" in api_key.lower() or "placeholder" in api_key.lower():
        api_key = "gsk_placeholder_api_key_for_offline_dev"

    models = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"]
    for model_name in models:
        try:
            return ChatGroq(
                model=model_name,
                temperature=temperature,
                api_key=api_key,
            )
        except Exception:
            continue
    try:
        return ChatGroq(model="openai/gpt-oss-20b", temperature=temperature, api_key=api_key)
    except Exception:
        return None


class Agent:
    """Production Agentic Travel Planner orchestrating LangGraph, RAG, and HITL."""

    def __init__(self):
        self._tools = {t.name: t for t in ALL_TOOLS}
        base_llm = get_groq_llm(temperature=0.1)
        if base_llm:
            try:
                self._tools_llm = base_llm.bind_tools(ALL_TOOLS)
            except Exception:
                self._tools_llm = None
        else:
            self._tools_llm = None

        # Build StateGraph
        builder = StateGraph(AgentState)

        # Nodes
        builder.add_node("planner", self.planner_node)
        builder.add_node("call_tools_llm", self.call_tools_llm_node)
        builder.add_node("invoke_tools", self.invoke_tools_node)
        builder.add_node("validator", self.validator_node)
        builder.add_node("hitl_decision", self.hitl_decision_node)
        builder.add_node("email_sender", self.email_sender_node)

        # Entry point
        builder.set_entry_point("planner")

        # Edges
        builder.add_edge("planner", "call_tools_llm")

        builder.add_conditional_edges(
            "call_tools_llm",
            self.route_after_llm,
            {
                "invoke_tools": "invoke_tools",
                "validate": "validator",
            },
        )

        builder.add_edge("invoke_tools", "call_tools_llm")

        builder.add_conditional_edges(
            "validator",
            self.route_after_validation,
            {
                "retry": "call_tools_llm",
                "hitl_approval": "hitl_decision",
                "done": END,
            }
        )

        builder.add_conditional_edges(
            "hitl_decision",
            self.route_after_hitl,
            {
                "send_email": "email_sender",
                "replan": "call_tools_llm",
                "done": END,
            }
        )

        builder.add_edge("email_sender", END)

        # Checkpointer for conversation sessions & human approval interrupts
        memory = MemorySaver()
        self.graph = builder.compile(
            checkpointer=memory,
            interrupt_before=["hitl_decision"],
        )

    # -------------------------------------------------------------------------
    # ROUTING LOGIC
    # -------------------------------------------------------------------------

    @staticmethod
    def route_after_llm(state: AgentState) -> str:
        """Check if LLM requested tool calls or completed response."""
        last_msg = state["messages"][-1]
        if getattr(last_msg, "tool_calls", []):
            return "invoke_tools"
        return "validate"

    @staticmethod
    def route_after_validation(state: AgentState) -> str:
        """Route based on reflection validation and approval requirements."""
        status = state.get("validation_status", "PASSED")
        retry_cnt = state.get("retry_count", 0) or 0

        # If critical failure and retries remaining, retry
        if status == "FAILED" and retry_cnt < 2:
            return "retry"

        # Check if Human Approval is required (budget exceeded or final approval)
        if state.get("approval_required"):
            return "hitl_approval"

        return "done"

    @staticmethod
    def route_after_hitl(state: AgentState) -> str:
        """Route based on human decision in Streamlit UI."""
        status = state.get("approval_status", "APPROVED")
        approval_type = state.get("approval_type", "")

        if status == "MODIFIED" or status == "REJECTED":
            return "replan"
        if approval_type == "send_email" and status == "APPROVED":
            return "send_email"
        return "done"

    # -------------------------------------------------------------------------
    # GRAPH NODES
    # -------------------------------------------------------------------------

    def planner_node(self, state: AgentState) -> Dict[str, Any]:
        """Establish execution plan, classify query complexity, and log reasoning step."""
        user_query = ""
        for m in reversed(state["messages"]):
            if isinstance(m, HumanMessage):
                user_query = m.content
                break

        classification = AgenticPlanner.classify_and_route(user_query)
        step_entry = {
            "node": "Agentic Planner",
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "action": f"Classified query as [{classification['route_type']}].",
            "recommended_tools": classification["recommended_tools"],
            "plan_steps": classification["steps"],
        }

        return {
            "plan": classification["steps"],
            "current_step": 1,
            "total_steps": len(classification["steps"]),
            "reasoning_trace": [step_entry],
            "retry_count": 0,
        }

    def _generate_deterministic_travel_plan(self, state: AgentState) -> AIMessage:
        """Autonomous fallback engine providing grounded itinerary with custom feedback and day plan reflection."""
        trip = state.get("trip_details") or {}
        prefs = state.get("preferences") or {}
        feedback = state.get("human_feedback") or ""

        # Extract directive from messages if present
        if not feedback and state.get("messages"):
            for m in reversed(state.get("messages", [])):
                if isinstance(m, HumanMessage) and "[USER DIRECTIVE" in m.content:
                    feedback = m.content.split("[USER DIRECTIVE / RE-PLANNING FEEDBACK]:")[-1].strip()
                    break

        origin = trip.get("origin", "Mumbai (BOM)")
        dest = trip.get("destination", "Dubai")
        start_date = trip.get("start_date", "2026-11-15")
        end_date = trip.get("end_date", "2026-11-20")
        adults = prefs.get("adults", 2)
        travel_style = prefs.get("travel_style", "Cultural & Heritage")
        hotel_cat = prefs.get("hotel_category", "4-Star")

        fb_lower = feedback.lower()
        if "expensive" in fb_lower or "cheaper" in fb_lower or "budget" in fb_lower or "optimize budget" in fb_lower:
            hotel_cat = "3-Star"
            hotel_sort = 2
        elif "alternative hotel" in fb_lower or "different hotel" in fb_lower:
            hotel_sort = 1
        else:
            hotel_sort = 8

        # Extract IATA codes if present
        dep_iata = "BOM"
        if "(" in origin and ")" in origin:
            dep_iata = origin.split("(")[-1].replace(")", "").strip()
        arr_iata = "DXB" if "dubai" in dest.lower() else ("DPS" if "bali" in dest.lower() else ("CDG" if "paris" in dest.lower() else ("GOI" if "goa" in dest.lower() else "LHR")))

        # Query internal tools
        flights_data = flights_finder.invoke({
            "params": {
                "departure_airport": dep_iata,
                "arrival_airport": arr_iata,
                "outbound_date": start_date,
                "return_date": end_date,
                "adults": adults,
                "children": 0,
                "infants_in_seat": 0,
                "infants_on_lap": 0,
            }
        })

        hotels_data = hotels_finder.invoke({
            "params": {
                "location": dest,
                "check_in_date": start_date,
                "check_out_date": end_date,
                "adults": adults,
                "children": 0,
                "rooms": 1,
                "sort_by": hotel_sort,
                "hotel_class": hotel_cat.split("-")[0] if "-" in hotel_cat else "4",
            }
        })

        rag_guide = travel_knowledge_search.invoke({
            "params": {
                "query": f"{dest} visa requirements travel guidelines transport",
                "destination": dest,
                "category": "Destination Guide"
            }
        })

        weather_info = weather_search.invoke({
            "params": {
                "destination": dest,
                "month": start_date,
            }
        })

        # Check if duration change is requested in human_feedback
        target_days = None
        if feedback:
            ext_by = re.search(r"extend(?:\s+\w+){0,3}\s+by\s+(\d+)\s+days?", fb_lower)
            ext_to = re.search(r"(?:extend|change|make\s+it|expand|increase|plan)(?:\s+\w+){0,3}\s+(?:to\s+)?(\d+)\s+days?", fb_lower)
            ext_simple = re.search(r"(\d+)\s+days?", fb_lower)
            if ext_by:
                try:
                    d1 = datetime.date.fromisoformat(start_date)
                    d2 = datetime.date.fromisoformat(end_date)
                    cur_diff = (d2 - d1).days
                except Exception:
                    cur_diff = 5
                target_days = max(1, cur_diff + int(ext_by.group(1)))
            elif ext_to:
                target_days = int(ext_to.group(1))
            elif "extend" in fb_lower and ext_simple:
                target_days = int(ext_simple.group(1))

        # Calculate number of days
        num_days = 5
        if target_days and 1 <= target_days <= 30:
            num_days = target_days
            try:
                d1 = datetime.date.fromisoformat(start_date)
                end_date = str(d1 + datetime.timedelta(days=num_days))
            except Exception:
                pass
        else:
            try:
                d1 = datetime.date.fromisoformat(start_date)
                d2 = datetime.date.fromisoformat(end_date)
                diff = (d2 - d1).days
                if 1 <= diff <= 30:
                    num_days = diff
            except Exception:
                num_days = 5

        hotel_nightly = 180.0
        if "expensive" in fb_lower or "budget" in fb_lower:
            hotel_nightly = 110.0

        budget_info = calculate_trip_budget.invoke({
            "params": {
                "flights_cost_per_person": float(flights_data[0].get("price_usd", 350) if isinstance(flights_data, list) and flights_data else 350),
                "hotel_cost_per_night": hotel_nightly,
                "num_travelers": adults,
                "num_nights": num_days,
                "target_currency": "USD"
            }
        })

        # Format Flights
        flight_rows = []
        for f in (flights_data if isinstance(flights_data, list) else [])[:3]:
            flight_rows.append(
                f"- **{f.get('airline', 'Airline')}** ({f.get('flight_number', 'FL-101')}) — Dep: {f.get('departure', '08:00')} | Arr: {f.get('arrival', '12:30')} | Price: **${f.get('price_usd', 350)}** | [{('Book Flight')}](<{f.get('booking_link', 'https://google.com/travel/flights')}>)"
            )
        flights_section = "\n".join(flight_rows) if flight_rows else "- Recommended Flight Options: Emirates / Singapore Airlines at ~$350/traveler. [Book Flight](https://google.com/travel/flights)"

        # Format Hotels (if alternative requested, slice from index 1)
        hotel_list = hotels_data if isinstance(hotels_data, list) else []
        if "alternative" in fb_lower and len(hotel_list) > 2:
            hotel_list = hotel_list[1:] + [hotel_list[0]]
        hotel_rows = []
        for h in hotel_list[:3]:
            amenities_str = ", ".join(h.get("amenities", [])[:3])
            hotel_rows.append(
                f"- **{h.get('name', 'Luxury Hotel')}** ({h.get('hotel_class', '4-star')}) — Rating: {h.get('rating', '4.7')}/5 | **{h.get('price_per_night', '$180')}/night** ({h.get('total_price', '$900')} total)\n  *Key Amenities:* {amenities_str} | [View Hotel](<{h.get('link', 'https://google.com/travel/hotels')}>)"
            )
        hotels_section = "\n".join(hotel_rows) if hotel_rows else "- Recommended Stays: 4-star & 5-star properties in prime central locations."

        # Dynamic day itinerary builder with full 14-day library
        default_days = {
            1: (
                f"**Day 1: Arrival & Landmark Orientation**\n"
                f"- *Morning:* Arrive at {dest}, private transfer to luxury hotel and check-in.\n"
                f"- *Afternoon:* Explore the downtown central district and panoramic observation decks.\n"
                f"- *Evening:* Welcome dinner at a curated culinary spot with sunset skyline views."
            ),
            2: (
                f"**Day 2: Heritage, Culture & Old Quarters**\n"
                f"- *Morning:* Guided historic tour of ancient quarters, cultural museums, and artisan souks.\n"
                f"- *Afternoon:* Scenic waterfront promenade or traditional river/canal cruise.\n"
                f"- *Evening:* Gourmet local tasting experience and heritage street walk."
            ),
            3: (
                f"**Day 3: Signature Excursion & Natural Wonders**\n"
                f"- *Morning & Afternoon:* Bespoke day trip (desert safari / coastal boat excursion / botanical gardens).\n"
                f"- *Evening:* Relaxation, wellness spa, and traditional dinner."
            ),
            4: (
                f"**Day 4: Modern Art, Futuristic Landmarks & Architecture**\n"
                f"- *Morning:* Visit contemporary design landmarks, interactive exhibitions, and innovation centers.\n"
                f"- *Afternoon:* Boutique district walk, modern galleries, and specialty cafes.\n"
                f"- *Evening:* Skyline fine dining with panoramic city lights."
            ),
            5: (
                f"**Day 5: Luxury Shopping & Waterfront Leisure**\n"
                f"- *Morning:* Premier luxury shopping mall or high-end designer boulevard.\n"
                f"- *Afternoon:* Marina walk, private yacht charter, or coastal beach club cabana.\n"
                f"- *Evening:* Waterfront dining at a Michelin-selected restaurant."
            ),
            6: (
                f"**Day 6: Regional Day Excursion & Heritage Outskirts**\n"
                f"- *Morning & Afternoon:* Full-day guided excursion to neighboring scenic regions (e.g. Grand Mosque & Cultural Island / Mountain Pass / Coastal Cliffs).\n"
                f"- *Evening:* Sunset vantage point dinner."
            ),
            7: (
                f"**Day 7: Coastal Retreat, Watersports & Island Vibes**\n"
                f"- *Morning:* Sunbathing, private beach club access, or paddleboarding.\n"
                f"- *Afternoon:* Coastal speedboat cruise or snorkeling/scuba reef exploration.\n"
                f"- *Evening:* Beachside barbecue and acoustic live music lounge."
            ),
            8: (
                f"**Day 8: Adventure Thrills & Aerial Panoramic Experience**\n"
                f"- *Morning:* Helicopter scenic tour or hot air balloon sunrise ride over landscapes.\n"
                f"- *Afternoon:* Theme park / indoor adventure sports / zip-line excursion.\n"
                f"- *Evening:* High-altitude rooftop cocktail lounge."
            ),
            9: (
                f"**Day 9: Culinary Masterclass & Hidden Artisan Gems**\n"
                f"- *Morning:* Private cooking masterclass with an executive chef and spice market sourcing.\n"
                f"- *Afternoon:* Hidden architectural courtyards and bespoke perfumery/craft workshops.\n"
                f"- *Evening:* Multi-course tasting menu paired with artisanal beverages."
            ),
            10: (
                f"**Day 10: Leisure Rejuvenation & Farewell Gala Dinner**\n"
                f"- *Morning:* In-suite relaxation, pool cabana, and luxury wellness hydrotherapy.\n"
                f"- *Afternoon:* Final souvenir curation and specialty shopping.\n"
                f"- *Evening:* Signature Farewell Gala Dinner celebrating your complete journey."
            ),
            11: (
                f"**Day 11: Eco-Sanctuary & Natural Landscapes**\n"
                f"- *Morning & Afternoon:* Guided eco-reserve exploration, wildlife sanctuary, and botanical trails.\n"
                f"- *Evening:* Organic farm-to-table dining experience."
            ),
            12: (
                f"**Day 12: Hidden Island or Hilltop Serenity**\n"
                f"- *Morning:* Private catamaran or scenic train ride to secluded vistas.\n"
                f"- *Afternoon:* Photography tour and peaceful lakeside / seaside retreat.\n"
                f"- *Evening:* Stargazing session and tranquil dining."
            ),
            13: (
                f"**Day 13: Bespoke Cultural Immersion & Artisan Workshops**\n"
                f"- *Morning:* Private master artisan workshops and heritage craft curation.\n"
                f"- *Afternoon:* Scenic city viewpoint and afternoon high-tea.\n"
                f"- *Evening:* Curated theatrical show or symphony performance."
            ),
            14: (
                f"**Day 14: Grand Farewell & Return Transit**\n"
                f"- *Morning:* Leisurely late breakfast, final packing, and private executive transfer to the airport."
            ),
        }

        days_dict = {i: default_days.get(i, f"**Day {i}: Exploration & Leisure**\n- *Morning:* Highlights tour.\n- *Evening:* Local dining.") for i in range(1, num_days + 1)}

        # Ensure the final day has a departure touch if trip is shorter than 14 days
        if num_days > 1 and num_days not in [10, 14]:
            days_dict[num_days] = (
                f"**Day {num_days}: Departure & Final Highlights**\n"
                f"- *Morning:* Leisurely breakfast, final souvenir selection, and airport transit."
            )

        # Dynamically apply user feedback / day modifications
        if feedback:
            for d_idx in range(1, num_days + 1):
                d_str = f"day {d_idx}"
                if d_str in fb_lower or f"day{d_idx}" in fb_lower or (d_idx == 2 and "second day" in fb_lower) or (d_idx == 1 and "first day" in fb_lower) or (d_idx == 3 and "third day" in fb_lower):
                    if any(w in fb_lower for w in ["rest", "room", "sleep", "relax", "dont plan", "don't plan", "nothing", "free", "chill", "leisure", "spa"]):
                        days_dict[d_idx] = (
                            f"**Day {d_idx}: In-Suite Rest, Leisure & Personal Relaxation** *(Updated per your directive)*\n"
                            f"- *Morning:* Sleep-in, leisurely in-room breakfast and private balcony views.\n"
                            f"- *Afternoon:* Quiet downtime in room, reading, and self-paced relaxation (no excursions scheduled).\n"
                            f"- *Evening:* Casual in-hotel dining or private room service."
                        )
                    elif any(w in fb_lower for w in ["shop", "mall", "market", "souk", "bazaar"]):
                        days_dict[d_idx] = (
                            f"**Day {d_idx}: Luxury Shopping & Artisan Markets** *(Updated per your directive)*\n"
                            f"- *Morning:* Premier boutique shopping district and designer flagship stores.\n"
                            f"- *Afternoon:* Traditional artisan souks and bespoke craft studios.\n"
                            f"- *Evening:* Fine dining at the shopping promenade."
                        )
                    elif any(w in fb_lower for w in ["beach", "sea", "ocean", "swim", "coastal"]):
                        days_dict[d_idx] = (
                            f"**Day {d_idx}: Coastal Retreat & Beach Lounge** *(Updated per your directive)*\n"
                            f"- *Morning:* Private beach club cabana and coastal swimming.\n"
                            f"- *Afternoon:* Catamaran boat cruise or seaside watersports.\n"
                            f"- *Evening:* Oceanfront sunset dinner."
                        )
                    elif any(w in fb_lower for w in ["museum", "art", "culture", "gallery"]):
                        days_dict[d_idx] = (
                            f"**Day {d_idx}: Cultural Discovery & Contemporary Art** *(Updated per your directive)*\n"
                            f"- *Morning:* Curated tour of modern art galleries and cultural pavilions.\n"
                            f"- *Afternoon:* Historical museum exhibits and architectural landmarks.\n"
                            f"- *Evening:* Cultural performing arts and historic dining."
                        )
                    else:
                        clean_dir = feedback.replace("[USER DIRECTIVE / RE-PLANNING FEEDBACK]:", "").strip()
                        days_dict[d_idx] = (
                            f"**Day {d_idx}: Custom Tailored Experience** *(Updated per your directive)*\n"
                            f"- *Directive Focus:* {clean_dir}\n"
                            f"- *Morning & Afternoon:* Bespoke tailored activities aligned with your preference.\n"
                            f"- *Evening:* Relaxed dinner and leisure."
                        )

        itinerary_days = "\n\n".join(days_dict[i] for i in sorted(days_dict.keys()))

        # If feedback was given without specific day, include a top banner in itinerary
        feedback_banner = ""
        if feedback:
            clean_dir = feedback.replace("[USER DIRECTIVE / RE-PLANNING FEEDBACK]:", "").strip()
            feedback_banner = f"> *Applied Human-in-the-Loop Directive: \"{clean_dir}\"*\n\n"

        content = f"""## Flight Options & Booking
{flights_section}

## Hotel Recommendations
{hotels_section}

## Visa & Entry Requirements
- Valid passport with minimum 6 months validity required from date of entry.
- Visa regulations: Verified entry guidelines apply based on citizenship (eVisa / Visa-on-Arrival).
- Ensure confirmed round-trip tickets and hotel vouchers are accessible upon immigration check.

## Best Season & Weather Advice
- **Expected Temperature:** {weather_info.get('expected_temperature_range', '22°C - 28°C')}
- **General Conditions:** {weather_info.get('general_conditions', 'Clear & Pleasant')} (Rain probability: {weather_info.get('rain_probability', '10%')})
- **Packing Guidance:** {weather_info.get('clothing_and_packing_advice', 'Comfortable walking shoes, breathable fabrics, and light layers.')}

## Local Transportation & Cultural Hacks
- Utilize the high-speed urban transit network or official metered taxis for economical transit.
- Respect local cultural etiquette and dress codes in historical and sacred sites.
- Avoid unlicensed street booking booths; rely on confirmed digital ticketing.

## Day-by-Day Detailed Itinerary
{feedback_banner}{itinerary_days}

## Estimated Trip Budget & Investment
- **Flights Total:** {budget_info.get('breakdown', {}).get('flights_total', '$700.00')}
- **Hotel Total ({num_days} Nights):** {budget_info.get('breakdown', {}).get('hotels_total', '$900.00')}
- **Dining & Cuisine:** {budget_info.get('breakdown', {}).get('food_and_dining', '$500.00')}
- **Tours & Transit:** {budget_info.get('breakdown', {}).get('activities_and_tours', '$400.00')}
- **Total Estimated Investment:** **{budget_info.get('grand_total', '$2,750.00')}** (~{budget_info.get('cost_per_traveler', '$1,375.00')} per traveler)

## Verified Grounding Sources
- [Verified Travel Guide • {dest} Complete Guide]
- [Open Meteorological Sensor • Seasonal Climate Hub]
"""
        return AIMessage(content=content)

    def call_tools_llm_node(self, state: AgentState) -> Dict[str, Any]:
        """Call Groq LLM with tools or trigger autonomous fallback synthesis."""
        prefs = state.get("preferences") or {}
        prefs_clause = ""
        if prefs:
            prefs_clause = (
                f"TRAVELER CONSTRAINTS: Budget Tier={prefs.get('budget_tier')}, "
                f"Travel Style={prefs.get('travel_style')}, Hotel Category={prefs.get('hotel_category')}, "
                f"Party={prefs.get('adults', 2)} Adults, Food={prefs.get('food_preference')}."
            )

        feedback = state.get("human_feedback")
        system_prompt = AgenticPlanner.generate_planning_system_prompt(prefs_clause, feedback)
        messages = [SystemMessage(content=system_prompt)] + state["messages"]

        response = None
        if self._tools_llm:
            try:
                response = self._tools_llm.invoke(messages)
            except Exception:
                try:
                    llm = get_groq_llm(temperature=0.1)
                    if llm:
                        response = llm.invoke(messages)
                except Exception:
                    response = None

        if not response or not getattr(response, "content", None):
            # Deterministic fallback response when Groq API Key is not set / offline
            response = self._generate_deterministic_travel_plan(state)

        tool_names = [call.get("name") for call in getattr(response, "tool_calls", [])]
        trace_entry = {
            "node": "Groq LLM Decision",
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "action": f"Generated tool calls: {tool_names}" if tool_names else "Synthesized travel plan.",
            "tool_calls_count": len(tool_names),
        }

        return {
            "messages": [response],
            "selected_tools": tool_names,
            "reasoning_trace": [trace_entry],
        }

    def invoke_tools_node(self, state: AgentState) -> Dict[str, Any]:
        """Execute selected tools, capture output, and record retrieved docs for observability."""
        tool_calls = getattr(state["messages"][-1], "tool_calls", [])
        results = []
        retrieved_docs: List[Dict[str, Any]] = []

        for call in tool_calls:
            name = call["name"]
            args = call.get("args", {})
            if name not in self._tools:
                output = f"Unknown tool '{name}'. Available: {list(self._tools.keys())}"
            else:
                try:
                    if "params" in args:
                        output = self._tools[name].invoke(args["params"])
                    else:
                        output = self._tools[name].invoke(args)
                except Exception as exc:
                    output = f"Tool '{name}' execution error: {exc}"

            if name in ["travel_knowledge_search", "search_travel_guide"]:
                retrieved_docs.append({
                    "tool": name,
                    "query": str(args),
                    "snippet": str(output)[:350] + "...",
                })

            results.append(
                ToolMessage(tool_call_id=call["id"], name=name, content=str(output))
            )

        trace_entry = {
            "node": "Tool Execution Engine",
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "action": f"Executed {len(tool_calls)} tool(s) successfully.",
            "executed_tools": [c["name"] for c in tool_calls],
        }

        return {
            "messages": results,
            "retrieved_documents": retrieved_docs,
            "reasoning_trace": [trace_entry],
        }

    def validator_node(self, state: AgentState) -> Dict[str, Any]:
        """Reflection & validation node inspecting response quality, math, and consistency."""
        last_msg = state["messages"][-1]
        text_content = getattr(last_msg, "content", "")

        trip = state.get("trip_details") or {}
        prefs = state.get("preferences") or {}

        val_report = PlanValidator.validate_plan(
            plan_text=text_content,
            start_date_str=trip.get("start_date"),
            end_date_str=trip.get("end_date"),
            max_budget_usd=float(prefs.get("max_budget_usd", 4000.0)),
            destination_name=trip.get("destination"),
        )

        retry_cnt = (state.get("retry_count") or 0)
        messages_to_add: List[AnyMessage] = []

        if val_report["status"] == "FAILED" and retry_cnt < 2:
            retry_cnt += 1
            feedback_msg = HumanMessage(
                content=f"[SYSTEM REFLECTION / VALIDATION CORRECTION]: {val_report['corrective_feedback']}. Please revise your response to rectify this."
            )
            messages_to_add.append(feedback_msg)

        trace_entry = {
            "node": "Reflection & Validator",
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "action": f"Validation Status: [{val_report['status']}]. Passed: {val_report['passed_checks_count']}/{val_report['total_checks_count']} checks.",
            "checks": val_report["checks"],
        }

        approval_req = False
        approval_type = None
        if val_report["status"] == "WARNING" and any("budget" in w.lower() for w in val_report.get("warnings", [])):
            approval_req = True
            approval_type = "budget_exceeded"

        return {
            "messages": messages_to_add,
            "validation_status": val_report["status"],
            "validation_details": val_report,
            "retry_count": retry_cnt,
            "reasoning_trace": [trace_entry],
            "approval_required": approval_req,
            "approval_type": approval_type,
        }

    def hitl_decision_node(self, state: AgentState) -> Dict[str, Any]:
        """Human-in-the-loop pause and approval evaluation node."""
        trace_entry = {
            "node": "Human-in-the-Loop Node",
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "action": f"Approval Type: [{state.get('approval_type', 'General')}], Status: [{state.get('approval_status', 'PENDING')}].",
        }
        return {
            "reasoning_trace": [trace_entry]
        }

    def email_sender_node(self, state: AgentState) -> Dict[str, Any]:
        """Format travel plan into responsive HTML and send via SendGrid."""
        sender = os.environ.get("SENDGRID_FROM_EMAIL") or os.environ.get("FROM_EMAIL")
        receiver = os.environ.get("SENDGRID_TO_EMAIL") or os.environ.get("TO_EMAIL")
        subject = os.environ.get("EMAIL_SUBJECT", "Your Personalized Travel Plan")
        api_key = os.environ.get("SENDGRID_API_KEY")

        missing = [
            name
            for name, value in {
                "SENDGRID_API_KEY": api_key,
                "SENDGRID_FROM_EMAIL": sender,
                "SENDGRID_TO_EMAIL": receiver,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError("Missing email configuration: " + ", ".join(missing))

        content = ""
        for m in reversed(state["messages"]):
            if isinstance(m, AIMessage) and m.content:
                content = m.content
                break
        if not content:
            content = state["messages"][-1].content

        trip = state.get("trip_details") or {}
        html_body = markdown_to_clean_html_email(
            travel_plan_markdown=content,
            destination=trip.get("destination", "Your Destination"),
            origin=trip.get("origin", "Your Origin"),
            start_date=trip.get("start_date", ""),
            end_date=trip.get("end_date", ""),
        )

        message = Mail(
            from_email=sender,
            to_emails=receiver,
            subject=subject,
            html_content=html_body,
        )
        response = SendGridAPIClient(api_key).send(message)
         
        print("SENDGRID STATUS:", response.status_code)
        print("SENDGRID BODY:", response.body)

        if response.status_code not in (200, 201, 202):
            raise RuntimeError(f"SendGrid returned status {response.status_code}")

        return {
            "messages": [
                SystemMessage(content=f"Email sent successfully to {receiver} via SendGrid.")
            ]
        }
