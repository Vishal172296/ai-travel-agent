"""
Unit tests for Agentic Planner, Reflection Validator, Export Generators, and Agent Graph.
"""

import pytest
from langchain_core.messages import HumanMessage, AIMessage
from agents.agent import Agent
from agents.planner import AgenticPlanner
from agents.validator import PlanValidator
from agents.memory import UserTravelPreferences
from export.pdf_generator import generate_pdf_travel_report
from export.email_template import markdown_to_clean_html_email


def test_agent_initialization():
    agent = Agent()
    assert agent.graph is not None
    assert len(agent._tools) >= 7


def test_planner_routing():
    # Simple single-tool query
    weather_q = "What is the weather in Paris in October?"
    w_plan = AgenticPlanner.classify_and_route(weather_q)
    assert w_plan["route_type"] == "SINGLE_TOOL_DIRECT"
    assert "weather_search" in w_plan["recommended_tools"]

    # Complex multi-step query
    complex_q = "Find flights from BOM to DXB on 2026-11-15, 5-star hotels, visa rules, and 5-day itinerary."
    c_plan = AgenticPlanner.classify_and_route(complex_q)
    assert c_plan["route_type"] == "MULTI_TOOL_ORCHESTRATED"
    assert "flights_finder" in c_plan["recommended_tools"]
    assert "hotels_finder" in c_plan["recommended_tools"]
    assert "travel_knowledge_search" in c_plan["recommended_tools"]


def test_planner_node_execution():
    agent = Agent()
    state = {
        "messages": [HumanMessage(content="What is the weather in Dubai in November?")],
        "preferences": {},
        "trip_details": {},
        "human_feedback": None,
        "reasoning_trace": [],
        "selected_tools": [],
        "retrieved_documents": [],
    }
    planner_res = agent.planner_node(state)
    assert "plan" in planner_res
    assert "reasoning_trace" in planner_res
    assert len(planner_res["reasoning_trace"]) >= 1


def test_reflection_validator():
    good_plan = (
        "## Flight Options\n- Emirates EK-501 at $310. [🔗 Book Flight](https://google.com)\n\n"
        "## Hotel Recommendations\n- Atlantis, The Palm at $380/night.\n\n"
        "## Visa Requirements\n- Visa on arrival for Indian passport with valid US visa.\n\n"
        "## Day-by-Day Itinerary\n- Day 1: Morning visit to Burj Khalifa."
    )
    val = PlanValidator.validate_plan(
        plan_text=good_plan,
        start_date_str="2026-11-15",
        end_date_str="2026-11-20",
        destination_name="Dubai",
    )
    assert val["status"] in ["PASSED", "WARNING"]
    assert val["passed_checks_count"] >= 3

    # Test bad date validation
    bad_val = PlanValidator.validate_plan(
        plan_text=good_plan,
        start_date_str="2026-11-20",
        end_date_str="2026-11-15",  # Return before departure
    )
    assert bad_val["status"] == "FAILED"
    assert len(bad_val["errors"]) >= 1


def test_user_preferences_store():
    prefs = UserTravelPreferences()
    prefs.update({"budget_tier": "Luxury ($$$)", "max_budget_usd": 6000.0, "travel_style": "Honeymoon"})
    data = prefs.get_all()
    assert data["max_budget_usd"] == 6000.0
    assert "Luxury" in data["budget_tier"]
    clause = prefs.get_summary_prompt_clause()
    assert "$6,000" in clause or "6000" in clause


def test_pdf_report_generation():
    travel_text = (
        "## ✈️ Flight Options\n- Emirates EK-501 (BOM -> DXB) - $310 [🔗 Book Flight](https://google.com)\n\n"
        "## 🏨 Hotels\n- Rove Downtown Dubai - $145/night\n\n"
        "## 📅 Itinerary\n- Day 1: Downtown Dubai and Burj Khalifa.\n"
    )
    pdf_bytes = generate_pdf_travel_report(
        destination="Dubai",
        origin="Mumbai (BOM)",
        start_date_str="2026-11-15",
        end_date_str="2026-11-20",
        travel_plan_text=travel_text,
        num_travelers=2,
    )
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")


def test_html_email_template():
    travel_text = "## Flights\n- Airline: Emirates at $310 [🔗 Book Flight](https://google.com)\n\n## Hotels\n- Atlantis Palm"
    html_output = markdown_to_clean_html_email(
        travel_plan_markdown=travel_text,
        destination="Dubai",
        origin="Mumbai",
        start_date="2026-11-15",
        end_date="2026-11-20",
    )
    assert "<!DOCTYPE html>" in html_output
    assert "Dubai" in html_output
    assert "href=\"https://google.com\"" in html_output
    assert "Emirates" in html_output


def test_replan_day_directive():
    agent = Agent()
    plan_msg = agent._generate_deterministic_travel_plan({
        "trip_details": {
            "origin": "Mumbai (BOM)",
            "destination": "Dubai",
            "start_date": "2026-11-15",
            "end_date": "2026-11-20",
        },
        "preferences": {},
        "human_feedback": "i want to rest on day 2 in my room so dont plan anything on day 2",
    })
    content = plan_msg.content
    assert "Day 2:" in content
    assert "Rest" in content or "Relaxation" in content or "directive" in content.lower()


def test_trip_extension_directive():
    agent = Agent()
    plan_msg = agent._generate_deterministic_travel_plan({
        "trip_details": {
            "origin": "Mumbai (BOM)",
            "destination": "Dubai",
            "start_date": "2026-10-08",
            "end_date": "2026-10-14",
        },
        "preferences": {},
        "human_feedback": "i want to extend the trip to 10 days",
    })
    content = plan_msg.content
    assert "Day 10:" in content
    assert "Hotel Total (10 Nights)" in content


