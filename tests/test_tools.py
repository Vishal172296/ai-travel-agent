"""
Unit tests for the expanded Tool Ecosystem:
Flights, Hotels, RAG tool, Weather, Currency, Maps, Budget Optimizer, and Tool Cache.
"""

import pytest
from agents.tools.flights_finder import flights_finder, FlightsInput
from agents.tools.hotels_finder import hotels_finder, HotelsInput
from agents.tools.rag_tool import travel_knowledge_search, search_travel_guide, TravelGuideInput
from agents.tools.weather_finder import weather_search, WeatherInput
from agents.tools.currency_tool import currency_conversion, calculate_trip_budget, CurrencyConversionInput, TripCostCalculationInput
from agents.tools.maps_finder import maps_location_search, MapsInput
from agents.tools.budget_tool import budget_optimizer, BudgetOptimizerInput
from agents.tools.cache import SimpleToolCache, GLOBAL_TOOL_CACHE


def test_tool_cache():
    cache = SimpleToolCache(default_ttl_seconds=60)
    args = {"query": "paris", "month": "may"}
    assert cache.get("test_tool", args) is None

    cache.set("test_tool", args, {"result": "ok"})
    assert cache.get("test_tool", args) == {"result": "ok"}


def test_flights_finder_fallback():
    params = FlightsInput(
        departure_airport="BOM",
        arrival_airport="DXB",
        outbound_date="2026-11-15",
        return_date="2026-11-20",
        adults=2,
    )
    result = flights_finder.invoke({"params": params.dict()})
    assert isinstance(result, list)
    assert len(result) >= 1
    assert "airline" in result[0]
    assert "price_usd" in result[0]
    assert "booking_link" in result[0]
    assert "google.com/travel/flights" in result[0]["booking_link"]


def test_hotels_finder_fallback():
    params = HotelsInput(
        location="Dubai",
        check_in_date="2026-11-15",
        check_out_date="2026-11-20",
        adults=2,
        hotel_class="5",
    )
    result = hotels_finder.invoke({"params": params.dict()})
    assert isinstance(result, list)
    assert len(result) >= 1
    assert "name" in result[0]
    assert "rating" in result[0]
    assert "price_per_night" in result[0]


def test_rag_tool():
    params = TravelGuideInput(
        query="Dubai visa rules Indian passport",
        destination="Dubai",
    )
    result = travel_knowledge_search.invoke({"params": params.dict()})
    assert isinstance(result, str)
    assert "SOURCE:" in result
    assert "Dubai" in result or "Visa" in result

    # Test backward compatible alias
    alias_res = search_travel_guide.invoke({"params": params.dict()})
    assert isinstance(alias_res, str)
    assert "SOURCE:" in alias_res


def test_weather_finder():
    params = WeatherInput(
        destination="Paris",
        month="May",
    )
    result = weather_search.invoke({"params": params.dict()})
    assert isinstance(result, dict)
    assert "expected_temperature_range" in result
    assert "general_conditions" in result
    assert "clothing_and_packing_advice" in result


def test_currency_and_budget_tool():
    # Forex conversion
    c_params = CurrencyConversionInput(
        amount=1000.0,
        from_currency="USD",
        to_currency="AED",
    )
    c_res = currency_conversion.invoke({"params": c_params.dict()})
    assert isinstance(c_res, dict)
    assert "exchange_rate" in c_res
    assert c_res["raw_converted_value"] > 3000

    # Trip Cost Calculation
    b_params = TripCostCalculationInput(
        flights_cost_per_person=350.0,
        hotel_cost_per_night=180.0,
        num_travelers=2,
        num_nights=5,
        target_currency="USD",
    )
    b_res = calculate_trip_budget.invoke({"params": b_params.dict()})
    assert isinstance(b_res, dict)
    assert "grand_total" in b_res
    assert "breakdown" in b_res
    assert b_res["grand_total_usd_numeric"] > 1500


def test_maps_location_search():
    params = MapsInput(destination="Tokyo")
    res = maps_location_search.invoke({"params": params.dict()})
    assert isinstance(res, dict)
    assert "center_coordinates" in res
    assert "landmarks_and_attractions" in res
    assert len(res["landmarks_and_attractions"]) >= 1


def test_budget_optimizer():
    params = BudgetOptimizerInput(
        total_budget_usd=2000.0,
        current_estimated_cost_usd=2800.0,
        num_travelers=2,
        num_nights=5,
    )
    res = budget_optimizer.invoke({"params": params.dict()})
    assert isinstance(res, dict)
    assert res["status"] == "OVER_BUDGET"
    assert "optimized_allocation" in res
    assert len(res["actionable_recommendations"]) >= 1
