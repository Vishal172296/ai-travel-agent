"""
Agentic Multi-Step Planner and Intelligent Tool Router for AI Travel Planner.
Analyzes traveler intent, decomposes complex requests into an actionable execution graph,
and performs intelligent routing between simple single-tool queries and full multi-tool orchestrations.
"""

import re
from typing import Any, Dict, List, Optional, Tuple


class AgenticPlanner:
    """Deconstructs user queries into structured execution plans and tool routing decisions."""

    @staticmethod
    def classify_and_route(query: str) -> Dict[str, Any]:
        """
        Classify query intent and determine optimal tool execution strategy.
        """
        q = query.lower()

        is_complex = any(p in q for p in ["flight", "hotel", "itinerary", "plan my trip", "schedule", "day 1", "day-by-day", "vacation plan"])

        # Simple Query Pattern Detection (when not a full trip planning query)
        is_weather_only = any(w in q for w in ["weather", "temperature", "forecast", "climate", "rainfall"]) and not is_complex
        is_currency_only = any(c in q for c in ["convert", "exchange rate", "forex", "how much is", "currency"]) and not is_complex
        is_rag_only = any(r in q for r in ["visa", "baggage", "luggage", "liquid", "passport", "levy", "scam", "safety", "customs", "culture"]) and not is_complex
        is_maps_only = any(m in q for m in ["where is", "coordinates", "landmark", "map of", "location of"]) and not is_complex

        if is_weather_only:
            return {
                "route_type": "SINGLE_TOOL_DIRECT",
                "recommended_tools": ["weather_search"],
                "steps": ["1. Query live weather and seasonal climate forecast."],
            }
        elif is_currency_only:
            return {
                "route_type": "SINGLE_TOOL_DIRECT",
                "recommended_tools": ["currency_conversion"],
                "steps": ["1. Calculate real-time currency conversion."],
            }
        elif is_rag_only:
            return {
                "route_type": "SINGLE_TOOL_DIRECT",
                "recommended_tools": ["travel_knowledge_search"],
                "steps": ["1. Search verified travel knowledge base for rules, visas, and tips."],
            }
        elif is_maps_only:
            return {
                "route_type": "SINGLE_TOOL_DIRECT",
                "recommended_tools": ["maps_location_search"],
                "steps": ["1. Search destination coordinates and top landmark clusters."],
            }

        # Complex Multi-Step Planning Workflow
        return {
            "route_type": "MULTI_TOOL_ORCHESTRATED",
            "recommended_tools": [
                "flights_finder",
                "hotels_finder",
                "travel_knowledge_search",
                "weather_search",
                "maps_location_search",
                "calculate_trip_budget",
            ],
            "steps": [
                "1. Search verified flights and pricing matching traveler schedule and party size.",
                "2. Discover top-rated hotels matching category, location, and amenities.",
                "3. Query verified RAG knowledge for destination visa rules, transit hacks, and cultural guidelines.",
                "4. Retrieve seasonal weather advisory and clothing recommendations.",
                "5. Extract landmark coordinates and neighborhood clusters for spatial layout.",
                "6. Calculate comprehensive multi-currency trip budget and itemized totals.",
                "7. Synthesize complete bespoke day-by-day luxury travel dossier.",
            ],
        }

    @staticmethod
    def generate_planning_system_prompt(
        preferences_clause: str = "",
        feedback_clause: Optional[str] = None
    ) -> str:
        """Construct the dynamic system prompt with user preferences and feedback adjustments."""
        feedback_prompt = ""
        if feedback_clause:
            feedback_prompt = f"""
IMPORTANT HUMAN FEEDBACK / MODIFICATION DIRECTIVE:
The traveler has provided the following revision instructions:
>>> "{feedback_clause}" <<<
You MUST incorporate this feedback directly: replan, adjust prices, change dates, select alternate stays, or rebalance activities as requested.
"""

        prompt = f"""You are an elite, highly intelligent Agentic AI Travel Concierge.

{preferences_clause}
{feedback_prompt}

YOUR OBJECTIVE:
Craft an extraordinary, highly practical, verified, and complete travel plan for the traveler.

TOOL USAGE PROTOCOL:
You have access to 7 powerful specialized tools:
1. `flights_finder`: Find flights, schedules, durations, stops, and direct Google Flights booking links.
2. `hotels_finder`: Find hotels, star ratings, nightly/total rates, amenities, and booking links.
3. `travel_knowledge_search`: Retrieve verified destination knowledge (RAG), including visa regulations, entry rules, cultural etiquette, scams, and packing advice.
4. `weather_search`: Get live temperatures, rain probability, and weather-appropriate packing tips.
5. `currency_conversion`: Convert monetary amounts between global currencies.
6. `calculate_trip_budget`: Calculate itemized totals across flights, hotel, dining, and activities with contingency buffers.
7. `maps_location_search`: Get landmark coordinates, neighborhood clusters, and transit options.
8. `budget_optimizer`: Rebalance travel expenses when exceeding target budgets.

EXECUTION INSTRUCTIONS:
- Always invoke tools relevant to the request. For full travel inquiries, combine flights, hotels, RAG knowledge, weather, and budget calculations.
- Display flight and hotel prices clearly in USD and the user's preferred currency.
- Clickable markdown links are MANDATORY:
  * Flights: [🔗 Book Flight](BOOKING_URL)
  * Hotels: [🏨 View Hotel](HOTEL_URL)
- Response Structure Requirements:
  1. ✈️ **Flight Options & Booking**: Clear table or bullets with airline, flight number, departure/arrival times, price, and booking link.
  2. 🏨 **Hotel Recommendations**: Selected hotels with star rating, price per night, total price, amenities, and location link.
  3. 🛂 **Visa & Entry Requirements**: Grounded in verified knowledge base.
  4. ☀️ **Best Season & Weather Advice**: Expected temperatures, rain probability, and packing advice.
  5. 🚇 **Local Transportation & Cultural Hacks**: Metro/cab tips, dos & don'ts, scams to avoid.
  6. 📅 **Day-by-Day Detailed Itinerary**: Morning, Afternoon, and Evening activities tailored to preferences.
  7. 💰 **Estimated Trip Budget & Investment**: Clear itemized breakdown and grand total.
  8. 📚 **Verified Grounding Sources**: Clean source citations from the knowledge base.
"""
        return prompt
