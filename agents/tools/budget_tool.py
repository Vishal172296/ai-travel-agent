"""
Budget Optimizer and Rebalancer Tool for Agentic AI Travel Planner.
Analyzes traveler financial constraints, adjusts lodging/dining/flight allocations,
and recommends actionable cost-saving substitutions without degrading experience.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from agents.tools.cache import GLOBAL_TOOL_CACHE


class BudgetOptimizerInput(BaseModel):
    total_budget_usd: float = Field(
        description="User's maximum target total budget for the trip in USD"
    )
    current_estimated_cost_usd: float = Field(
        description="Current total estimated cost of the trip in USD"
    )
    num_travelers: int = Field(
        default=2,
        description="Number of travelers in the party"
    )
    num_nights: int = Field(
        default=5,
        description="Number of nights staying"
    )
    travel_style: Optional[str] = Field(
        default="Moderate",
        description="Travel style preference, e.g. 'Luxury', 'Moderate', 'Backpacker / Budget'"
    )


class BudgetOptimizerInputSchema(BaseModel):
    params: BudgetOptimizerInput


@tool(args_schema=BudgetOptimizerInputSchema)
def budget_optimizer(params: BudgetOptimizerInput) -> Dict[str, Any]:
    """
    Optimize travel expenses when a trip exceeds the desired budget.
    Calculates exact percentage distributions, identifies biggest cost drivers,
    and returns tailored optimization actions for flights, hotels, food, and tours.
    """
    cached = GLOBAL_TOOL_CACHE.get("budget_optimizer", params.dict())
    if cached:
        return cached

    budget = max(params.total_budget_usd, 100.0)
    current = max(params.current_estimated_cost_usd, 100.0)
    diff = current - budget
    is_over_budget = diff > 0
    percent_diff = round((diff / budget) * 100.0, 1)

    # Calculate optimal standard allocations
    flight_alloc = budget * 0.35
    hotel_alloc = budget * 0.30
    food_alloc = budget * 0.20
    activities_alloc = budget * 0.10
    contingency_alloc = budget * 0.05

    max_flight_per_person = round(flight_alloc / max(params.num_travelers, 1), 2)
    max_hotel_per_night = round(hotel_alloc / max(params.num_nights, 1), 2)
    max_food_per_person_day = round((food_alloc / max(params.num_nights, 1)) / max(params.num_travelers, 1), 2)

    suggestions: List[str] = []
    if is_over_budget:
        suggestions.append(f"⚠️ Current plan exceeds target budget by ${diff:,.2f} ({percent_diff}% over).")
        suggestions.append(f"✈️ Target flight cost: Keep under ${max_flight_per_person:,.2f} per traveler.")
        suggestions.append(f"🏨 Target hotel rate: Select boutique or 4-star options under ${max_hotel_per_night:,.2f} per room/night.")
        suggestions.append(f"🍽️ Dining allocation: Target approx ${max_food_per_person_day:,.2f} per person/day for gourmet street food and casual bistros.")
        if percent_diff > 25:
            suggestions.append("💡 Consider shifting travel dates by 1-2 weeks or choosing shoulder season to drop flight & hotel tariffs by 20-35%.")
    else:
        savings = abs(diff)
        suggestions.append(f"✅ Current plan is comfortably within budget with ${savings:,.2f} surplus remaining.")
        suggestions.append(f"✨ You can safely upgrade 1-2 dinners to Michelin-starred dining or book a private VIP guided day tour.")

    result = {
        "status": "OVER_BUDGET" if is_over_budget else "WITHIN_BUDGET",
        "target_budget_usd": f"${budget:,.2f}",
        "current_estimated_usd": f"${current:,.2f}",
        "difference_usd": f"${diff:,.2f}" if is_over_budget else f"-${abs(diff):,.2f} (Savings)",
        "percentage_difference": f"{percent_diff}%",
        "optimized_allocation": {
            "flights_total_max": f"${flight_alloc:,.2f} (${max_flight_per_person:,.2f}/person)",
            "hotels_total_max": f"${hotel_alloc:,.2f} (${max_hotel_per_night:,.2f}/night)",
            "dining_total_max": f"${food_alloc:,.2f} (${max_food_per_person_day:,.2f}/person/day)",
            "activities_tours_max": f"${activities_alloc:,.2f}",
            "emergency_buffer_5pct": f"${contingency_alloc:,.2f}",
        },
        "actionable_recommendations": suggestions,
    }

    GLOBAL_TOOL_CACHE.set("budget_optimizer", params.dict(), result)
    return result
