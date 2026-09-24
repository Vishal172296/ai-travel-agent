"""
Currency conversion and trip expenditure calculator tool.
Provides real-time forex rates, conversion between major currencies,
and automatic total trip cost estimation across flight, lodging, dining, and activities.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
import urllib.request
import json

from agents.tools.cache import GLOBAL_TOOL_CACHE


class CurrencyConversionInput(BaseModel):
    amount: float = Field(description="The monetary amount to convert")
    from_currency: str = Field(description="Base currency 3-letter ISO code, e.g. USD, EUR, INR, AED, GBP, JPY, SGD")
    to_currency: str = Field(description="Target currency 3-letter ISO code, e.g. USD, EUR, INR, AED, GBP, JPY, SGD")


class CurrencyConversionInputSchema(BaseModel):
    params: CurrencyConversionInput


class TripCostCalculationInput(BaseModel):
    flights_cost_per_person: float = Field(description="Flight cost per person in USD")
    hotel_cost_per_night: float = Field(description="Hotel cost per room night in USD")
    num_travelers: int = Field(default=1, description="Number of travelers")
    num_nights: int = Field(default=5, description="Number of nights")
    daily_food_per_person: float = Field(default=50.0, description="Estimated daily food/dining budget per person in USD")
    daily_activities_per_person: float = Field(default=40.0, description="Estimated daily activities & tours per person in USD")
    target_currency: str = Field(default="USD", description="Target currency for total output (e.g. USD, EUR, INR, AED, GBP)")


class TripCostCalculationInputSchema(BaseModel):
    params: TripCostCalculationInput


# Verified baseline exchange rates relative to 1 USD
BASE_FOREX_RATES = {
    "USD": 1.0,
    "EUR": 0.92,
    "GBP": 0.79,
    "INR": 86.8,
    "AED": 3.6725,
    "JPY": 152.5,
    "SGD": 1.34,
    "CHF": 0.89,
    "AUD": 1.54,
    "CAD": 1.38,
    "THB": 35.8,
    "IDR": 15850.0,
    "QAR": 3.64,
    "SAR": 3.75,
}

CURRENCY_SYMBOLS = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "INR": "₹",
    "AED": "AED ",
    "JPY": "¥",
    "SGD": "S$",
    "CHF": "CHF ",
    "AUD": "A$",
    "CAD": "C$",
    "THB": "฿",
    "IDR": "IDR ",
}


def _fetch_live_rate(from_curr: str, to_curr: str) -> Optional[float]:
    """Attempt to fetch live rates from free open exchange API with fast timeout."""
    try:
        url = f"https://open.er-api.com/v6/latest/{from_curr.upper()}"
        req = urllib.request.Request(url, headers={"User-Agent": "AITravelPlanner/2.0"})
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            data = json.loads(resp.read().decode())
            rates = data.get("rates", {})
            return rates.get(to_curr.upper())
    except Exception:
        return None


def get_conversion_rate(from_curr: str, to_curr: str) -> float:
    """Calculate conversion rate between two currencies with live check and fallback."""
    from_c = from_curr.upper().strip()
    to_c = to_curr.upper().strip()
    if from_c == to_c:
        return 1.0

    live = _fetch_live_rate(from_c, to_c)
    if live:
        return float(live)

    # Fallback to base table
    rate_from = BASE_FOREX_RATES.get(from_c, 1.0)
    rate_to = BASE_FOREX_RATES.get(to_c, 1.0)
    # Both relative to USD
    usd_val = 1.0 / rate_from
    return usd_val * rate_to


@tool(args_schema=CurrencyConversionInputSchema)
def currency_conversion(params: CurrencyConversionInput) -> Dict[str, Any]:
    """
    Convert amounts between global currencies (USD, EUR, GBP, INR, AED, JPY, SGD, etc.)
    with up-to-date forex calculations.
    """
    cached = GLOBAL_TOOL_CACHE.get("currency_conversion", params.dict())
    if cached:
        return cached

    rate = get_conversion_rate(params.from_currency, params.to_currency)
    converted = round(params.amount * rate, 2)
    to_sym = CURRENCY_SYMBOLS.get(params.to_currency.upper(), params.to_currency.upper() + " ")
    from_sym = CURRENCY_SYMBOLS.get(params.from_currency.upper(), params.from_currency.upper() + " ")

    result = {
        "original_amount": f"{from_sym}{params.amount:,.2f} ({params.from_currency.upper()})",
        "converted_amount": f"{to_sym}{converted:,.2f} ({params.to_currency.upper()})",
        "exchange_rate": f"1 {params.from_currency.upper()} = {rate:.4f} {params.to_currency.upper()}",
        "raw_converted_value": converted,
    }
    GLOBAL_TOOL_CACHE.set("currency_conversion", params.dict(), result)
    return result


@tool(args_schema=TripCostCalculationInputSchema)
def calculate_trip_budget(params: TripCostCalculationInput) -> Dict[str, Any]:
    """
    Calculate full itemized trip budget including flights, hotel nights, daily dining,
    sightseeing tours, and contingency buffers with multi-currency conversion.
    """
    total_flights_usd = params.flights_cost_per_person * params.num_travelers
    total_hotels_usd = params.hotel_cost_per_night * params.num_nights
    total_food_usd = params.daily_food_per_person * params.num_nights * params.num_travelers
    total_activities_usd = params.daily_activities_per_person * params.num_nights * params.num_travelers
    subtotal_usd = total_flights_usd + total_hotels_usd + total_food_usd + total_activities_usd
    contingency_usd = subtotal_usd * 0.10  # 10% safety buffer
    grand_total_usd = subtotal_usd + contingency_usd

    rate = get_conversion_rate("USD", params.target_currency)
    sym = CURRENCY_SYMBOLS.get(params.target_currency.upper(), params.target_currency.upper() + " ")

    return {
        "num_travelers": params.num_travelers,
        "trip_duration_nights": params.num_nights,
        "target_currency": params.target_currency.upper(),
        "breakdown": {
            "flights_total": f"{sym}{total_flights_usd * rate:,.2f}",
            "hotels_total": f"{sym}{total_hotels_usd * rate:,.2f}",
            "food_and_dining": f"{sym}{total_food_usd * rate:,.2f}",
            "activities_and_tours": f"{sym}{total_activities_usd * rate:,.2f}",
            "recommended_contingency_10pct": f"{sym}{contingency_usd * rate:,.2f}",
        },
        "grand_total": f"{sym}{grand_total_usd * rate:,.2f}",
        "cost_per_traveler": f"{sym}{(grand_total_usd / params.num_travelers) * rate:,.2f}",
        "grand_total_usd_numeric": round(grand_total_usd, 2),
    }
