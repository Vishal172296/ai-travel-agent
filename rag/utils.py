"""
Interactive Utilities for AI Travel Agent.
Includes ICS calendar generator, destination mapping data, and packing checklist templates.
"""

from datetime import datetime, timedelta
import pandas as pd


DESTINATION_COORDINATES = {
    "dubai": {"lat": 25.2048, "lon": 55.2708, "name": "Dubai, UAE"},
    "bali": {"lat": -8.4095, "lon": 115.1889, "name": "Bali, Indonesia"},
    "paris": {"lat": 48.8566, "lon": 2.3522, "name": "Paris, France"},
    "tokyo": {"lat": 35.6762, "lon": 139.6503, "name": "Tokyo, Japan"},
    "goa": {"lat": 15.2993, "lon": 74.1240, "name": "Goa, India"},
    "london": {"lat": 51.5074, "lon": -0.1278, "name": "London, UK"},
    "singapore": {"lat": 1.3521, "lon": 103.8198, "name": "Singapore"},
    "new york": {"lat": 40.7128, "lon": -74.0060, "name": "New York, USA"},
    "rome": {"lat": 41.9028, "lon": 12.4964, "name": "Rome, Italy"},
    "bangkok": {"lat": 13.7563, "lon": 100.5018, "name": "Bangkok, Thailand"},
    "switzerland": {"lat": 46.8182, "lon": 8.2275, "name": "Zurich, Switzerland"},
}


def get_map_dataframe(destination_query: str) -> pd.DataFrame:
    """Find matching coordinates or default to global hubs for st.map."""
    q = destination_query.lower()
    matched = None
    for key, val in DESTINATION_COORDINATES.items():
        if key in q:
            matched = val
            break

    if not matched:
        matched = {"lat": 25.2048, "lon": 55.2708, "name": destination_query}

    # Add key landmark spots around the destination for an interactive map visual
    lat = matched["lat"]
    lon = matched["lon"]
    data = [
        {"lat": lat, "lon": lon, "name": f"📍 {matched['name']} City Center"},
        {"lat": lat + 0.02, "lon": lon + 0.02, "name": "🏨 Recommended Hotel Area"},
        {"lat": lat - 0.025, "lon": lon - 0.015, "name": "🏛️ Main Historic / Cultural Attraction"},
        {"lat": lat + 0.035, "lon": lon - 0.025, "name": "🍽️ Famous Food & Market District"},
    ]
    return pd.DataFrame(data)


def generate_ics_calendar(
    trip_title: str,
    start_date_str: str,
    end_date_str: str,
    description: str,
    location: str
) -> str:
    """Generate iCalendar (.ics) format file content for the trip."""
    try:
        start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
    except Exception:
        start_dt = datetime.now() + timedelta(days=7)
        end_dt = start_dt + timedelta(days=5)

    s_stamp = start_dt.strftime("%Y%m%d")
    e_stamp = end_dt.strftime("%Y%m%d")
    now_stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    clean_desc = description.replace("\n", "\\n").replace(",", "\\,")

    ics_content = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//AI Travel Agent//EN
CALSCALE:GREGORIAN
METHOD:PUBLISH
BEGIN:VEVENT
UID:trip-{s_stamp}-{now_stamp}@aitravelagent
DTSTAMP:{now_stamp}
DTSTART;VALUE=DATE:{s_stamp}
DTEND;VALUE=DATE:{e_stamp}
SUMMARY:✈️ {trip_title}
DESCRIPTION:{clean_desc[:400]}...
LOCATION:{location}
STATUS:CONFIRMED
BEGIN:VALARM
TRIGGER:-P1D
DESCRIPTION:Reminder: Your trip to {location} starts tomorrow!
ACTION:DISPLAY
END:VALARM
END:VEVENT
END:VCALENDAR"""
    return ics_content


def get_packing_checklist(destination: str, trip_style: str) -> dict:
    """Generate tailored packing checklist categories."""
    d = destination.lower()
    is_beach = "beach" in trip_style.lower() or any(b in d for b in ["bali", "goa", "maldives", "phuket", "dubai"])
    is_cold = any(c in d for c in ["switzerland", "london", "paris", "tokyo", "new york", "winter"])

    clothing = [
        "4-5 Light breathable shirts / t-shirts",
        "2-3 Pairs of comfortable pants / trousers",
        "Comfortable walking shoes / sneakers (15k+ steps/day)",
        "Undergarments & sleepwear",
    ]
    if is_beach:
        clothing.extend(["Swimwear / board shorts", "UV Sun protection rashguard", "Flip-flops / beach sandals", "Polarized sunglasses", "Sun hat / cap"])
    if is_cold:
        clothing.extend(["Insulated thermal innerwear", "Warm fleece jacket / down coat", "Woolen beanie & gloves", "Waterproof boots"])

    documents = [
        "Original Passport (valid for 6+ months from travel date)",
        "Printed Visa approval / eVisa copy",
        "Confirmed Flight E-tickets & Hotel vouchers",
        "Travel Insurance policy document",
        "International Driving Permit (IDP) if renting bikes/cars",
        "2x Passport size photos (for emergency visas/SIMs)",
    ]

    electronics = [
        "Universal International Travel Adapter",
        "Power Bank (Max 20,000 mAh - Keep in cabin bag!)",
        "Phone charger & high-speed braided USB-C cables",
        "Noise-cancelling headphones / Earbuds",
        "E-SIM app installed (Airalo / Nomad / Holafly)",
    ]

    toiletries_meds = [
        "Sunscreen (SPF 50+ broad spectrum, Reef-safe)",
        "Prescription medications with doctor's prescription note",
        "Travel First-Aid: Paracetamol, Ibuprofen, Band-aids",
        "ORS hydration packets & Antidiarrheal (Loperamide)",
        "Motion sickness tablets (for flights / boat rides)",
        "Mosquito repellent cream or spray (DEET based)",
    ]

    return {
        "👕 Clothing & Footwear": clothing,
        "🛂 Essential Documents": documents,
        "🔌 Electronics & Gadgets": electronics,
        "💊 Health & Toiletries": toiletries_meds,
    }
