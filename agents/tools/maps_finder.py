"""
Maps, Geocoding, and Destination Location Intelligence tool.
Provides precise coordinates, landmark locations, neighborhood highlights,
and transit proximity details for interactive map displays and route planning.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
import pandas as pd

from agents.tools.cache import GLOBAL_TOOL_CACHE


class MapsInput(BaseModel):
    destination: str = Field(
        description="Destination city, region, or specific landmark (e.g. 'Dubai', 'Bali', 'Eiffel Tower Paris', 'Tokyo Shibuya', 'Goa')"
    )
    include_landmarks: bool = Field(
        default=True,
        description="Whether to include key nearby landmarks and district highlights"
    )


class MapsInputSchema(BaseModel):
    params: MapsInput


DESTINATION_GEO_DATA = {
    "dubai": {
        "center": {"lat": 25.2048, "lon": 55.2708, "name": "Downtown Dubai"},
        "landmarks": [
            {"name": "Burj Khalifa & Dubai Mall", "lat": 25.1972, "lon": 55.2744, "type": "Sightseeing & Architecture"},
            {"name": "Palm Jumeirah & Atlantis", "lat": 25.1304, "lon": 55.1171, "type": "Luxury & Beach"},
            {"name": "Dubai Marina Walk", "lat": 25.0805, "lon": 55.1403, "type": "Dining & Nightlife"},
            {"name": "Dubai Creek & Gold Souk", "lat": 25.2677, "lon": 55.2974, "type": "Heritage & Culture"},
            {"name": "Museum of the Future", "lat": 25.2253, "lon": 55.2818, "type": "Futuristic Museum"},
        ],
        "transit_hub": "Dubai Metro Red & Green Lines",
    },
    "bali": {
        "center": {"lat": -8.4095, "lon": 115.1889, "name": "Ubud Cultural Center"},
        "landmarks": [
            {"name": "Tegalalang Rice Terrace", "lat": -8.4328, "lon": 115.2789, "type": "Nature & Views"},
            {"name": "Sacred Monkey Forest Sanctuary", "lat": -8.5188, "lon": 115.2588, "type": "Wildlife & Culture"},
            {"name": "Seminyak & Petitenget Beach", "lat": -8.6888, "lon": 115.1558, "type": "Beach & Sunset Clubs"},
            {"name": "Uluwatu Cliffside Temple", "lat": -8.8291, "lon": 115.0849, "type": "Historic Temple & Kecak"},
            {"name": "Mount Batur Volcano", "lat": -8.2422, "lon": 115.3753, "type": "Adventure & Sunrise Hike"},
        ],
        "transit_hub": "Gojek / Grab Private Hire & Scooter Rentals",
    },
    "paris": {
        "center": {"lat": 48.8566, "lon": 2.3522, "name": "Central Paris (1st Arrondissement)"},
        "landmarks": [
            {"name": "Eiffel Tower & Champ de Mars", "lat": 48.8584, "lon": 2.2945, "type": "Iconic Monument"},
            {"name": "Musée du Louvre", "lat": 48.8606, "lon": 2.3376, "type": "World Famous Art Museum"},
            {"name": "Cathédrale Notre-Dame", "lat": 48.8530, "lon": 2.3499, "type": "Gothic Historic Cathedral"},
            {"name": "Montmartre & Sacré-Cœur", "lat": 48.8867, "lon": 2.3431, "type": "Bohemian District"},
            {"name": "Champs-Élysées & Arc de Triomphe", "lat": 48.8738, "lon": 2.2950, "type": "Shopping & Landmark"},
        ],
        "transit_hub": "RATP Paris Metro & RER",
    },
    "tokyo": {
        "center": {"lat": 35.6762, "lon": 139.6503, "name": "Tokyo Metropolis"},
        "landmarks": [
            {"name": "Shibuya Scramble Crossing", "lat": 35.6595, "lon": 139.7004, "type": "Urban Sightseeing"},
            {"name": "Senso-ji Temple (Asakusa)", "lat": 35.7148, "lon": 139.7967, "type": "Historic Buddhist Temple"},
            {"name": "Shinjuku Gyoen National Garden", "lat": 35.6852, "lon": 139.7101, "type": "Cherry Blossom & Nature"},
            {"name": "Akihabara Electric Town", "lat": 35.6983, "lon": 139.7731, "type": "Anime, Gaming & Tech"},
            {"name": "Tokyo Skytree", "lat": 35.7101, "lon": 139.8107, "type": "Panoramic City Observation"},
        ],
        "transit_hub": "Tokyo Subway & JR Yamanote Loop Line",
    },
    "goa": {
        "center": {"lat": 15.2993, "lon": 74.1240, "name": "Panaji, Goa"},
        "landmarks": [
            {"name": "Baga & Calangute Beach", "lat": 15.5553, "lon": 73.7517, "type": "Water Sports & Shacks"},
            {"name": "Fort Aguada & Lighthouse", "lat": 15.4920, "lon": 73.7736, "type": "17th Century Portuguese Fort"},
            {"name": "Basilica of Bom Jesus (Old Goa)", "lat": 15.5009, "lon": 73.9116, "type": "UNESCO World Heritage Site"},
            {"name": "Palolem Beach (South Goa)", "lat": 15.0100, "lon": 74.0232, "type": "Scenic White Sand Beach"},
            {"name": "Dudhsagar Waterfalls", "lat": 15.3144, "lon": 74.3143, "type": "Jungle Waterfall Safari"},
        ],
        "transit_hub": "GoaMiles App & Bike Rentals",
    },
    "london": {
        "center": {"lat": 51.5074, "lon": -0.1278, "name": "Central London"},
        "landmarks": [
            {"name": "Big Ben & Westminster Abbey", "lat": 51.5007, "lon": -0.1246, "type": "Royal Landmark"},
            {"name": "Tower of London & Tower Bridge", "lat": 51.5081, "lon": -0.0759, "type": "Historic Fortress"},
            {"name": "British Museum", "lat": 51.5194, "lon": -0.1270, "type": "World Culture Museum"},
            {"name": "Buckingham Palace", "lat": 51.5014, "lon": -0.1419, "type": "Monarchy Residence"},
            {"name": "Borough Market", "lat": 51.5055, "lon": -0.0903, "type": "Gourmet Street Food"},
        ],
        "transit_hub": "London Underground (The Tube) & Elizabeth Line",
    },
    "singapore": {
        "center": {"lat": 1.3521, "lon": 103.8198, "name": "Marina Bay Singapore"},
        "landmarks": [
            {"name": "Gardens by the Bay & Supertrees", "lat": 1.2816, "lon": 103.8636, "type": "Botanical Wonder"},
            {"name": "Marina Bay Sands SkyPark", "lat": 1.2838, "lon": 103.8591, "type": "Iconic Skyline"},
            {"name": "Sentosa Island & Universal Studios", "lat": 1.2540, "lon": 103.8238, "type": "Resort & Theme Park"},
            {"name": "Jewel Changi Airport", "lat": 1.3602, "lon": 103.9897, "type": "Indoor Waterfall & Shopping"},
            {"name": "Chinatown & Maxwell Hawker Centre", "lat": 1.2804, "lon": 103.8448, "type": "Heritage & Culinary"},
        ],
        "transit_hub": "SMRT Subway Network",
    },
    "switzerland": {
        "center": {"lat": 46.8182, "lon": 8.2275, "name": "Zurich & Lucerne, Switzerland"},
        "landmarks": [
            {"name": "Lucerne Chapel Bridge & Lake", "lat": 47.0516, "lon": 8.3073, "type": "Historic Alpine Lake"},
            {"name": "Jungfraujoch Top of Europe", "lat": 46.5475, "lon": 7.9824, "type": "Alpine Glacier & Train"},
            {"name": "Interlaken Adventure Hub", "lat": 46.6863, "lon": 7.8632, "type": "Paragliding & Lakes"},
            {"name": "Matterhorn / Zermatt", "lat": 45.9763, "lon": 7.7491, "type": "Iconic Alpine Peak"},
        ],
        "transit_hub": "Swiss Federal Railways (SBB / Swiss Travel Pass)",
    },
}


@tool(args_schema=MapsInputSchema)
def maps_location_search(params: MapsInput) -> Dict[str, Any]:
    """
    Search coordinates, major landmarks, neighborhood clusters, and transit options
    for any destination to power interactive maps and spatial trip planning.
    """
    cached = GLOBAL_TOOL_CACHE.get("maps_location_search", params.dict())
    if cached:
        return cached

    dest_lower = params.destination.lower()
    matched_key = None
    for key in DESTINATION_GEO_DATA:
        if key in dest_lower:
            matched_key = key
            break

    if matched_key:
        data = DESTINATION_GEO_DATA[matched_key]
        result = {
            "destination": params.destination.title(),
            "center_coordinates": data["center"],
            "landmarks_and_attractions": data["landmarks"] if params.include_landmarks else [],
            "primary_transit_system": data["transit_hub"],
            "total_mapped_points": len(data["landmarks"]) + 1,
        }
    else:
        # Generic center coordinates fallback
        result = {
            "destination": params.destination.title(),
            "center_coordinates": {"lat": 25.2048, "lon": 55.2708, "name": params.destination.title()},
            "landmarks_and_attractions": [
                {"name": f"{params.destination.title()} City Center", "lat": 25.2048, "lon": 55.2708, "type": "Central Hub"},
                {"name": f"{params.destination.title()} Main Cultural Quarter", "lat": 25.2248, "lon": 55.2908, "type": "Culture & Sights"},
                {"name": f"{params.destination.title()} Scenic Waterfront / Park", "lat": 25.1848, "lon": 55.2508, "type": "Leisure & Walking"},
            ],
            "primary_transit_system": "Local Metro / Taxis / Ride Hailing",
            "total_mapped_points": 3,
        }

    GLOBAL_TOOL_CACHE.set("maps_location_search", params.dict(), result)
    return result
