"""
User Preferences & Conversation Memory Manager for AI Travel Planner.
Provides user-controlled preferences (budget, hotel tier, dietary, travel style)
and persistent session memory storage.
"""

from typing import Any, Dict, Optional


class UserTravelPreferences:
    """User-controlled travel profile and preferences store."""

    def __init__(self):
        self._preferences: Dict[str, Any] = {
            "budget_tier": "Moderate ($$)",
            "max_budget_usd": 3000.0,
            "preferred_currency": "USD",
            "travel_style": "Cultural & Sightseeing",
            "adults": 2,
            "children": 0,
            "rooms": 1,
            "hotel_category": "4-Star",
            "food_preference": "All / Local Gourmet",
            "preferred_activities": ["Sightseeing", "Local Food", "Cultural Landmarks"],
            "special_notes": "",
        }

    def update(self, new_prefs: Dict[str, Any]) -> None:
        """Update preferences with validation."""
        for k, v in new_prefs.items():
            if k in self._preferences and v is not None:
                self._preferences[k] = v

    def get_all(self) -> Dict[str, Any]:
        return self._preferences.copy()

    def get_summary_prompt_clause(self) -> str:
        """Generate a concise prompt fragment summarizing user preferences for the LLM."""
        p = self._preferences
        return (
            f"USER PREFERENCES & CONSTRAINTS:\n"
            f"- Party: {p.get('adults', 2)} adult(s), {p.get('children', 0)} child(ren), {p.get('rooms', 1)} room(s)\n"
            f"- Budget Tier: {p.get('budget_tier', 'Moderate')} (Max: ${p.get('max_budget_usd', 3000):,.0f} {p.get('preferred_currency', 'USD')})\n"
            f"- Travel Style: {p.get('travel_style', 'Cultural')}\n"
            f"- Hotel Category: {p.get('hotel_category', '4-Star')}\n"
            f"- Dietary/Food: {p.get('food_preference', 'Local')}\n"
            f"- Preferred Activities: {', '.join(p.get('preferred_activities', ['Sightseeing']))}\n"
            f"- Special Notes: {p.get('special_notes', 'None')}\n"
        )

    def reset(self) -> None:
        self.__init__()
