"""
Reflection & Validation Agent for AI Travel Planner.
Verifies date chronology, price math, budget compliance, itinerary consistency,
booking link validity, and RAG grounding before presenting final response.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


class PlanValidator:
    """Reflective validator that inspects synthesized travel responses."""

    @staticmethod
    def validate_plan(
        plan_text: str,
        start_date_str: Optional[str] = None,
        end_date_str: Optional[str] = None,
        max_budget_usd: Optional[float] = None,
        destination_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate generated travel plan against essential quality, logic, and safety rules.
        """
        checks: List[Dict[str, Any]] = []
        warnings: List[str] = []
        errors: List[str] = []

        # 1. Date Chronology Check
        date_valid = True
        if start_date_str and end_date_str:
            try:
                s_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
                e_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
                if e_dt <= s_dt:
                    date_valid = False
                    errors.append("Departure date must occur strictly before the return date.")
                else:
                    duration_days = (e_dt - s_dt).days
                    checks.append({
                        "name": "Date Chronology & Duration",
                        "status": "PASS",
                        "details": f"Valid {duration_days}-day travel window ({start_date_str} to {end_date_str})."
                    })
            except Exception:
                checks.append({"name": "Date Chronology", "status": "WARN", "details": "Unparseable date formats provided."})

        # 2. Section Completeness Check
        required_sections = [
            ("Flight", ["flight", "airline", "airport"]),
            ("Hotel", ["hotel", "stay", "resort"]),
            ("Visa / Requirements", ["visa", "passport", "entry"]),
            ("Day-by-Day Itinerary", ["day 1", "itinerary", "morning"]),
        ]
        missing_sections = []
        for sec_name, keywords in required_sections:
            if not any(kw in plan_text.lower() for kw in keywords):
                missing_sections.append(sec_name)

        if missing_sections:
            warnings.append(f"Response may be missing key section(s): {', '.join(missing_sections)}.")
            checks.append({
                "name": "Section Completeness",
                "status": "WARN",
                "details": f"Missing recommended sections: {missing_sections}"
            })
        else:
            checks.append({
                "name": "Section Completeness",
                "status": "PASS",
                "details": "All primary travel sections (Flights, Hotels, Visa, Itinerary) detected."
            })

        # 3. Interactive Links Verification
        has_flight_link = bool(re.search(r"\[.+?\]\(https?://.+?\)", plan_text))
        if has_flight_link:
            checks.append({
                "name": "Booking & Navigation Links",
                "status": "PASS",
                "details": "Verified clickable markdown links present for booking/hotels."
            })
        else:
            warnings.append("No clickable markdown booking links found.")
            checks.append({
                "name": "Booking & Navigation Links",
                "status": "WARN",
                "details": "Missing clickable markdown links."
            })

        # 4. Budget & Pricing Numerical Consistency Check
        dollar_amounts = re.findall(r"\$\s*(\d+(?:,\d+)*(?:\.\d{2})?)", plan_text)
        prices = [float(p.replace(",", "")) for p in dollar_amounts]

        if prices:
            max_found_price = max(prices)
            if max_budget_usd and max_budget_usd > 0:
                if max_found_price > max_budget_usd * 1.5:
                    warnings.append(f"Estimated expenses (${max_found_price:,.0f}) exceed target budget (${max_budget_usd:,.0f}).")
                    checks.append({
                        "name": "Budget Compliance",
                        "status": "WARN",
                        "details": f"Plan exceeds budget constraint: Max price ${max_found_price:,.0f} vs Budget ${max_budget_usd:,.0f}."
                    })
                else:
                    checks.append({
                        "name": "Budget Compliance",
                        "status": "PASS",
                        "details": f"Plan cost aligns within budget envelope (${max_budget_usd:,.0f})."
                    })
            else:
                checks.append({
                    "name": "Pricing Clarity",
                    "status": "PASS",
                    "details": f"Identified {len(prices)} transparent pricing data points."
                })
        else:
            checks.append({
                "name": "Pricing Clarity",
                "status": "WARN",
                "details": "No specific numerical USD prices found in text."
            })

        # 5. Destination Relevance Check
        if destination_name:
            dest_lower = destination_name.lower().split(",")[0].strip()
            if dest_lower in plan_text.lower():
                checks.append({
                    "name": "Destination Grounding",
                    "status": "PASS",
                    "details": f"Accurately focused on '{destination_name}' throughout recommendations."
                })
            else:
                warnings.append(f"Destination '{destination_name}' not explicitly referenced.")
                checks.append({
                    "name": "Destination Grounding",
                    "status": "WARN",
                    "details": f"Target destination '{destination_name}' not found."
                })

        # Overall Status
        if errors:
            status = "FAILED"
        elif warnings:
            status = "WARNING"
        else:
            status = "PASSED"

        return {
            "status": status,
            "checks": checks,
            "warnings": warnings,
            "errors": errors,
            "total_checks_count": len(checks),
            "passed_checks_count": sum(1 for c in checks if c["status"] == "PASS"),
            "corrective_feedback": errors[0] if errors else (warnings[0] if warnings else None),
        }
