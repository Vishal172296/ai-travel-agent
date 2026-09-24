"""
Luxury PDF Travel Report Generator using ReportLab.
Generates comprehensive travel dossiers with flights, hotel options,
day-by-day itinerary, itemized budget breakdown, weather advisories, and source citations.
"""

import io
import re
from datetime import datetime
from typing import Any, Dict, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)


def generate_pdf_travel_report(
    destination: str,
    origin: str,
    start_date_str: str,
    end_date_str: str,
    travel_plan_text: str,
    num_travelers: int = 2,
    budget_estimate_str: Optional[str] = None,
) -> bytes:
    """
    Generate professional binary PDF travel dossier.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#0284c7"),
        spaceAfter=12,
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        "BulletCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
        leftIndent=12,
        spaceAfter=4,
    )

    story = []

    # Document Header Banner
    header_data = [
        [
            Paragraph(f"<b>AI TRAVEL CONCIERGE DOSSIER</b>", title_style),
            Paragraph(f"Generated: {datetime.now().strftime('%b %d, %Y')}", subtitle_style),
        ]
    ]
    header_table = Table(header_data, colWidths=[360, 170])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=10))

    # Journey Summary Box
    summary_data = [
        [
            Paragraph("<b>Origin:</b>", body_style),
            Paragraph(origin, body_style),
            Paragraph("<b>Destination:</b>", body_style),
            Paragraph(destination, body_style),
        ],
        [
            Paragraph("<b>Dates:</b>", body_style),
            Paragraph(f"{start_date_str} to {end_date_str}", body_style),
            Paragraph("<b>Travelers:</b>", body_style),
            Paragraph(f"{num_travelers} Guest(s)", body_style),
        ],
    ]
    summary_table = Table(summary_data, colWidths=[70, 190, 80, 190])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f0f9ff")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#bae6fd")),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 12))

    # Parse and structure markdown content into PDF sections
    raw_lines = travel_plan_text.split("\n")
    current_section = None

    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line:
            continue

        # Clean markdown links for clean PDF printing
        clean_line = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1 (\2)", line)
        clean_line = re.sub(r"[*_#]", "", clean_line)

        # Detect headers
        if raw_line.startswith("#") or any(h in raw_line for h in ["Flight Options", "Hotel Recommendations", "Visa", "Weather", "Itinerary", "Budget"]):
            story.append(Paragraph(clean_line, section_heading))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=6))
        elif line.startswith("-") or line.startswith("•") or (len(line) > 2 and line[0].isdigit() and line[1] in [".", ")"]):
            story.append(Paragraph(f"• {clean_line.lstrip('-•0123456789. ')}", bullet_style))
        else:
            story.append(Paragraph(clean_line, body_style))

    # Footer
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0284c7"), spaceAfter=6))
    footer_text = Paragraph(
        "<i>This bespoke travel dossier was generated by AI Travel Concierge. Please verify all flight and visa details before departure.</i>",
        subtitle_style
    )
    story.append(footer_text)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
