"""
Professional Responsive HTML Email Template Builder for SendGrid.
Converts travel plans into polished, mobile-responsive HTML emails
with card structures, highlights, and verified links.
"""

import html
import re
from typing import Optional


def markdown_to_clean_html_email(
    travel_plan_markdown: str,
    destination: str = "Your Destination",
    origin: str = "Origin",
    start_date: str = "",
    end_date: str = "",
) -> str:
    """
    Format travel plan markdown into a mobile-friendly HTML email template.
    """
    # Convert markdown links to HTML anchors
    html_content = travel_plan_markdown
    html_content = re.sub(
        r"\[(.+?)\]\((https?://.+?)\)",
        r'<a href="\2" style="color: #0284c7; font-weight: 600; text-decoration: underline;">\1</a>',
        html_content
    )

    # Convert headers
    lines = html_content.split("\n")
    formatted_body_elements = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("### "):
            header_text = stripped[4:]
            formatted_body_elements.append(
                f'<h3 style="color: #1e3a8a; font-size: 16px; margin-top: 18px; margin-bottom: 8px; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px;">{header_text}</h3>'
            )
        elif stripped.startswith("## ") or stripped.startswith("# "):
            header_text = stripped.lstrip("# ")
            formatted_body_elements.append(
                f'<h2 style="color: #0f172a; font-size: 18px; margin-top: 22px; margin-bottom: 10px; background: #f0f9ff; padding: 8px 12px; border-left: 4px solid #0284c7; border-radius: 4px;">{header_text}</h2>'
            )
        elif stripped.startswith("- ") or stripped.startswith("* ") or stripped.startswith("• "):
            bullet_text = stripped[2:]
            # Bold conversion
            bullet_text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", bullet_text)
            formatted_body_elements.append(
                f'<li style="margin-bottom: 6px; color: #334155; line-height: 1.5;">{bullet_text}</li>'
            )
        else:
            text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", stripped)
            formatted_body_elements.append(
                f'<p style="margin-bottom: 8px; color: #334155; line-height: 1.6;">{text}</p>'
            )

    body_html = "\n".join(formatted_body_elements)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>✈️ Your AI Travel Dossier</title>
</head>
<body style="margin: 0; padding: 0; background-color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background-color: #f8fafc; padding: 30px 10px;">
        <tr>
            <td align="center">
                <table role="presentation" width="100%" style="max-width: 640px; background-color: #ffffff; border-radius: 16px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.06); border: 1px solid #e2e8f0;">
                    <!-- Hero Header -->
                    <tr>
                        <td style="background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%); padding: 32px 28px; text-align: center;">
                            <div style="font-size: 28px; margin-bottom: 4px;">✈️</div>
                            <h1 style="color: #ffffff; font-size: 22px; margin: 0 0 6px 0; font-weight: 800; letter-spacing: -0.5px;">Bespoke Travel Itinerary</h1>
                            <p style="color: #38bdf8; font-size: 14px; margin: 0; font-weight: 600;">{origin} &nbsp;➔&nbsp; {destination}</p>
                            {f'<p style="color: #94a3b8; font-size: 12px; margin: 6px 0 0 0;">📅 {start_date} to {end_date}</p>' if start_date else ''}
                        </td>
                    </tr>

                    <!-- Body Content -->
                    <tr>
                        <td style="padding: 28px; font-size: 14px;">
                            {body_html}
                        </td>
                    </tr>

                    <!-- Footer -->
                    <tr>
                        <td style="background-color: #f1f5f9; padding: 20px; text-align: center; border-top: 1px solid #e2e8f0;">
                            <p style="color: #64748b; font-size: 12px; margin: 0 0 4px 0;">Curated exclusively for you by <strong>AI Travel Concierge</strong></p>
                            <p style="color: #94a3b8; font-size: 11px; margin: 0;">Safe travels! Please re-confirm flight times and visa guidelines before traveling.</p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>"""
    return full_html
