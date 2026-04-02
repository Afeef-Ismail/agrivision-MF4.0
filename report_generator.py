# report_generator.py
# Generates downloadable TXT and PDF crop disease reports from AgriVision prediction data.

import os
import requests
from datetime import datetime
from html import escape as html_escape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---------------------------------------------------------------------------
# Noto font setup — one font per Indian script.
# NotoSans-Regular only covers Latin/Greek/Cyrillic; every Indic script
# needs its own dedicated Noto font file.
# Fonts are downloaded once on first use, cached in fonts/, then reused.
# ---------------------------------------------------------------------------
FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
os.makedirs(FONT_DIR, exist_ok=True)

NOTO_FONTS = {
    "NotoSans":           "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf",
    "NotoSans-Bold":      "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Bold.ttf",
    "NotoSansDevanagari": "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf",
    "NotoSansBengali":    "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansBengali/NotoSansBengali-Regular.ttf",
    "NotoSansGurmukhi":   "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansGurmukhi/NotoSansGurmukhi-Regular.ttf",
    "NotoSansGujarati":   "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansGujarati/NotoSansGujarati-Regular.ttf",
    "NotoSansOriya":      "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansOriya/NotoSansOriya-Regular.ttf",
    "NotoSansTamil":      "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansTamil/NotoSansTamil-Regular.ttf",
    "NotoSansTelugu":     "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansTelugu/NotoSansTelugu-Regular.ttf",
    "NotoSansKannada":    "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansKannada/NotoSansKannada-Regular.ttf",
    "NotoSansMalayalam":  "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansMalayalam/NotoSansMalayalam-Regular.ttf",
}

# Unicode block start/end → font name for each Indic script
_SCRIPT_RANGES = [
    (0x0900, 0x097F, "NotoSansDevanagari"),  # Hindi, Marathi
    (0x0980, 0x09FF, "NotoSansBengali"),
    (0x0A00, 0x0A7F, "NotoSansGurmukhi"),    # Punjabi
    (0x0A80, 0x0AFF, "NotoSansGujarati"),
    (0x0B00, 0x0B7F, "NotoSansOriya"),       # Odia
    (0x0B80, 0x0BFF, "NotoSansTamil"),
    (0x0C00, 0x0C7F, "NotoSansTelugu"),
    (0x0C80, 0x0CFF, "NotoSansKannada"),
    (0x0D00, 0x0D7F, "NotoSansMalayalam"),
]

# Download any missing fonts silently
for _name, _url in NOTO_FONTS.items():
    _path = os.path.join(FONT_DIR, f"{_name}.ttf")
    if not os.path.exists(_path):
        try:
            _r = requests.get(_url, timeout=15)
            _r.raise_for_status()
            with open(_path, "wb") as _f:
                _f.write(_r.content)
        except Exception:
            pass  # falls back to Helvetica if unavailable


def _register_fonts():
    """Registers all downloaded Noto fonts with ReportLab."""
    for font_name in NOTO_FONTS:
        font_path = os.path.join(FONT_DIR, f"{font_name}.ttf")
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont(font_name, font_path))
            except Exception:
                pass
    # Register family so <b> tags inside Paragraphs resolve to NotoSans-Bold
    try:
        pdfmetrics.registerFontFamily(
            "NotoSans",
            normal="NotoSans",
            bold="NotoSans-Bold",
            italic="NotoSans",
            boldItalic="NotoSans-Bold",
        )
    except Exception:
        pass


_register_fonts()


def _font_for(text: str) -> str:
    """
    Returns the correct registered Noto font for the dominant script in text.
    Scans characters until it finds one in an Indic Unicode block, then
    returns that script's font — but only if it was successfully downloaded
    and registered. Falls back to NotoSans (Latin) otherwise.
    """
    for char in text:
        cp = ord(char)
        for start, end, font_name in _SCRIPT_RANGES:
            if start <= cp <= end:
                try:
                    pdfmetrics.getFont(font_name)
                    return font_name
                except Exception:
                    return "NotoSans"
    return "NotoSans"


def _safe(text: str) -> str:
    """HTML-escapes text so it is safe inside ReportLab XML Paragraph markup."""
    return html_escape(str(text), quote=False)


def _wrap(text: str) -> str:
    """
    Returns ReportLab XML for the text, wrapping in <font> tag when the text
    uses an Indic script that needs a different font than the base NotoSans.
    Labels (always Latin) are left unwrapped so <b> resolves correctly.
    """
    font = _font_for(text)
    safe = _safe(text)
    if font != "NotoSans":
        return f'<font name="{font}">{safe}</font>'
    return safe


def generate_txt_report(data: dict) -> str:
    """
    Formats all prediction data into a clean plain-text report string.

    Args:
        data: prediction response dict (same shape returned by /predict)

    Returns:
        Formatted multi-line string (UTF-8 safe — caller encodes on write).
    """
    pred    = data.get("prediction", {})
    sev     = data.get("severity", {})
    rec     = data.get("recommendation", {})
    weather = data.get("weather", {})

    timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    crop       = data.get("crop_type", "Unknown")
    disease    = pred.get("disease_display", "Unknown")
    confidence = pred.get("confidence", 0)
    severity   = sev.get("level") or "Healthy"
    location   = data.get("location") or "Not specified"
    spray      = weather.get("advice") or "Not available"

    lines = [
        "==========================================",
        "     AGRIVISION CROP DISEASE REPORT",
        "==========================================",
        f"Date & Time : {timestamp}",
        f"Crop        : {crop}",
        f"Disease     : {disease}",
        f"Confidence  : {confidence}%",
        f"Severity    : {severity}",
        f"Location    : {location}",
        f"Weather     : {spray}",
        "------------------------------------------",
        "             AI RECOMMENDATIONS",
        "------------------------------------------",
    ]

    immediate = rec.get("immediate_actions", [])
    if immediate:
        lines.append("\nImmediate Actions:")
        for i, action in enumerate(immediate, 1):
            lines.append(f"  {i}. {action}")

    treatment = rec.get("treatment", [])
    if treatment:
        lines.append("\nTreatment:")
        for i, t in enumerate(treatment, 1):
            lines.append(f"  {i}. {t}")

    recovery = rec.get("recovery_time", "")
    if recovery:
        lines.append(f"\nExpected Recovery: {recovery}")

    measures = rec.get("preventive_measures", [])
    if measures:
        lines.append("\nPreventive Measures:")
        for i, m in enumerate(measures, 1):
            lines.append(f"  {i}. {m}")

    risk = rec.get("neighbouring_crop_risk", "")
    if risk:
        lines.append(f"\nNeighbouring Crop Risk: {risk}")

    lines.extend([
        "\n==========================================",
        "  Generated by AgriVision | Matrix Fusion 4.0",
        "==========================================",
    ])

    return "\n".join(lines)


def generate_pdf_report(data: dict, output_path: str) -> str:
    """
    Generates a clean A4 PDF report from prediction data using reportlab.

    Uses per-script Noto fonts so Indian language text (Malayalam, Hindi,
    Kannada, Tamil, etc.) renders correctly instead of blank squares.
    Each paragraph detects its dominant script and wraps the value in the
    correct <font> XML tag. Labels are always in NotoSans (Latin).

    Falls back to Helvetica silently if fonts could not be downloaded.

    Args:
        data: prediction response dict (same shape as generate_txt_report)
        output_path: full path where the .pdf file should be written

    Returns:
        output_path on success
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

    pred    = data.get("prediction", {})
    sev     = data.get("severity", {})
    rec     = data.get("recommendation", {})
    weather = data.get("weather", {})

    timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    crop       = data.get("crop_type", "Unknown")
    disease    = pred.get("disease_display", "Unknown")
    confidence = pred.get("confidence", 0)
    severity   = sev.get("level") or "Healthy"
    location   = data.get("location") or "Not specified"
    spray      = weather.get("advice") or "Not available"

    GREEN = colors.HexColor("#2d8a4e")
    DARK  = colors.HexColor("#1a1a1a")
    GRAY  = colors.HexColor("#555555")

    # Determine base font — NotoSans if registered, else Helvetica fallback
    try:
        pdfmetrics.getFont("NotoSans")
        font_normal = "NotoSans"
        font_bold   = "NotoSans-Bold"
    except Exception:
        font_normal = "Helvetica"
        font_bold   = "Helvetica-Bold"

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "AgriTitle", parent=styles["Title"],
        fontName=font_bold,
        textColor=GREEN, fontSize=18, spaceAfter=4,
    )
    heading_style = ParagraphStyle(
        "AgriHeading", parent=styles["Heading2"],
        fontName=font_bold,
        textColor=GREEN, fontSize=12, spaceBefore=12, spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "AgriBody", parent=styles["Normal"],
        fontName=font_normal,
        textColor=DARK, fontSize=10, leading=16,
    )
    meta_style = ParagraphStyle(
        "AgriMeta", parent=styles["Normal"],
        fontName=font_normal,
        textColor=GRAY, fontSize=10, leading=16,
    )

    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=inch, rightMargin=inch,
        topMargin=inch, bottomMargin=inch,
    )

    story = []

    # Title block
    story.append(Paragraph("AgriVision Crop Disease Report", title_style))
    story.append(Paragraph("AI Crop Intelligence for Indian Farmers — Matrix Fusion 4.0", meta_style))
    story.append(Spacer(1, 0.2 * inch))

    # Report meta fields — label always Latin, value wrapped in script font
    story.append(Paragraph("Report Details", heading_style))
    for label, value in [
        ("Date &amp; Time", timestamp),
        ("Crop",            crop),
        ("Disease",         disease),
        ("Confidence",      f"{confidence}%"),
        ("Severity",        severity),
        ("Location",        location),
        ("Weather",         spray),
    ]:
        story.append(Paragraph(f"<b>{label}:</b>  {_wrap(str(value))}", meta_style))

    story.append(Spacer(1, 0.1 * inch))

    # Immediate Actions
    immediate = rec.get("immediate_actions", [])
    if immediate:
        story.append(Paragraph("Immediate Actions", heading_style))
        for i, action in enumerate(immediate, 1):
            story.append(Paragraph(f"{i}. {_wrap(action)}", body_style))

    # Treatment
    treatment = rec.get("treatment", [])
    if treatment:
        story.append(Paragraph("Treatment", heading_style))
        for i, t in enumerate(treatment, 1):
            story.append(Paragraph(f"{i}. {_wrap(t)}", body_style))

    # Recovery Time
    recovery = rec.get("recovery_time", "")
    if recovery:
        story.append(Paragraph("Expected Recovery Time", heading_style))
        story.append(Paragraph(_wrap(recovery), body_style))

    # Preventive Measures
    measures = rec.get("preventive_measures", [])
    if measures:
        story.append(Paragraph("Preventive Measures", heading_style))
        for i, m in enumerate(measures, 1):
            story.append(Paragraph(f"{i}. {_wrap(m)}", body_style))

    # Neighbouring Crop Risk
    risk = rec.get("neighbouring_crop_risk", "")
    if risk:
        story.append(Paragraph("Neighbouring Crop Risk", heading_style))
        story.append(Paragraph(_wrap(risk), body_style))

    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("Generated by AgriVision | Matrix Fusion 4.0", meta_style))

    doc.build(story)
    return output_path
