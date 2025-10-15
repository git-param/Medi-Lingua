import io
from typing import List, Dict

import streamlit as st


def _import_reportlab():
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        from reportlab.lib import colors
        return {
            "A4": A4,
            "getSampleStyleSheet": getSampleStyleSheet,
            "mm": mm,
            "canvas": canvas,
            "Paragraph": Paragraph,
            "SimpleDocTemplate": SimpleDocTemplate,
            "Spacer": Spacer,
            "Table": Table,
            "TableStyle": TableStyle,
            "colors": colors,
        }
    except Exception:
        return None


def build_chat_pdf(messages: List[Dict[str, str]], title: str = "MediLingua Chat Transcript") -> io.BytesIO:
    """
    Create a PDF from chat messages and return as an in-memory BytesIO buffer.
    Each message is a dict with keys: role ('user'|'assistant'), content (str).
    """
    libs = _import_reportlab()
    if libs is None:
        st.error(
            "PDF generation library not found. Install with: `pip install reportlab` and rerun."
        )
        return None

    buffer = io.BytesIO()

    doc = libs["SimpleDocTemplate"](buffer, pagesize=libs["A4"], rightMargin=28, leftMargin=28, topMargin=36, bottomMargin=28)
    styles = libs["getSampleStyleSheet"]()

    elements = []

    # Title
    title_style = styles["Title"]
    elements.append(libs["Paragraph"](title, title_style))
    elements.append(libs["Spacer"](1, 12))

    # Build a table-like layout for messages
    data = []
    table_style_cmds = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, libs["colors"].lightgrey),
        ("BOX", (0, 0), (-1, -1), 0.25, libs["colors"].lightgrey),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]

    role_style = styles["Heading5"]
    body_style = styles["BodyText"]

    for msg in messages:
        role = msg.get("role", "").capitalize()
        content = msg.get("content", "")

        # Left column: role, Right column: content paragraph
        role_par = libs["Paragraph"](f"<b>{role}</b>", role_style)
        content_par = libs["Paragraph"](content.replace("\n", "<br/>"), body_style)
        data.append([role_par, content_par])

    table = libs["Table"](data, colWidths=[30 * libs["mm"], None])
    table.setStyle(libs["TableStyle"](table_style_cmds))
    elements.append(table)

    doc.build(elements)
    buffer.seek(0)
    return buffer


