import os
from datetime import datetime
from fpdf import FPDF

class MeetingReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 120)
        self.cell(0, 8, "AI VIDEO ASSISTANT - INTELLIGENCE REPORT", border=0, align="L")
        self.ln(4)
        self.set_draw_color(200, 200, 220)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y() + 4, self.w - self.r_margin, self.get_y() + 4)
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130, 130, 150)
        page_str = f"Page {self.page_no()}/{{nb}}"
        self.cell(0, 10, page_str, border=0, align="C")

def _clean_text(text: str) -> str:
    """Cleans markdown symbols and replaces Unicode characters for clean PDF typography."""
    if not text:
        return ""
    replacements = {
        "—": "-",
        "–": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "•": "*",
        "…": "...",
        "₹": "INR ",
        "✅": "[DONE] ",
        "🔑": "[KEY] ",
        "❓": "[?] ",
        "📋": "[SUMMARY] ",
        "📌": "[TITLE] ",
    }
    for orig, rep in replacements.items():
        text = text.replace(orig, rep)
    
    # Strip markdown bolding ** and *
    import re
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"###?\s*", "", text)
    
    # Encode and decode latin-1 safely
    return text.encode("latin-1", "replace").decode("latin-1")

def generate_pdf_report(title: str, summary: str, action_items: str, key_decisions: str, open_questions: str) -> bytes:
    pdf = MeetingReportPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Main Document Header
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(30, 30, 50)
    pdf.multi_cell(0, 9, _clean_text(title))
    pdf.ln(2)

    # Metadata Date
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(110, 110, 130)
    pdf.cell(0, 6, f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", ln=True)
    pdf.ln(6)

    def add_section(header_title: str, content: str):
        # Section Header Box
        pdf.set_fill_color(240, 242, 248)
        pdf.set_text_color(40, 40, 80)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, f"  {header_title}", ln=True, fill=True)
        pdf.ln(3)

        # Content
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 50)
        cleaned = _clean_text(content)
        pdf.multi_cell(0, 5.5, cleaned)
        pdf.ln(6)

    if summary:
        add_section("1. Executive Summary", summary)
    if action_items:
        add_section("2. Action Items & Implementation Steps", action_items)
    if key_decisions:
        add_section("3. Key Decisions & Technical Architecture", key_decisions)
    if open_questions:
        add_section("4. Open Questions & Technical Challenges", open_questions)

    return bytes(pdf.output())
