"""Generate the assignment submission PDF directly from the current README.

The default output is deliberately marked DRAFT because the current BITS Lab screenshot does not
show readable metric results. After an original replacement screenshot is supplied, run with
--final and --screenshot to create the exact final filename.
"""
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
README_PATH = PROJECT_ROOT / "README.md"
DEFAULT_SCREENSHOT = PROJECT_ROOT / "screenshots" / "bits_lab_execution.png"
OUTPUT_DIR = PROJECT_ROOT / "output" / "pdf"
BASE_NAME = "Sumer_Lakhera_2025NS04051_ML_Assignment_2"
GITHUB_URL = "https://github.com/LakheraSumer/Use-bits-wilp-ml-assignment-2"
APP_URL = "https://bits-wilp-ml-assignment-2-sumer.streamlit.app/"
BLUE = colors.HexColor("#153A5B")
LINK_BLUE = "#1D4ED8"
LIGHT_BLUE = colors.HexColor("#EAF2F8")
LIGHT_GREY = colors.HexColor("#F4F6F7")
GRID_GREY = colors.HexColor("#AAB2B8")
TEXT = colors.HexColor("#17202A")


def styles():
    sheet = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "SubmissionTitle",
            parent=sheet["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=BLUE,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=sheet["Normal"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=14,
            alignment=TA_CENTER,
            textColor=TEXT,
            spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "Heading1",
            parent=sheet["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=BLUE,
            spaceBefore=12,
            spaceAfter=7,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "Heading2",
            parent=sheet["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=BLUE,
            spaceBefore=10,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "Heading3",
            parent=sheet["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=TEXT,
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=sheet["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=TEXT,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=sheet["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12.5,
            leftIndent=12,
            firstLineIndent=-9,
            textColor=TEXT,
            spaceAfter=4,
        ),
        "table": ParagraphStyle(
            "TableCell",
            parent=sheet["BodyText"],
            fontName="Helvetica",
            fontSize=8.1,
            leading=10.3,
            textColor=TEXT,
            spaceAfter=0,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=sheet["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.1,
            leading=10.3,
            textColor=colors.white,
            alignment=TA_LEFT,
            spaceAfter=0,
        ),
        "code": ParagraphStyle(
            "Code",
            parent=sheet["Code"],
            fontName="Courier",
            fontSize=8,
            leading=10.5,
            leftIndent=8,
            rightIndent=8,
            spaceBefore=3,
            spaceAfter=7,
            backColor=LIGHT_GREY,
            borderPadding=7,
        ),
        "draft": ParagraphStyle(
            "Draft",
            parent=sheet["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9.5,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#8B1E1E"),
            backColor=colors.HexColor("#FDEDEC"),
            borderColor=colors.HexColor("#C0392B"),
            borderWidth=0.6,
            borderPadding=7,
            spaceAfter=10,
        ),
        "link": ParagraphStyle(
            "Link",
            parent=sheet["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            leftIndent=10,
            spaceAfter=7,
            textColor=TEXT,
        ),
    }


def _link_markup(label: str, target: str) -> str:
    if target == "test_data.csv":
        target = f"{GITHUB_URL}/blob/main/test_data.csv"
    return f'<link href="{html.escape(target, quote=True)}" color="{LINK_BLUE}">{label}</link>'


def inline_markdown(value: str) -> str:
    """Convert the small inline-Markdown subset used by README.md to ReportLab markup."""
    tokens: list[str] = []

    def stash(markup: str) -> str:
        tokens.append(markup)
        return f"@@TOKEN{len(tokens) - 1}@@"

    def markdown_link(match: re.Match[str]) -> str:
        label = inline_markdown(match.group(1))
        return stash(_link_markup(label, match.group(2)))

    value = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", markdown_link, value)
    value = re.sub(
        r"<(https?://[^>]+)>",
        lambda match: stash(_link_markup(html.escape(match.group(1)), match.group(1))),
        value,
    )
    value = re.sub(
        r"`([^`]+)`",
        lambda match: stash(f'<font name="Courier">{html.escape(match.group(1))}</font>'),
        value,
    )
    value = html.escape(value)
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"\*(.+?)\*", r"<i>\1</i>", value)
    for index, markup in enumerate(tokens):
        value = value.replace(f"@@TOKEN{index}@@", markup)
    return value


def make_table(rows: list[list[str]], style_map: dict[str, ParagraphStyle]) -> LongTable:
    column_count = len(rows[0])
    if column_count == 7:
        widths = [1.55 * inch] + [0.90 * inch] * 6
    elif column_count == 2:
        widths = [1.62 * inch, 5.33 * inch]
    else:
        widths = [6.95 * inch / column_count] * column_count

    table_rows = []
    for row_index, row in enumerate(rows):
        cell_style = style_map["table_header"] if row_index == 0 else style_map["table"]
        table_rows.append([Paragraph(inline_markdown(cell.strip()), cell_style) for cell in row])

    table = LongTable(
        table_rows,
        colWidths=widths,
        repeatRows=1,
        splitByRow=1,
        hAlign="LEFT",
    )
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.45, GRID_GREY),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for row_index in range(1, len(table_rows)):
        background = colors.white if row_index % 2 else LIGHT_BLUE
        commands.append(("BACKGROUND", (0, row_index), (-1, row_index), background))
    table.setStyle(TableStyle(commands))
    return table


def parse_readme(markdown: str, style_map: dict[str, ParagraphStyle]):
    lines = markdown.splitlines()
    story = []
    index = 0
    paragraph_parts: list[str] = []

    def flush_paragraph():
        nonlocal paragraph_parts
        if paragraph_parts:
            text = " ".join(part.strip() for part in paragraph_parts)
            story.append(Paragraph(inline_markdown(text), style_map["body"]))
            paragraph_parts = []

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph()
            index += 1
            code_lines = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            story.append(Preformatted("\n".join(code_lines), style_map["code"]))
            index += 1
            continue

        if stripped.startswith("|"):
            flush_paragraph()
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            rows = [[cell.strip() for cell in row.strip("|").split("|")] for row in table_lines]
            if len(rows) > 1 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in rows[1]):
                rows.pop(1)
            story.append(make_table(rows, style_map))
            story.append(Spacer(1, 7))
            continue

        if stripped.startswith("#"):
            flush_paragraph()
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped[level:].strip()
            style_name = "h1" if level == 1 else "h2" if level == 2 else "h3"
            story.append(Paragraph(inline_markdown(text), style_map[style_name]))
            index += 1
            continue

        if stripped.startswith("- "):
            flush_paragraph()
            bullet_parts = [stripped[2:]]
            index += 1
            while index < len(lines):
                continuation = lines[index]
                if not continuation.strip() or continuation.strip().startswith(("- ", "#", "|", "```")):
                    break
                bullet_parts.append(continuation.strip())
                index += 1
            story.append(Paragraph("- " + inline_markdown(" ".join(bullet_parts)), style_map["bullet"]))
            continue

        if not stripped:
            flush_paragraph()
            index += 1
            continue

        paragraph_parts.append(stripped)
        index += 1

    flush_paragraph()
    return story


def page_decoration(canvas, document, is_draft: bool):
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(colors.HexColor("#D5D8DC"))
    canvas.setLineWidth(0.4)
    canvas.line(document.leftMargin, 27, width - document.rightMargin, 27)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#616A6B"))
    footer = f"Sumer Lakhera - 2025NS04051  |  Page {document.page}"
    canvas.drawString(document.leftMargin, 16, footer)
    if is_draft:
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.setFillColor(colors.HexColor("#A93226"))
        canvas.drawRightString(
            width - document.rightMargin,
            16,
            "DRAFT - replacement BITS Lab results screenshot required",
        )
    canvas.restoreState()


def build_pdf(output_path: Path, screenshot_path: Path, final: bool):
    style_map = styles()
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.48 * inch,
        title=f"{BASE_NAME}{'' if final else ' - DRAFT'}",
        author="Sumer Lakhera",
        subject="BITS Pilani WILP Machine Learning Assignment 2",
    )

    story = [
        Paragraph(
            f"{'FINAL SUBMISSION' if final else 'DRAFT SUBMISSION'}<br/>"
            "Machine Learning - Assignment 2",
            style_map["title"],
        ),
        Paragraph(
            "Sumer Lakhera | BITS ID: 2025NS04051 | M.Tech (DSE), BITS Pilani WILP",
            style_map["subtitle"],
        ),
    ]
    if not final:
        story.append(
            Paragraph(
                "DRAFT: the current original BITS Virtual Lab screenshot does not show readable "
                "metric results. Replace it with the requested original screenshot before submission.",
                style_map["draft"],
            )
        )

    story.extend(
        [
            Paragraph("<b>1. Clickable GitHub repository link</b>", style_map["h2"]),
            Paragraph(_link_markup(html.escape(GITHUB_URL), GITHUB_URL), style_map["link"]),
            Paragraph("<b>2. Clickable live Streamlit Community Cloud app link</b>", style_map["h2"]),
            Paragraph(_link_markup(html.escape(APP_URL), APP_URL), style_map["link"]),
            Paragraph("<b>3. Original BITS Virtual Lab execution screenshot</b>", style_map["h2"]),
        ]
    )

    screenshot = Image(str(screenshot_path))
    available_width = 6.95 * inch
    available_height = 3.75 * inch
    scale = min(available_width / screenshot.imageWidth, available_height / screenshot.imageHeight)
    screenshot.drawWidth = screenshot.imageWidth * scale
    screenshot.drawHeight = screenshot.imageHeight * scale
    story.append(KeepTogether([screenshot, Spacer(1, 5)]))
    if not final:
        story.append(
            Paragraph(
                "Screenshot status: original lab context is visible, but the complete model-results "
                "table is below the captured area. This image is preserved unchanged as a placeholder.",
                style_map["draft"],
            )
        )

    story.extend(
        [
            PageBreak(),
            Paragraph("4. Complete README content", style_map["h1"]),
        ]
    )
    story.extend(parse_readme(README_PATH.read_text(encoding="utf-8"), style_map))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.build(
        story,
        onFirstPage=lambda canvas, doc: page_decoration(canvas, doc, not final),
        onLaterPages=lambda canvas, doc: page_decoration(canvas, doc, not final),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=DEFAULT_SCREENSHOT,
        help="Original BITS Virtual Lab screenshot to embed.",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="Create the exact final filename without a DRAFT label.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    screenshot_path = args.screenshot.resolve()
    if not screenshot_path.is_file():
        raise FileNotFoundError(f"Screenshot not found: {screenshot_path}")
    if args.final and screenshot_path == DEFAULT_SCREENSHOT.resolve():
        raise ValueError(
            "The current screenshot does not show readable model results. "
            "Supply the replacement original with --screenshot before using --final."
        )

    suffix = ".pdf" if args.final else "_DRAFT.pdf"
    output_path = OUTPUT_DIR / f"{BASE_NAME}{suffix}"
    build_pdf(output_path, screenshot_path, final=args.final)
    print(output_path)


if __name__ == "__main__":
    main()
