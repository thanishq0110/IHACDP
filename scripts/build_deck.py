"""Build the review presentation from Markdown slide content.

Every slide gets a layout chosen for its own content rather than one bullet
template repeated twenty times: stat cards where the point is a number, a
split panel where the point is a contrast, a full-width figure where the
diagram carries the argument.

PowerPoint's autofit cannot be driven reliably through python-pptx, so text is
measured here (fit_size) and the largest size that still fits its box is used.
"""
from __future__ import annotations
import re
from math import ceil
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "chapters" / "slides.md"
FIG = ROOT / "docs" / "figures"
OUT = ROOT / "docs" / "IHACDP_Presentation.pptx"
VIT_LOGO = ROOT / "docs" / "assets" / "vit-logo.png"

FONT = "Helvetica Neue"

INK = RGBColor(0x0F, 0x2A, 0x33)
BODY = RGBColor(0x33, 0x50, 0x5C)
MUTED = RGBColor(0x6B, 0x84, 0x8E)
TEAL = RGBColor(0x0D, 0x6E, 0x6E)
TEAL_BRIGHT = RGBColor(0x14, 0xA0, 0xA0)
DARK = RGBColor(0x0B, 0x1B, 0x24)
PANEL = RGBColor(0x12, 0x27, 0x31)
EDGE = RGBColor(0x1F, 0x3C, 0x48)
CARD = RGBColor(0xF3, 0xF8, 0xF8)
LINE = RGBColor(0xD5, 0xE4, 0xE4)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PALE = RGBColor(0xDD, 0xEA, 0xEA)
AMBER = RGBColor(0x9A, 0x5B, 0x1E)

W, H = Inches(13.333), Inches(7.5)
M = 0.62                     # page margin, inches
CW = 13.333 - 2 * M          # content width
TOP = 1.62                   # first usable row under the title block
BOT = 6.78                   # last usable row above the footer


# ---------------------------------------------------------------- primitives

def clean(t) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"\1", str(t)).strip()


def _txt(sl, text, x, y, w, h, size, *, color=BODY, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=1.0, italic=False):
    tb = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    lines = text.split("\n") if isinstance(text, str) else list(text)
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = clean(ln)
        p.alignment = align
        p.line_spacing = spacing
        f = p.font
        f.name, f.size, f.bold, f.italic = FONT, Pt(size), bold, italic
        f.color.rgb = color
    return tb


def _rect(sl, x, y, w, h, fill, *, line=None, radius=None, line_w=1.0):
    shape = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    s = sl.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if radius:
        s.adjustments[0] = radius
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line; s.line.width = Pt(line_w)
    s.shadow.inherit = False
    return s


def fit_size(text, w: float, h: float, sizes, spacing=1.30) -> float:
    """Largest size from `sizes` whose wrapped text fits a w x h inch box."""
    for size in sizes:
        cpl = max(1, int(w * 72 / (0.505 * size)))
        lines = sum(max(1, ceil(len(clean(ln)) / cpl)) for ln in str(text).split("\n"))
        if lines * size * spacing / 72 <= h:
            return size
    return sizes[-1]


def picture_fit(sl, path: Path, x, y, w, h):
    """Place an image centred in a box, preserving aspect ratio."""
    from PIL import Image
    iw, ih = Image.open(path).size
    scale = min(w / (iw / 96), h / (ih / 96))
    pw, ph = (iw / 96) * scale, (ih / 96) * scale
    return sl.shapes.add_picture(str(path), Inches(x + (w - pw) / 2),
                                 Inches(y + (h - ph) / 2), width=Inches(pw))


# -------------------------------------------------------------------- chrome

def page(prs, title, number, kicker=None, *, dark=False):
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    if dark:
        _rect(sl, 0, 0, 13.333, 7.5, DARK)
    _rect(sl, 0, 0, 13.333, 0.075, TEAL_BRIGHT if dark else TEAL)

    if kicker:
        _txt(sl, kicker.upper(), M, 0.42, CW, 0.26, 11.5,
             color=TEAL_BRIGHT if dark else TEAL, bold=True)
    ty = 0.74 if kicker else 0.52
    _txt(sl, title, M, ty, CW, 0.82, fit_size(title, CW, 0.8, [30, 27, 24, 21]),
         color=WHITE if dark else INK, bold=True)
    _rect(sl, M, ty + 0.74, 1.35, 0.045, TEAL_BRIGHT if dark else TEAL)

    _txt(sl, "IHACDP", M, 6.95, 4.0, 0.32, 12, color=MUTED)
    _txt(sl, str(number), 13.333 - M - 0.7, 6.95, 0.7, 0.32, 12,
         color=MUTED, align=PP_ALIGN.RIGHT, bold=True)
    return sl


def cards(sl, items, *, cols, top, bottom, numbered=False, tone=CARD):
    """Grid of cards sized to their content, then centred in the region.

    Stretching cards to fill the region left two-thirds of each one empty and
    let every slide pick its own text size; both read as sloppy across a deck.
    """
    rows = ceil(len(items) / cols)
    gap = 0.24
    cw = (CW - gap * (cols - 1)) / cols
    inner = cw - (1.16 if numbered else 0.56)
    avail = (bottom - top - gap * (rows - 1)) / rows

    size = fit_size(max(items, key=len), inner, avail - 0.56,
                    [21, 20, 19, 18, 17, 16, 15, 14, 13])
    pad = 0.28
    lines = max(ceil(len(clean(i)) / max(1, int(inner * 72 / (0.505 * size)))) for i in items)
    ch = min(avail, max(1.0, lines * size * 1.32 / 72 + 2 * pad))
    y0 = top + (bottom - top - (rows * ch + gap * (rows - 1))) / 2   # centre the block

    for i, item in enumerate(items):
        r, c = divmod(i, cols)
        in_row = min(cols, len(items) - r * cols)
        indent = (CW - (in_row * cw + gap * (in_row - 1))) / 2 if in_row < cols else 0
        x = M + indent + (i - r * cols) * (cw + gap)
        y = y0 + r * (ch + gap)
        _rect(sl, x, y, cw, ch, tone, line=LINE, radius=0.06)
        _rect(sl, x, y, 0.05, ch, TEAL)
        tx, tw = x + 0.28, cw - 0.56
        if numbered:
            _txt(sl, f"{i + 1:02d}", tx, y + ch / 2 - 0.17, 0.62, 0.36, 16,
                 color=TEAL_BRIGHT, bold=True)
            tx, tw = tx + 0.6, tw - 0.6
        _txt(sl, item, tx, y, tw, ch, size, anchor=MSO_ANCHOR.MIDDLE)


def stat(sl, value, label, x, y, w, h, *, tone=CARD, value_color=TEAL,
         border=LINE, label_color=BODY):
    _rect(sl, x, y, w, h, tone, line=border, radius=0.05)
    _txt(sl, value, x, y + h * 0.18, w, h * 0.44,
         fit_size(value, w - 0.3, h * 0.44, [40, 34, 28, 24]),
         color=value_color, bold=True, align=PP_ALIGN.CENTER)
    _txt(sl, label, x + 0.18, y + h * 0.64, w - 0.36, h * 0.3,
         fit_size(label, w - 0.36, h * 0.3, [14, 13, 12, 11]),
         color=label_color, align=PP_ALIGN.CENTER)


def callout(sl, text, y, *, tone=TEAL, color=WHITE, h=0.78, label=None):
    _rect(sl, M, y, CW, h, tone, radius=0.05)
    x, w = M + 0.32, CW - 0.64
    if label:
        _txt(sl, label.upper(), x, y + 0.14, 3.4, 0.24, 11, color=PALE, bold=True)
        _txt(sl, text, x, y + 0.42, w, h - 0.54,
             fit_size(text, w, h - 0.54, [17, 16, 15, 14, 13]), color=color, bold=True)
    else:
        _txt(sl, text, x, y, w, h, fit_size(text, w, h - 0.2, [18, 17, 16, 15]),
             color=color, bold=True, anchor=MSO_ANCHOR.MIDDLE)


def row_list(sl, items, top, bottom, *, accent=TEAL, width=None, tag=None):
    """Full-width rows with a coloured spine - used for gaps and limitations."""
    width = width or CW
    gap = 0.11
    gh = (bottom - top - gap * (len(items) - 1)) / len(items)
    for i, it in enumerate(items):
        y = top + i * (gh + gap)
        _rect(sl, M, y, width, gh, WHITE, line=LINE, radius=0.06)
        _rect(sl, M, y, 0.055, gh, accent)
        tx = M + 0.34
        if tag:
            _txt(sl, f"{tag} {i + 1}", tx, y + gh / 2 - 0.13, 0.95, 0.26, 11.5,
                 color=accent, bold=True)
            tx += 1.05
        _txt(sl, it, tx, y, width - (tx - M) - 0.3, gh,
             fit_size(it, width - (tx - M) - 0.3, gh - 0.1, [19, 18, 17, 16, 15]),
             anchor=MSO_ANCHOR.MIDDLE)


def bullet_column(sl, items, x, y, w, bottom, *, size=None, color=BODY, dot=TEAL):
    """Flow the list with one constant gap, so wrapped items do not crowd."""
    tw = w - 0.34
    for size in ([size] if size else [19, 18, 17, 16, 15, 14, 13]):
        cpl = max(1, int(tw * 72 / (0.505 * size)))
        heights = [max(1, ceil(len(clean(i)) / cpl)) * size * 1.32 / 72 for i in items]
        gap = (bottom - y - sum(heights)) / max(len(items) - 1, 1)
        if gap >= 0.16:
            break
    gap = min(gap, 0.62)
    cy = y
    for it, h in zip(items, heights):
        _rect(sl, x, cy + size / 72 * 0.42, 0.1, 0.1, dot, radius=0.5)
        _txt(sl, it, x + 0.34, cy, tw, h + 0.04, size, color=color, spacing=1.32)
        cy += h + gap


def table(sl, rows, x, y, w, *, max_h=None, col_widths=None, max_row=0.46):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [c for c in cells if not all(set(v) <= set("-: ") for v in c)]
    if not cells:
        return None
    ncol = max(len(c) for c in cells)
    cells = [c + [""] * (ncol - len(c)) for c in cells]
    nrow = len(cells)
    rh = min(max_row, (max_h or 99) / nrow)
    tbl = sl.shapes.add_table(nrow, ncol, Inches(x), Inches(y),
                              Inches(w), Inches(rh * nrow)).table
    tbl.first_row = True
    if col_widths:
        for i, cwid in enumerate(col_widths[:ncol]):
            tbl.columns[i].width = Inches(cwid)
    size = 14 if ncol <= 4 else 12
    for r in range(nrow):
        tbl.rows[r].height = Inches(rh)
        for c in range(ncol):
            cell = tbl.cell(r, c)
            cell.text = clean(cells[r][c])
            cell.margin_left = cell.margin_right = Inches(0.1)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.font.name, p.font.size, p.font.bold = FONT, Pt(size), r == 0
            p.font.color.rgb = WHITE if r == 0 else BODY
            cell.fill.solid()
            cell.fill.fore_color.rgb = TEAL if r == 0 else (CARD if r % 2 else WHITE)
    return tbl


def chip_row(sl, labels, x, y, width, h, *, fill=WHITE, border=LINE, text=INK,
             size=15, gap=0.17):
    """Lay chips on a single line, shrinking them together if they would not fit."""
    widths = [0.4 + len(l) * 0.115 for l in labels]
    total = sum(widths) + gap * (len(labels) - 1)
    if total > width:
        scale = (width - gap * (len(labels) - 1)) / sum(widths)
        widths = [w * scale for w in widths]
        size = max(11, size * min(1.0, scale + 0.08))
    cx = x
    for label, w in zip(labels, widths):
        _rect(sl, cx, y, w, h, fill, line=border, radius=0.35)
        _txt(sl, label, cx, y, w, h, size, color=text, bold=True,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        cx += w + gap
    return y + h


def body_lines(s):
    return [ln.strip("-*• ").strip() for ln in s["body"].splitlines()
            if ln.strip().startswith(("-", "*", "•"))]


def table_rows(s):
    return [ln for ln in s["body"].splitlines() if ln.strip().startswith("|")]


def parse(text: str):
    slides, cur = [], None
    for line in text.splitlines():
        m = re.match(r"^##\s*Slide\s*(\d+)\s*[::]\s*(.+)$", line.strip(), re.I)
        if m:
            cur = {"n": int(m.group(1)), "title": m.group(2).strip(), "body": []}
            slides.append(cur); continue
        if cur is not None:
            cur["body"].append(line)
    for s in slides:
        s["body"] = "\n".join(s["body"]).strip()
    return slides


# --------------------------------------------------------------------- pages

def s01_title(prs, s):
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    _rect(sl, 0, 0, 13.333, 7.5, DARK)
    _rect(sl, 0, 0, 13.333, 0.075, TEAL_BRIGHT)

    if VIT_LOGO.exists():
        from PIL import Image
        iw, ih = Image.open(VIT_LOGO).size
        lh, pad = 0.56, 0.15
        _rect(sl, M, 0.48, lh * iw / ih + 2 * pad, lh + 2 * pad, WHITE, radius=0.12)
        sl.shapes.add_picture(str(VIT_LOGO), Inches(M + pad), Inches(0.48 + pad),
                              height=Inches(lh))

    _txt(sl, "IHACDP", M, 1.72, 9.0, 1.2, 66, color=WHITE, bold=True)
    _txt(sl, "Intelligent Healthcare Analytics and Clinical Decision Support Platform",
         M, 3.02, 9.8, 0.5, 20, color=RGBColor(0x8F, 0xC6, 0xC6))
    _rect(sl, M, 3.66, 1.6, 0.045, TEAL_BRIGHT)
    _txt(sl, "Offline clinical decision support with statistical risk models\n"
             "and a local language model",
         M, 3.92, 8.4, 0.9, 16, color=RGBColor(0xB9, 0xD2, 0xD2), spacing=1.25)
    _txt(sl, "B.Tech, School of Computer Science and Engineering, VIT Chennai"
             "      ·      Capstone Project Review",
         M, 5.12, CW, 0.3, 13, color=MUTED)

    _txt(sl, "GUIDE", 10.55, 3.92, 1.4, 0.24, 11, color=TEAL_BRIGHT, bold=True)
    _txt(sl, "Dr. Vivekanandan M", 10.55, 4.2, 2.3, 0.3, 15, color=WHITE, bold=True)
    _txt(sl, "(53646)", 10.55, 4.52, 2.3, 0.28, 13, color=MUTED)

    team = [b for b in body_lines(s) if re.search(r"\d{2}[A-Z]{3}\d{4}", b)]
    cw, gap = (CW - 2 * 0.28) / 3, 0.28
    for i, member in enumerate(team[:3]):
        name, reg = [p.strip() for p in re.split(r"\s+[—–-]\s+", member, maxsplit=1)]
        x = M + i * (cw + gap)
        _rect(sl, x, 5.78, cw, 0.9, PANEL, line=EDGE, radius=0.08)
        _rect(sl, x, 5.78, 0.05, 0.9, TEAL_BRIGHT)
        _txt(sl, name, x + 0.26, 5.96, cw - 0.5, 0.3,
             fit_size(name, cw - 0.5, 0.3, [15, 14, 13]), color=WHITE, bold=True)
        _txt(sl, reg, x + 0.26, 6.28, cw - 0.5, 0.26, 13, color=TEAL_BRIGHT)
    return sl


def s02_agenda(prs, s):
    sl = page(prs, "Agenda", 2, "What we will cover")
    cards(sl, body_lines(s), cols=2, top=TOP + 0.08, bottom=BOT, numbered=True)
    return sl


def s03_problem(prs, s):
    sl = page(prs, "Problem Definition and Domain Understanding", 3, "Context · 1 of 2")
    items = body_lines(s)
    tagged = [i for i in items if i.startswith(("Core problem", "Constraint"))]
    plain = [i for i in items if i not in tagged]
    cards(sl, plain, cols=2, top=TOP, bottom=TOP + 2.05)
    for i, t in enumerate(tagged[:2]):
        label, text = t.split(":", 1)
        callout(sl, text.strip(), TOP + 2.42 + i * 1.10, label=label.strip(),
                tone=TEAL if i == 0 else INK, h=0.86)
    return sl


def s04_motivation(prs, s):
    sl = page(prs, "Motivation: the Indian Primary-Care Context", 4, "Context · 2 of 2")
    items = body_lines(s)
    cards(sl, items[:-1], cols=3, top=TOP, bottom=TOP + 2.6)
    callout(sl, items[-1], TOP + 2.92, tone=TEAL, h=0.9)
    return sl


def s05_lit_checkers(prs, s):
    sl = page(prs, "Symptom Checkers and Their Limitations", 5, "Literature review · 1 of 2")
    items = body_lines(s)
    hero = next((i for i in items if "46%" in i), None)
    rest = [i for i in items if i is not hero]
    bullet_column(sl, rest, M, TOP + 0.08, 7.9, BOT - 0.15)

    x, pw = M + 8.4, CW - 8.4
    _rect(sl, x, TOP, pw, BOT - TOP, CARD, line=LINE, radius=0.05)
    _txt(sl, "THE FAILURE MODE", x, TOP + 0.42, pw, 0.26, 11.5, color=TEAL, bold=True,
         align=PP_ALIGN.CENTER)
    _txt(sl, "46%", x, TOP + 1.02, pw, 1.4, 66, color=AMBER, bold=True, align=PP_ALIGN.CENTER)
    _txt(sl, "confidence that a head cold\nwas AIDS", x + 0.28, TOP + 2.6, pw - 0.56, 0.8,
         15, color=INK, bold=True, align=PP_ALIGN.CENTER, spacing=1.25)
    _rect(sl, x + 0.5, TOP + 3.66, pw - 1.0, 0.02, LINE)
    _txt(sl, "Naive Bayes on our own data, before\nIDF-weighted overlap replaced it",
         x + 0.28, TOP + 3.92, pw - 0.56, 0.9, 14.5, color=BODY, align=PP_ALIGN.CENTER,
         spacing=1.3)
    return sl


def s06_lit_ml(prs, s):
    sl = page(prs, "ML Risk Prediction and Explainability", 6, "Literature review · 2 of 2")
    items = body_lines(s)
    cards(sl, items[:4], cols=2, top=TOP, bottom=TOP + 2.95)
    _txt(sl, items[4], M, TOP + 3.22, CW, 0.42, 15.5, color=MUTED, italic=True)
    callout(sl, items[5], TOP + 3.86, tone=INK, h=1.02, label="The gap this project sits in")
    return sl


def s07_gap(prs, s):
    sl = page(prs, "Research Gap", 7, "Where the field stops")
    items = body_lines(s)
    row_list(sl, items[:-1], TOP, BOT - 0.92, accent=AMBER, tag="GAP")
    callout(sl, items[-1], BOT - 0.78, tone=TEAL, h=0.78)
    return sl


def s08_objectives(prs, s):
    sl = page(prs, "Objectives and Scope", 8, "What we set out to build")
    cards(sl, body_lines(s), cols=3, top=TOP, bottom=BOT, numbered=True)
    return sl


def s09_architecture(prs, s):
    sl = page(prs, "System Architecture: the Request Path", 9, "System design")
    fig = FIG / "fig_architecture_wide.png"
    if fig.exists():
        picture_fit(sl, fig, M, TOP - 0.05, CW, 3.95)
    pts = body_lines(s)
    keys = [pts[0], pts[2], pts[5]]
    cw, gap = (CW - 2 * 0.24) / 3, 0.24
    for i, k in enumerate(keys):
        x = M + i * (cw + gap)
        _rect(sl, x, 5.66, cw, 1.12, CARD, line=LINE, radius=0.06)
        _rect(sl, x, 5.66, cw, 0.05, TEAL)
        _txt(sl, k, x + 0.26, 5.78, cw - 0.52, 0.88,
             fit_size(k, cw - 0.52, 0.98, [15, 14, 13, 12]), anchor=MSO_ANCHOR.MIDDLE)
    return sl


def s10_datasets(prs, s):
    sl = page(prs, "Datasets Used", 10, "Data sources")
    table(sl, table_rows(s), M, TOP, 8.1, max_h=4.85, max_row=0.54)
    x, pw = M + 8.45, CW - 8.45
    _rect(sl, x, TOP, pw, 4.85, CARD, line=LINE, radius=0.05)
    _txt(sl, "244 TRIAGE CONDITIONS", x, TOP + 0.24, pw, 0.3, 12.5, color=TEAL,
         bold=True, align=PP_ALIGN.CENTER)
    fig = FIG / "fig_triage_sources.png"
    if fig.exists():
        picture_fit(sl, fig, x + 0.14, TOP + 0.62, pw - 0.28, 3.3)
    _txt(sl, "41 public  ·  174 SymCat\n29 curated clinical patterns",
         x + 0.2, TOP + 4.02, pw - 0.4, 0.66, 14, color=BODY, align=PP_ALIGN.CENTER,
         spacing=1.3)
    return sl


def s11_wall(prs, s):
    sl = page(prs, "Deterministic Extraction and the Observed/Imputed Wall", 11,
              "Methodology · 1 of 2")
    pw = 5.36
    cx = M + pw + 0.5                      # wall centre line
    _rect(sl, M, TOP, pw, 3.02, CARD, line=LINE, radius=0.05)
    _txt(sl, "WHAT THE RISK MODEL SEES", M + 0.3, TOP + 0.3, pw - 0.6, 0.26, 11.5,
         color=TEAL, bold=True)
    _txt(sl, "Observed features\n+  imputed placeholders", M + 0.3, TOP + 0.72, pw - 0.6,
         1.0, 19, color=INK, bold=True, spacing=1.3)
    _txt(sl, "Imputed values fill the input vector so the model can score a complete record.",
         M + 0.3, TOP + 1.94, pw - 0.6, 0.9, 14, spacing=1.25)

    rx = cx + 0.5
    _rect(sl, rx, TOP, pw, 3.02, WHITE, line=TEAL, radius=0.05, line_w=1.6)
    _txt(sl, "WHAT THE NARRATOR SEES", rx + 0.3, TOP + 0.3, pw - 0.6, 0.26, 11.5,
         color=TEAL, bold=True)
    _txt(sl, "Observed features only", rx + 0.3, TOP + 0.72, pw - 0.6, 0.5, 19,
         color=INK, bold=True, spacing=1.3)
    _txt(sl, "The language model can only describe evidence the patient actually gave, "
             "so a fabricated finding has nothing to be built from.",
         rx + 0.3, TOP + 1.5, pw - 0.6, 1.3, 14, spacing=1.35)

    _rect(sl, cx - 0.05, TOP - 0.14, 0.1, 3.3, INK)
    _rect(sl, cx - 0.42, TOP + 1.16, 0.84, 0.7, WHITE, radius=0.2)   # chip clears the bar
    _txt(sl, "THE\nWALL", cx - 0.42, TOP + 1.16, 0.84, 0.7, 11, color=INK, bold=True,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, spacing=1.15)

    said = ("sees only evidence", "never reach the narrator")
    rest = [b for b in body_lines(s)
            if "observed or imputed" not in b and not any(t in b for t in said)][:3]
    cw, gap = (CW - 2 * 0.24) / 3, 0.24
    for i, b in enumerate(rest):
        x = M + i * (cw + gap)
        _rect(sl, x, TOP + 3.34, cw, 1.24, CARD, line=LINE, radius=0.06)
        _rect(sl, x, TOP + 3.34, cw, 0.05, TEAL)
        _txt(sl, b, x + 0.26, TOP + 3.5, cw - 0.52, 0.92,
             fit_size(b, cw - 0.52, 0.92, [14, 13, 12]), anchor=MSO_ANCHOR.MIDDLE)
    return sl


def s12_triage(prs, s):
    sl = page(prs, "Triage Layer: Why IDF Overlap Beat Naive Bayes", 12,
              "Methodology · 2 of 2")
    sw, gap = (CW - 2 * 0.26) / 3, 0.26
    for i, (v, l) in enumerate([("244", "conditions"), ("389", "symptoms"),
                                ("516", "unique patterns")]):
        stat(sl, v, l, M + i * (sw + gap), TOP, sw, 1.4)

    y, lw, ph = TOP + 1.66, 6.0, BOT - (TOP + 1.66)
    _rect(sl, M, y, lw, ph, WHITE, line=LINE, radius=0.05)
    _txt(sl, "MERGED FROM THREE SOURCES", M + 0.3, y + 0.32, lw - 0.6, 0.26, 11.5,
         color=TEAL, bold=True)
    for i, (n, src) in enumerate([("41", "public symptom matrix"), ("174", "SymCat"),
                                  ("29", "curated clinical patterns")]):
        ly = y + 0.88 + i * 0.62
        _txt(sl, n, M + 0.3, ly, 0.85, 0.4, 20, color=TEAL_BRIGHT, bold=True)
        _txt(sl, src, M + 1.25, ly + 0.06, lw - 1.55, 0.36, 16)
    _txt(sl, "Curated patterns cover injuries and musculoskeletal complaints "
             "that no public dataset carries.", M + 0.3, y + 2.86, lw - 0.6, 0.6, 13.5,
         color=MUTED, spacing=1.25)

    x, rw = M + lw + 0.32, CW - lw - 0.32
    _rect(sl, x, y, rw, ph, CARD, line=LINE, radius=0.05)
    _txt(sl, "WHY NOT NAIVE BAYES", x + 0.3, y + 0.32, rw - 0.6, 0.26, 11.5,
         color=AMBER, bold=True)
    _txt(sl, "Naive Bayes treats an unmentioned symptom as confirmed absent.",
         x + 0.3, y + 0.82, rw - 0.6, 0.9, 18, color=INK, bold=True, spacing=1.25)
    _txt(sl, "That fit once ranked a head cold as AIDS at 46%. IDF-weighted overlap "
             "treats silence as unknown and weights rare symptoms higher.",
         x + 0.3, y + 1.9, rw - 0.6, 1.2, 15, spacing=1.35)
    return sl


def s13_models(prs, s):
    sl = page(prs, "Risk Model Selection", 13, "Results · 1 of 3")
    table(sl, table_rows(s), M, TOP, 6.55, max_h=2.9,
          col_widths=[2.0, 2.05, 1.25, 1.25])
    fig = FIG / "fig_algorithm_contest.png"
    if fig.exists():
        picture_fit(sl, fig, M + 6.9, TOP - 0.06, CW - 6.9, 3.02)
    notes = body_lines(s)[:2]
    row_list(sl, notes, TOP + 3.3, BOT, accent=TEAL)
    return sl


def s14_llm(prs, s):
    sl = page(prs, "Language Model Benchmark", 14, "Results · 2 of 3")
    table(sl, table_rows(s), M, TOP, 6.55, max_h=1.9)
    fig = FIG / "fig_llm_benchmark.png"
    if fig.exists():
        picture_fit(sl, fig, M + 6.9, TOP - 0.06, CW - 6.9, 3.02)
    _rect(sl, M, TOP + 2.2, 6.55, 0.8, TEAL, radius=0.05)
    _txt(sl, "gemma3:4b selected   ·   100 / 100", M + 0.3, TOP + 2.2, 5.95, 0.8, 17,
         color=WHITE, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    row_list(sl, body_lines(s)[-2:], TOP + 3.3, BOT, accent=AMBER)
    return sl


def s15_shap(prs, s):
    sl = page(prs, "Explainability with SHAP: a Worked CKD Example", 15,
              "How a prediction is explained")
    items = body_lines(s)
    bullet_column(sl, items[:-1], M, TOP + 0.08, 5.95, BOT - 0.95)
    callout(sl, items[-1], BOT - 0.8, tone=INK, h=0.8)
    x, pw = M + 6.35, CW - 6.35
    fig = FIG / "fig_shap_ckd.png"
    if fig.exists():
        from PIL import Image
        iw, ih = Image.open(fig).size
        ph = (pw - 0.32) * ih / iw                      # height the figure will actually take
        cy = TOP + (3.95 - ph - 0.32) / 2
        _rect(sl, x, cy, pw, ph + 0.32, CARD, line=LINE, radius=0.05)
        sl.shapes.add_picture(str(fig), Inches(x + 0.16), Inches(cy + 0.16),
                              width=Inches(pw - 0.32))
    return sl


def s16_prescribing(prs, s):
    sl = page(prs, "Deterministic Prescribing and Urgent-Care Escalation", 16, "Safety rules")
    items = body_lines(s)

    lead = next(i for i in items if i.startswith("Prescribing is deterministic"))
    _rect(sl, M, TOP, CW, 0.72, CARD, line=LINE, radius=0.05)
    _rect(sl, M, TOP, 0.05, 0.72, TEAL)
    _txt(sl, lead, M + 0.32, TOP, CW - 0.64, 0.72, 17, color=INK, bold=True,
         anchor=MSO_ANCHOR.MIDDLE)

    _txt(sl, "CURATED INDIAN OTC FORMULARY", M, TOP + 0.98, 6.0, 0.26, 11.5,
         color=TEAL, bold=True)
    drugs = ["Dolo 650", "Brufen 400", "Sinarest", "Strepsils", "Cetzine",
             "Digene", "Electral", "Otrivin", "Volini"]
    cy = chip_row(sl, drugs, M, TOP + 1.32, CW, 0.56)

    rules = [i for i in items if i.startswith("Hard rule")]
    y, rw = cy + 0.46, (CW - 0.3) / 2
    for i, r in enumerate(rules[:2]):
        x = M + i * (rw + 0.3)
        _rect(sl, x, y, rw, 1.02, WHITE, line=TEAL, radius=0.06, line_w=1.4)
        _txt(sl, "HARD RULE", x + 0.3, y + 0.2, 2.0, 0.24, 11, color=TEAL, bold=True)
        rule = r.split(":", 1)[1].strip()
        _txt(sl, rule, x + 0.3, y + 0.5, rw - 0.6, 0.44,
             fit_size(rule, rw - 0.6, 0.44, [16, 15, 14]), color=INK, bold=True)

    ey = y + 1.26
    _rect(sl, M, ey, CW, 1.5, AMBER, radius=0.05)
    _txt(sl, "ESCALATION  ·  THESE BYPASS PRESCRIBING ENTIRELY", M + 0.32, ey + 0.18,
         6.5, 0.26, 11, color=RGBColor(0xF6, 0xE3, 0xD2), bold=True)
    red_flags = ["Appendicitis", "Meningitis", "Sepsis", "Stroke",
                 "Suspected fracture", "Concussion"]
    chip_row(sl, red_flags, M + 0.32, ey + 0.52, CW - 0.64, 0.44,
             fill=RGBColor(0xB3, 0x72, 0x33), border=None, text=WHITE, size=13, gap=0.14)

    _txt(sl, "They return urgent-care advice instead of any medication.",
         M + 0.32, ey + 1.06, CW - 0.64, 0.32, 14, color=RGBColor(0xF9, 0xEC, 0xE0))
    return sl


def s17_results(prs, s):
    sl = page(prs, "Results: Triage Accuracy and System Validation", 17, "Results · 3 of 3")
    lw = 7.15
    sw, gap = (lw - 2 * 0.22) / 3, 0.22
    for i, (v, l) in enumerate([("78.4%", "top-3 from 2 symptoms"),
                                ("88.8%", "top-3 from 3 symptoms"),
                                ("90.4%", "top-3 from 4 symptoms")]):
        last = i == 2
        stat(sl, v, l, M + i * (sw + gap), TOP, sw, 1.72,
             tone=TEAL if last else WHITE, value_color=WHITE if last else TEAL,
             border=TEAL if last else LINE, label_color=PALE if last else BODY)
    fig = FIG / "fig_triage_accuracy.png"
    if fig.exists():
        picture_fit(sl, fig, M + lw + 0.32, TOP, CW - lw - 0.32, BOT - TOP)
    row_list(sl, body_lines(s)[1:4], TOP + 2.12, BOT, accent=TEAL, width=lw)
    return sl


def s18_limitations(prs, s):
    sl = page(prs, "Limitations, Stated Honestly", 18, "What this system cannot do")
    cards(sl, body_lines(s), cols=2, top=TOP, bottom=BOT)
    return sl


def s19_future(prs, s):
    sl = page(prs, "Future Scope", 19, "What comes next")
    cards(sl, body_lines(s), cols=2, top=TOP, bottom=BOT, numbered=True)
    return sl


def s20_conclusion(prs, s):
    sl = page(prs, "Conclusion", 20, "Wrapping up", dark=True)
    items = [b for b in body_lines(s) if not b.lower().startswith("thank you")]
    lw, gh = 8.85, 0.82
    for i, it in enumerate(items[:5]):
        y = TOP + i * (gh + 0.13)
        _rect(sl, M, y, lw, gh, PANEL, line=EDGE, radius=0.06)
        _rect(sl, M, y, 0.05, gh, TEAL_BRIGHT)
        _txt(sl, it, M + 0.32, y, lw - 0.62, gh,
             fit_size(it, lw - 0.62, gh - 0.1, [16, 15, 14]), color=PALE,
             anchor=MSO_ANCHOR.MIDDLE)
    x, pw = M + lw + 0.32, CW - lw - 0.32
    _rect(sl, x, TOP, pw, 4.62, TEAL, radius=0.05)
    _txt(sl, "Thank you", x, TOP + 1.6, pw, 0.8, 34, color=WHITE, bold=True,
         align=PP_ALIGN.CENTER)
    _txt(sl, "Questions welcome", x, TOP + 2.5, pw, 0.4, 16, color=PALE,
         align=PP_ALIGN.CENTER)
    return sl


LAYOUTS = {1: s01_title, 2: s02_agenda, 3: s03_problem, 4: s04_motivation,
           5: s05_lit_checkers, 6: s06_lit_ml, 7: s07_gap, 8: s08_objectives,
           9: s09_architecture, 10: s10_datasets, 11: s11_wall, 12: s12_triage,
           13: s13_models, 14: s14_llm, 15: s15_shap, 16: s16_prescribing,
           17: s17_results, 18: s18_limitations, 19: s19_future, 20: s20_conclusion}


def build():
    if not SRC.exists():
        print(f"  ! {SRC} not found"); return
    slides = parse(SRC.read_text())
    prs = Presentation(); prs.slide_width, prs.slide_height = W, H
    for s in slides:
        render = LAYOUTS.get(s["n"])
        if render is None:
            print(f"  ! no layout for slide {s['n']}"); continue
        render(prs, s)
    prs.save(OUT)
    print(f"  deck: {OUT.name}  {len(prs.slides._sldIdLst)} slides "
          f"({OUT.stat().st_size//1024} KB)")


if __name__ == "__main__":
    build()
