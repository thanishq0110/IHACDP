"""Assemble the capstone report: Markdown chapters -> paginated HTML -> PDF.

Chrome renders the PDF, so there is no LaTeX or LibreOffice dependency. Page
numbers come from Chrome's own print footer.
"""
from __future__ import annotations
import re, subprocess, sys
from pathlib import Path
import markdown

ROOT = Path(__file__).resolve().parents[1]
DOCS, FIG = ROOT / "docs", ROOT / "docs" / "figures"
CHAP = DOCS / "chapters"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

TITLE = "INTELLIGENT HEALTHCARE ANALYTICS AND CLINICAL DECISION SUPPORT PLATFORM"
DEGREE = "Bachelor of Technology in Computer Science and Engineering"
TEAM = [("SHIVA CHARAN AMBATI", "23BRS1203"),
        ("ATLA ABHINAV KARTHIK REDDY", "23BCE1085"),
        ("KOTHA RITHVIK REDDY", "23BCE1582")]
GUIDE = "Dr. Vivekanandan M"
GUIDE_ID = "53646"
MONTH = "April, 2026"
VIT_LOGO = ROOT / "docs" / "assets" / "vit-logo.png"

CHAPTERS = [
    ("ch1", "CHAPTER 1: INTRODUCTION"),
    ("ch2", "CHAPTER 2: LITERATURE SURVEY"),
    ("ch3", "CHAPTER 3: SYSTEM METHODOLOGY"),
    ("ch4", "CHAPTER 4: IMPLEMENTATION"),
    ("ch5", "CHAPTER 5: RESULTS AND DISCUSSION"),
    ("ch6", "CHAPTER 6: CONCLUSION AND FUTURE WORK"),
]

FIGURES = [
    ("3.1", "System architecture and request path", "fig_architecture.png"),
    ("3.2", "Provenance of the triage conditions", "fig_triage_sources.png"),
    ("4.1", "Landing page of the deployed platform", "ui_index.png"),
    ("4.2", "Consultation and generated prescription", "ui_consultation.png"),
    ("4.3", "Model performance page", "ui_models.png"),
    ("5.1", "ROC curves on the held-out test split", "fig_roc_curves.png"),
    ("5.2", "Three-algorithm contest per disease", "fig_algorithm_contest.png"),
    ("5.3", "Test-split performance of the selected models", "fig_metrics_panel.png"),
    ("5.4", "Triage recovery from partial information", "fig_triage_accuracy.png"),
    ("5.5", "Language model benchmark and safety failures", "fig_llm_benchmark.png"),
]

CSS = """
@page { size: A4; margin: 25mm 22mm 22mm 28mm; }
* { box-sizing: border-box; }
body { font-family: "Times New Roman", Times, serif; font-size: 12pt; line-height: 1.6;
       color: #000; margin: 0; text-align: justify; }
h1, h2, h3, h4 { font-family: "Times New Roman", Times, serif; text-align: left;
                 page-break-after: avoid; }
h1 { font-size: 16pt; font-weight: bold; text-align: center; margin: 0 0 22pt;
     text-transform: uppercase; }
h2 { font-size: 13.5pt; font-weight: bold; margin: 20pt 0 8pt; }
h3 { font-size: 12.5pt; font-weight: bold; margin: 14pt 0 6pt; }
p  { margin: 0 0 10pt; text-indent: 0; }
ul, ol { margin: 0 0 10pt 0; padding-left: 22pt; }
li { margin-bottom: 4pt; }
table { border-collapse: collapse; width: 100%; margin: 10pt 0 14pt; font-size: 10.5pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #000; padding: 5pt 7pt; text-align: left; vertical-align: top; }
th { background: #eaeaea; font-weight: bold; }
table.wide { font-size: 8pt; }
table.wide th, table.wide td { padding: 2pt 3pt; }
code, pre { font-family: "Courier New", monospace; font-size: 9.5pt; }
pre { border: 1px solid #999; padding: 8pt; background: #f7f7f7; white-space: pre;
      overflow-x: hidden; page-break-inside: avoid; line-height: 1.35; }
.page { page-break-after: always; }
.nobreak { page-break-inside: avoid; }
.cover { text-align: center; padding-top: 3mm; }
.cover .pre { font-style: italic; font-size: 12pt; margin: 0 0 15mm; }
.cover .pre.mid { margin-bottom: 18mm; }
.cover .title { font-size: 17pt; font-weight: bold; line-height: 1.45; margin: 0 8mm 20mm;
                text-transform: uppercase; }
.cover .deg { font-size: 13.5pt; font-weight: bold; line-height: 1.5; margin: 0 10mm 12mm; }
.cover .by { font-style: italic; margin-bottom: 15mm; }
.cover .names { font-size: 13pt; font-weight: bold; line-height: 1.8; margin-bottom: 20mm; }
.cover .inst { font-size: 15pt; font-weight: bold; letter-spacing: .5pt; margin-top: 3mm; }
.cover .sub { font-size: 11.5pt; margin-top: 4pt; }
.cover .school { font-size: 13pt; font-weight: bold; margin-top: 5mm; }
.cover .date { margin-top: 9mm; }
.cover img.mark { width: 26mm; }
.cover img.mark.vit { width: 100mm; }
.centre-head { text-align: center; font-size: 14pt; font-weight: bold; text-decoration: underline;
               margin: 0 0 22pt; letter-spacing: .5pt; }
.sig { margin-top: 34pt; }
.sig-row { display: flex; justify-content: space-between; margin-top: 30pt; }
.toc-line { display: flex; align-items: baseline; margin-bottom: 5pt; font-size: 11.5pt; }
.toc-line .t { white-space: nowrap; }
.toc-line .dots { flex: 1; border-bottom: 1px dotted #666; margin: 0 5pt; transform: translateY(-3px); }
.toc-line.lvl0 { font-weight: bold; margin-top: 12pt; text-transform: uppercase; }
.toc-line.lvl1 { padding-left: 16pt; }
figure { margin: 14pt 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; max-height: 108mm; border: 1px solid #bbb; }
figcaption { font-size: 10.5pt; margin-top: 6pt; font-style: italic; }
.abstract p:first-of-type { margin-top: 0; }
"""


def md(text: str) -> str:
    html = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    return _tag_wide_tables(html)


def _tag_wide_tables(html: str) -> str:
    """Mark many-column tables so they get a tighter style.

    An 11-column table's min-content width is about 217mm, well over the 160mm
    text block. Chrome does not clip that -- it silently scales the WHOLE
    document down to fit, which is how the body text ended up at 8.6pt.
    """
    def fix(m):
        table = m.group(0)
        first_row = re.search(r"<tr>.*?</tr>", table, re.S)
        cols = len(re.findall(r"<t[hd][ >]", first_row.group(0))) if first_row else 0
        return table.replace("<table>", '<table class="wide">', 1) if cols >= 8 else table
    return re.sub(r"<table>.*?</table>", fix, html, flags=re.S)


def cover() -> str:
    names = "<br>".join(f"{n} ({r})" for n, r in TEAM)
    if VIT_LOGO.exists():
        # the official lockup already carries the university name and the UGC line
        mark = ('<img class="mark vit" src="assets/vit-logo.png">'
                '<div class="inst">CHENNAI</div>')
    else:
        mark = ('<img class="mark" src="figures/../../frontend/assets/logo.png">'
                '<div class="inst">VELLORE INSTITUTE OF TECHNOLOGY</div>'
                '<div class="sub">(Deemed to be University under section 3 of UGC Act, 1956)</div>'
                '<div class="sub">CHENNAI</div>')
    return f"""<div class="page cover">
<div class="pre">A project report on</div>
<div class="title">{TITLE}</div>
<div class="pre mid">Submitted in partial fulfillment for the award of the degree of</div>
<div class="deg">{DEGREE}</div>
<div class="by">by</div>
<div class="names">{names}</div>
{mark}
<div class="school">SCHOOL OF COMPUTER SCIENCE AND ENGINEERING</div>
<div class="date">{MONTH}</div>
</div>"""


def declaration() -> str:
    names = ", ".join(f"{n} ({r})" for n, r in TEAM)
    return f"""<div class="page">
<div class="centre-head">DECLARATION</div>
<p>We hereby declare that the thesis entitled &ldquo;{TITLE.title()}&rdquo; submitted by
{names}, for the award of the degree of {DEGREE}, Vellore Institute of Technology, Chennai
is a record of bonafide work carried out by us under the supervision of {GUIDE}.</p>
<p>We further declare that the work reported in this thesis has not been submitted and will
not be submitted, either in part or in full, for the award of any other degree or diploma in
this institute or any other institute or university.</p>
<div class="sig">
<p>Place: Chennai</p><p>Date:</p>
<p style="margin-top:34pt; text-align:right">Signature of the Candidates</p>
</div></div>"""


def certificate() -> str:
    names = ", ".join(f"{n} ({r})" for n, r in TEAM)
    return f"""<div class="page">
<div class="centre-head">CERTIFICATE</div>
<p>This is to certify that the report entitled &ldquo;{TITLE.title()}&rdquo; is prepared and
submitted by {names} to Vellore Institute of Technology, Chennai, in partial fulfillment of the
requirement for the award of the degree of {DEGREE} is a bonafide record carried out under my
guidance. The project fulfills the requirements as per the regulations of this University and in
my opinion meets the necessary standards for submission. The contents of this report have not
been submitted and will not be submitted either in part or in full, for the award of any other
degree or diploma and the same is certified.</p>
<div class="sig">
<p>Signature of the Guide:</p><p>Name: {GUIDE} ({GUIDE_ID})</p><p>Date:</p>
<div class="sig-row"><div><p>Signature of the Examiner</p><p>Name:</p><p>Date:</p></div>
<div><p>Signature of the Examiner</p><p>Name:</p><p>Date:</p></div></div>
</div></div>"""


def toc(entries) -> str:
    rows = "".join(
        f'<div class="toc-line lvl{lvl}"><span class="t">{t}</span>'
        f'<span class="dots"></span><span>{p}</span></div>' for lvl, t, p in entries)
    return f'<div class="page"><div class="centre-head">TABLE OF CONTENTS</div>{rows}</div>'


def listing(head, rows) -> str:
    body = "".join(f'<div class="toc-line"><span class="t">{a}</span>'
                   f'<span class="dots"></span><span>{b}</span></div>' for a, b in rows)
    return f'<div class="page"><div class="centre-head">{head}</div>{body}</div>'


def build(numbers: dict | None = None) -> list[str]:
    numbers = numbers or {}
    parts = [cover(), declaration(), certificate()]

    abstract = (CHAP / "abstract.md")
    if abstract.exists():
        parts.append(f'<div class="page abstract"><div class="centre-head">ABSTRACT</div>'
                     f'{md(abstract.read_text())}</div>')
    ack = (CHAP / "acknowledgement.md")
    if ack.exists():
        parts.append(f'<div class="page"><div class="centre-head">ACKNOWLEDGEMENT</div>'
                     f'{md(ack.read_text())}</div>')

    # table of contents from the chapter headings themselves
    entries, tables = [], []
    for key, heading in CHAPTERS:
        f = CHAP / f"{key}.md"
        if not f.exists():
            continue
        entries.append((0, heading, numbers.get(heading, "")))
        for m in re.finditer(r"^##\s+([\d.]+\s+.*)$", f.read_text(), re.M):
            sec = m.group(1).strip()
            entries.append((1, sec, numbers.get(sec, "")))
        for m in re.finditer(r"^\s*\*?\*?Table\s+([\d.]+)[:.]?\s*([^*\n]+)", f.read_text(), re.M):
            lbl = f"Table {m.group(1)}: {m.group(2).strip()}"
            tables.append((lbl, numbers.get(lbl, "")))
    entries.append((0, "REFERENCES", numbers.get("REFERENCES", "")))
    parts.append(toc(entries))
    parts.append(listing("LIST OF FIGURES",
                         [(f"Fig {n} {c}", numbers.get(f"Fig {n} {c}", ""))
                          for n, c, _ in FIGURES]))
    if tables:
        parts.append(listing("LIST OF TABLES", tables[:20]))

    for key, heading in CHAPTERS:
        f = CHAP / f"{key}.md"
        if not f.exists():
            print(f"  ! missing {f.name}")
            continue
        body = md(f.read_text())
        # drop in the figures that belong to this chapter
        num = key[-1]
        figs = "".join(
            f'<figure><img src="figures/{src}"><figcaption>Fig {n} {cap}</figcaption></figure>'
            for n, cap, src in FIGURES if src and n.startswith(num))
        parts.append(f'<div class="page"><h1>{heading}</h1>{body}{figs}</div>')

    refs = CHAP / "references.md"
    if refs.exists():
        parts.append(f'<div class="page"><h1>REFERENCES</h1>{md(refs.read_text())}</div>')

    html = (f'<!doctype html><html><head><meta charset="utf-8">'
            f'<title>{TITLE}</title><style>{CSS}</style></head><body>'
            + "".join(parts) + "</body></html>")
    out_html = DOCS / "IHACDP_Report.html"
    out_html.write_text(html)
    print(f"  html: {out_html.name} ({len(html)//1024} KB)")

    pdf = DOCS / "IHACDP_Report.pdf"
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", out_html.as_uri()],
                   check=False, capture_output=True, timeout=300)
    if pdf.exists():
        print(f"  pdf : {pdf.name} ({pdf.stat().st_size//1024} KB)")
    else:
        print("  ! PDF generation failed")
    return {"headings": [e[1] for e in entries],
            "figures": [f"Fig {n} {c}" for n, c, _ in FIGURES],
            "tables": [t[0] for t in tables[:20]]}


def resolve_pages(groups: dict[str, list[str]]) -> dict[str, int]:
    """Look up the printed page each heading, figure and table landed on.

    Front matter repeats these labels, so the search starts after the lists and
    each label is matched only once, in document order.
    """
    from pypdf import PdfReader
    pages = [re.sub(r"\s+", " ", (pg.extract_text() or "")).lower()
             for pg in PdfReader(DOCS / "IHACDP_Report.pdf").pages]
    body_start = next((i for i, t in enumerate(pages)
                       if "chapter 1: introduction" in t and "1.2 motivation" not in t), 0)
    found = {}
    for labels in groups.values():           # each group scans from the body start
        cursor = body_start
        for label in labels:
            needle = re.sub(r"\s+", " ", label).strip().lower()
            for i in range(cursor, len(pages)):
                if needle in pages[i]:
                    found[label] = i + 1
                    cursor = i
                    break
    return found


if __name__ == "__main__":
    groups = build()
    hits = resolve_pages(groups)
    total = sum(len(v) for v in groups.values())
    print(f"  toc : resolved {len(hits)}/{total} page numbers")
    for name, labels in groups.items():
        gone = [l for l in labels if l not in hits]
        if gone:
            print(f"  ! no page found for {len(gone)} {name}: {gone[0][:50]}...")
    build(hits)
