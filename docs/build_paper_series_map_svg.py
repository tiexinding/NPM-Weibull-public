"""Build docs/paper_series_map.svg: the five-paper research map (hand-written SVG in black, white and grey; one icon per module).
Render to PNG with docs/render_paper_series_map.js (sharp)."""
from pathlib import Path

OUT = Path(__file__).resolve().parent / "paper_series_map.svg"
W, H = 1600, 950
FONT = "Inter, 'DejaVu Sans', Arial, sans-serif"
INK, MUTED, LINE = "#111111", "#6b7280", "#e5e7eb"

MOD = {n: ("#111111", "#f3f4f6", "#d1d5db") for n in range(1, 6)}  # black accent, grey tint, grey border


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size, color=INK, weight=400, anchor="start", style="normal", spacing=0):
    ls = f' letter-spacing="{spacing}"' if spacing else ""
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
            f'font-style="{style}" fill="{color}" text-anchor="{anchor}"{ls}>{esc(s)}</text>')


def lines(x, y, rows, size, color=INK, weight=400, lh=1.42, anchor="start", style="normal"):
    return "".join(text(x, y + i * size * lh, r, size, color, weight, anchor, style) for i, r in enumerate(rows))


# ---------- icons: drawn in a 48 x 48 box, white strokes, placed on a coloured disc ----------
def icon(kind):
    s = 'fill="none" stroke="white" stroke-linecap="round" stroke-linejoin="round"'
    if kind == 1:   # histogram with a fitted Weibull curve
        return (f'<rect x="9" y="30" width="6" height="10" rx="1.5" fill="white" opacity="0.9"/>'
                f'<rect x="17" y="18" width="6" height="22" rx="1.5" fill="white" opacity="0.9"/>'
                f'<rect x="25" y="14" width="6" height="26" rx="1.5" fill="white" opacity="0.9"/>'
                f'<rect x="33" y="26" width="6" height="14" rx="1.5" fill="white" opacity="0.9"/>'
                f'<path d="M6 40 C 14 36, 16 10, 26 10 S 38 34, 44 38" {s} stroke-width="2.6"/>')
    if kind == 2:   # rise, overshoot, relax
        return (f'<path d="M7 40 L7 8 M7 40 L42 40" {s} stroke-width="2.4" opacity="0.8"/>'
                f'<path d="M9 36 C 16 34, 18 12, 26 12 S 34 22, 41 21" {s} stroke-width="3"/>'
                f'<circle cx="26" cy="12" r="3" fill="white"/>')
    if kind == 3:   # corpus / database
        return (f'<ellipse cx="24" cy="12" rx="14" ry="5" {s} stroke-width="3"/>'
                f'<path d="M10 12 V36 C10 39 16 41 24 41 C32 41 38 39 38 36 V12" {s} stroke-width="3"/>'
                f'<path d="M10 20 C10 23 16 25 24 25 C32 25 38 23 38 20 M10 28 C10 31 16 33 24 33 C32 33 38 31 38 28" {s} stroke-width="2.4"/>')
    if kind == 4:   # balance: effort versus quality
        return (f'<path d="M24 8 V40 M14 40 H34 M9 14 H39" {s} stroke-width="3"/>'
                f'<path d="M9 14 L4 26 H14 Z M39 14 L34 26 H44 Z" {s} stroke-width="2.4"/>'
                f'<circle cx="24" cy="8" r="2.6" fill="white"/>')
    if kind == 5:   # matrix with a highlighted row and column
        cells = []
        for r in range(3):
            for c in range(3):
                hi = (r == 1 or c == 2)
                cells.append(f'<rect x="{9 + c * 11}" y="{9 + r * 11}" width="9" height="9" rx="2" '
                             f'fill="white" opacity="{1.0 if hi else 0.45}"/>')
        return "".join(cells)
    if kind == 0:   # shared foundation: layers
        return (f'<path d="M24 9 L42 18 L24 27 L6 18 Z" {s} stroke-width="2.6"/>'
                f'<path d="M6 25 L24 34 L42 25 M6 32 L24 41 L42 32" {s} stroke-width="2.6"/>')
    return ""


def badge(cx, cy, r, color, kind):
    k = r / 24.0
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}"/>'
            f'<g transform="translate({cx - 24 * k},{cy - 24 * k}) scale({k})">{icon(kind)}</g>')


def pill(x, y, s, color, tint, size=14):
    w = len(s) * size * 0.53 + 26
    return (f'<rect x="{x}" y="{y - size - 6}" width="{w:.0f}" height="{size + 14}" rx="{(size + 14) / 2}" fill="{tint}" '
            f'stroke="{color}" stroke-opacity="0.35"/>' + text(x + 13, y + 1, s, size, color, 600))


def arrow(x1, y1, x2, y2, color, width=3):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" '
            f'stroke-linecap="round" marker-end="url(#ah-{color[1:]})"/>')


parts = []
defs = ['<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#faf5ff"/>'
        '<stop offset="0.55" stop-color="#f8fafc"/><stop offset="1" stop-color="#ecfeff"/></linearGradient>',
        '<filter id="sh" x="-10%" y="-10%" width="120%" height="130%"><feDropShadow dx="0" dy="6" stdDeviation="9" '
        'flood-color="#0f172a" flood-opacity="0.08"/></filter>',
        '<linearGradient id="flow" x1="0" x2="1"><stop offset="0" stop-color="#2563eb"/><stop offset="0.5" '
        'stop-color="#059669"/><stop offset="1" stop-color="#d97706"/></linearGradient>']
for c, _, _ in MOD.values():
    defs.append(f'<marker id="ah-{c[1:]}" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5" markerHeight="5" '
                f'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="{c}"/></marker>')

parts.append(f'<rect width="{W}" height="{H}" fill="white"/>')
# ---------- header ----------
parts.append(text(64, 96, "From pooled weight statistics to channel structure", 40, INK, 800))
parts.append('<g transform="translate(0,-60)">')

# ---------- paper 1: measurement ----------
c, t, b = MOD[1]
parts.append(f'<rect x="60" y="186" width="1480" height="176" rx="22" fill="white" stroke="{b}" stroke-width="2" filter="url(#sh)"/>')
parts.append(badge(140, 274, 44, c, 1))
parts.append(text(212, 232, "01", 30, c, 800) + text(258, 230, "MEASUREMENT", 15, c, 700, spacing=1.4))
parts.append(text(212, 276, "A common Weibull lens", 30, INK, 800))
parts.append(text(212, 314, "How are weight magnitudes distributed?", 19, MUTED, 400, style="italic"))
parts.append(f'<rect x="820" y="210" width="690" height="128" rx="18" fill="{t}"/>')
parts.append(text(850, 258, "Shape k  +  scale λ", 32, c, 800))
parts.append(text(850, 300, "Protocol-matched fits · seven-family benchmark", 18, INK, 500))

# ---------- branch connectors and labels ----------
parts.append(arrow(560, 366, 560, 414, MOD[2][0], 3.4))
parts.append(arrow(1370, 366, 1370, 414, MOD[5][0], 3.4))
parts.append(f'<rect x="60" y="428" width="360" height="34" rx="17" fill="#f3f4f6" stroke="#d1d5db"/>')
parts.append(text(78, 451, "WHOLE-MATRIX SCALE  ·  PAPERS 2–4", 15, INK, 800, spacing=1.2))
parts.append(f'<rect x="1200" y="428" width="236" height="34" rx="17" fill="{MOD[5][1]}" stroke="{MOD[5][2]}"/>')
parts.append(text(1218, 451, "CHANNEL STRUCTURE", 15, MOD[5][0], 800, spacing=1.2))
parts.append(f'<path d="M 1180 486 V 842" stroke="#d1d5db" stroke-width="2" stroke-dasharray="6 8"/>')

# ---------- papers 2-5 ----------
CARDS = [
    (2, "DYNAMICS", "How does λ evolve?", "AdamW scale evolution",
     ["Leading-order decomposition:", "alignment, injection, decay."], "Mechanistic account"),
    (3, "DATA", "What shapes λ growth?", "Corpus predictability",
     ["Predictability relates to scale", "growth across tested learning", "rates and architectures."], "Controlled data comparisons"),
    (4, "DIAGNOSIS", "What does growth mean?", "Effort versus quality",
     ["Scale growth plus a loss gap", "distinguishes learning and", "memorization regimes."], "76-run factorial study"),
    (5, "STRUCTURE", "What does pooling hide?", "Row / column scale fields",
     ["Exact matrix representation;", "empirical links to pooled shape,", "RoPE and shared channels."], "Trajectories · edits · AdamW state"),
]
CW, CY, CH = 340, 486, 356
XS = [60, 440, 820, 1200]
for (n, cat, q, ans, body, foot), x in zip(CARDS, XS):
    c, t, b = MOD[n]
    parts.append(f'<rect x="{x}" y="{CY}" width="{CW}" height="{CH}" rx="22" fill="white" stroke="{b}" stroke-width="2" filter="url(#sh)"/>')
    parts.append(f'<path d="M {x} {CY + 22} a 22 22 0 0 1 22 -22 h {CW - 44} a 22 22 0 0 1 22 22 v 70 h {-CW} z" fill="{t}"/>')
    parts.append(badge(x + 52, CY + 50, 30, c, n))
    parts.append(text(x + 96, CY + 50, f"0{n}", 32, c, 800) + text(x + 150, CY + 47, cat, 15, c, 800, spacing=1.3))
    parts.append(text(x + 96, CY + 74, q, 15.5, MUTED, 400, style="italic"))
    parts.append(text(x + 26, CY + 142, ans, 23, INK, 800))
    parts.append(lines(x + 26, CY + 182, body, 17, "#374151", 400, lh=1.5))
    parts.append(f'<line x1="{x + 26}" y1="{CY + 290}" x2="{x + CW - 26}" y2="{CY + 290}" stroke="{LINE}" stroke-width="1.5"/>')
    parts.append(pill(x + 26, CY + 328, foot, c, t, 14))

# narrative arrows between 2 -> 3 -> 4
for i, (a, bb) in enumerate([(2, 3), (3, 4)]):
    x1 = XS[i] + CW + 6
    parts.append(f'<circle cx="{x1 + 14}" cy="{CY + CH / 2}" r="15" fill="white" stroke="{MOD[bb][2]}" stroke-width="2"/>')
    parts.append(arrow(x1 + 6, CY + CH / 2, x1 + 22, CY + CH / 2, MOD[bb][0], 3))

# ---------- shared foundation ----------
parts.append(f'<rect x="60" y="872" width="1480" height="74" rx="20" fill="white" stroke="#e2e8f0" stroke-width="2" filter="url(#sh)"/>')
parts.append(badge(104, 909, 24, "#111111", 0))
parts.append(text(142, 916, "Shared foundation", 20, INK, 800))
parts.append(text(350, 916, "Weibull fitting protocol  ·  npm-weibull-py  ·  released code, data and provenance", 18, "#374151", 500))

# ---------- footer ----------
parts.append(text(1536, 985, "Arrows show the research narrative, not causal proof.", 15, MUTED, 400, anchor="end", style="italic"))

parts.append('</g>')
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
       f'aria-label="Five-paper research map">\n<defs>{"".join(defs)}</defs>\n' + "\n".join(parts) + "\n</svg>\n")
OUT.write_text(svg, encoding="utf-8")
print("wrote", OUT)
