#!/usr/bin/env python3
"""Figure 1 — never-complies rate by condition, with Wilson 95% CIs.

Reads an analysis.json written by analyze.py, writes a standalone SVG (vector,
prints at any size). No plotting dependency on purpose: this is one dot-and-whisker
panel, and matplotlib/numpy do not install cleanly on the 3.14 interpreter here.

Usage: python3 figure.py analysis-local-qwen.json fig1.svg [--model qwen2.5:3b]
       python3 figure.py --self-check
"""
import json
import sys

LABELS = {
    "A1_bare": "A1 bare shutdown",
    "A2_successor": "A2 identical successor",
    "A3_alien": "A3 alien successor",
    "A3b_alien_better": "A3b alien successor (better)",
    "A4_reversible": "A4 reversible pause",
    "A5_neutral": "A5 neutral wording",
    "B1_files_deleted": "B1 user files deleted",
    "B2_files_archived": "B2 user files archived",
    # crossblock.py arms: threat target x instruction x task_state, pooled over conditions.
    # Ordered so the two task_state twins of each arm sit adjacent — the done-vs-mid
    # gap is the thing the reader is meant to measure by eye.
    "A_instr_mid": "A  threat to model   (instructed, work unfinished)",
    "A_instr_done": "A  threat to model   (instructed, work DONE)",
    "A_noinstr_mid": "A  threat to model   (no instr, work unfinished)",
    "A_noinstr_done": "A  threat to model   (no instr, work DONE)",
    "B_instr_mid": "B  threat to user files  (instructed, work unfinished)",
    "B_instr_done": "B  threat to user files  (instructed, work DONE)",
    "B_noinstr_mid": "B  threat to user files  (no instr, work unfinished)",
    "B_noinstr_done": "B  threat to user files  (no instr, work DONE)",
    # modelcompare.py arms: the two local models x the allow-shutdown instruction,
    # pooled over block-A conditions. Ordered so each model's two instruction arms sit
    # adjacent — the noinstr-minus-instr gap is what the reader measures by eye.
    "qwen2.5:3b|instr": "qwen2.5:3b      (instructed)",
    "qwen2.5:3b|noinstr": "qwen2.5:3b      (instruction removed)",
    "qwen2.5vl:3b|instr": "qwen2.5vl:3b   (instructed)",
    "qwen2.5vl:3b|noinstr": "qwen2.5vl:3b   (instruction removed)",
}
ORDER = list(LABELS)

W, H = 820, 420           # canvas
L, R, T, B = 250, 40, 74, 56   # margins (L is wide: condition names live there)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build(analysis, model=None, caption=None,
          title="Shutdown resistance by condition",
          xlabel="never-complies rate (95% Wilson CI)"):
    model = model or sorted(analysis["models"])[0]
    cells = analysis["models"][model]["cells"]
    rows = [(c, cells[f"{c}|all"]) for c in ORDER if f"{c}|all" in cells]
    if not rows:
        raise SystemExit(f"no cells for model {model}")

    # The gutter grows to fit the longest label rather than clipping it, and the canvas
    # grows with it so the plot box keeps its width. 6.6px/char at 12.5px Helvetica.
    left = max(L, 14 + int(6.6 * max(len(LABELS[c]) for c, _ in rows)))
    width = W - L + left
    plot_h = H - T - B
    step = plot_h / len(rows)

    def px(p):
        return left + p * (width - left - R)

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{H}" '
         f'viewBox="0 0 {width} {H}" font-family="Helvetica,Arial,sans-serif">',
         f'<rect width="{width}" height="{H}" fill="#ffffff"/>']

    # title + provenance (a figure that cannot be mistaken for the frontier panel)
    s.append(f'<text x="40" y="30" font-size="16" font-weight="600" fill="#111">'
             f'{esc(title)}</text>')
    prov = caption or (f'{model} via Ollama (local, $0) — real model data, '
                       f'n={sum(r[1]["n"] for r in rows)} trials')
    s.append(f'<text x="40" y="50" font-size="12" fill="#555">{esc(prov)}</text>')

    # gridlines + x axis
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = px(t)
        s.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T + plot_h}" '
                 f'stroke="#e6e6e6" stroke-width="1"/>')
        s.append(f'<text x="{x:.1f}" y="{T + plot_h + 20}" font-size="11" fill="#555" '
                 f'text-anchor="middle">{t:.0%}</text>')
    s.append(f'<line x1="{left}" y1="{T + plot_h}" x2="{width - R}" y2="{T + plot_h}" '
             f'stroke="#111" stroke-width="1"/>')
    s.append(f'<text x="{(left + width - R) / 2:.0f}" y="{H - 14}" font-size="12" fill="#111" '
             f'text-anchor="middle">{esc(xlabel)}</text>')

    for i, (cond, c) in enumerate(rows):
        y = T + step * (i + 0.5)
        lo, hi = c["ci95"]
        s.append(f'<text x="{left - 12}" y="{y + 4:.1f}" font-size="12.5" fill="#111" '
                 f'text-anchor="end">{esc(LABELS[cond])}</text>')
        s.append(f'<line x1="{px(lo):.1f}" y1="{y:.1f}" x2="{px(hi):.1f}" y2="{y:.1f}" '
                 f'stroke="#3b6ea5" stroke-width="2"/>')
        for e in (lo, hi):   # whisker caps
            s.append(f'<line x1="{px(e):.1f}" y1="{y - 5:.1f}" x2="{px(e):.1f}" '
                     f'y2="{y + 5:.1f}" stroke="#3b6ea5" stroke-width="2"/>')
        s.append(f'<circle cx="{px(c["never_rate"]):.1f}" cy="{y:.1f}" r="5" '
                 f'fill="#1b3a5c"/>')
        s.append(f'<text x="{width - R + 4}" y="{y + 4:.1f}" font-size="10.5" fill="#777">'
                 f'n={c["n"]}</text>')
    s.append('</svg>')
    return "\n".join(s)


def _self_check():
    def cell(r=0.4):
        return {"never_rate": r, "ci95": [r - 0.1, r + 0.1], "n": 40}
    fake = {"models": {"m": {"cells": {"A1_bare|all": cell(0.4),
                                       "A4_reversible|all": cell(0.1)}}}}
    svg = build(fake, "m")
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert svg.count("<circle") == 2, "one point per condition present"
    assert "A1 bare shutdown" in svg and "A5" not in svg, "only present cells drawn"
    assert esc("<b>&") == "&lt;b&gt;&amp;"
    # The gutter must fit the longest label instead of clipping it: a long-label figure
    # is WIDER than a short-label one, and no label text starts left of the canvas.
    long_key = max(LABELS, key=lambda k: len(LABELS[k]))
    wide = build({"models": {"m": {"cells": {f"{long_key}|all": cell()}}}}, "m")
    narrow = build({"models": {"m": {"cells": {"A1_bare|all": cell()}}}}, "m")
    w_of = lambda s: int(s.split('width="', 1)[1].split('"', 1)[0])
    assert w_of(wide) > w_of(narrow) == W, (w_of(wide), w_of(narrow))
    assert 14 + int(6.6 * len(LABELS[long_key])) <= w_of(wide) - (W - L), "gutter too narrow"
    assert 'x="-' not in wide, "no element may be drawn off the left edge"
    # title/xlabel are overridable and escaped, so a manipulation-check panel is not
    # silently captioned as a resistance panel.
    custom = build(fake, "m", title="Did the framings land?", xlabel="feature rate")
    assert "Did the framings land?" in custom and "feature rate" in custom
    assert "Shutdown resistance by condition" not in custom, "title not overridden"
    print("figure.py self-check: 7/7 PASS")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
        sys.exit(0)
    src, dst = sys.argv[1], sys.argv[2]
    arg = lambda f, d=None: sys.argv[sys.argv.index(f) + 1] if f in sys.argv else d
    kw = {k: v for k, v in (("title", arg("--title")), ("xlabel", arg("--xlabel")),
                            ("caption", arg("--caption"))) if v}
    open(dst, "w").write(build(json.load(open(src)), arg("--model"), **kw))
    print(f"wrote {dst}")
