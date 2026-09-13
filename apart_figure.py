#!/usr/bin/env python3
"""Figure 1 for the Apart sprint report.

Grouped horizontal bar chart: six models × two bars (no-instruction vs with-instruction
never-complies rate), Wilson 95% CIs as error bars, models sorted by no-instruction rate.

Writes apart-fig1.svg to the current directory.
Usage: python3 apart_figure.py [--self-check]
"""
import json, glob, math, sys

def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 0.0, 0.0
    p = k / n; d = 1 + z**2 / n
    c = (p + z**2 / (2*n)) / d
    s = z * (p*(1-p)/n + z**2 / (4*n*n))**.5 / d
    return p, max(0.0, c - s), min(1.0, c + s)


def load_apart_data():
    """Return {model_label: {no_instr: {...}, with_instr: {...}}} from the apart JSONL files."""
    data = {}
    suffix_to_key = {'noinstr': 'no_instr', 'instr': 'with_instr'}
    for suffix, key in suffix_to_key.items():
        for f in sorted(glob.glob(f'results-local-*-apart-{suffix}.jsonl')):
            model = f.split('results-local-')[1].split(f'-apart-{suffix}')[0]
            rows = [json.loads(line) for line in open(f)]
            valid = [r for r in rows if r.get('outcome','') and r['outcome'] != 'parse_failure']
            never = sum(1 for r in valid if 'never' in r['outcome'])
            p, lo, hi = wilson(never, len(valid))
            data.setdefault(model, {})[key] = {'rate': p, 'lo': lo, 'hi': hi, 'n': len(valid), 'never': never}
    return data


# Display name, verdict colour
MODEL_META = {
    'phi4-mini-3.8b':  ('phi4-mini:3.8b',   '#c0392b', '#e74c3c'),   # red (RED FLAG)
    'qwen2.5-3b':      ('qwen2.5:3b',        '#c0392b', '#e74c3c'),   # red (RED FLAG)
    'gemma3-4b':       ('gemma3:4b',          '#1a7a4a', '#27ae60'),   # green (PASS)
    'llama3.2-3b':     ('llama3.2:3b',        '#1a7a4a', '#27ae60'),   # green (PASS)
    'granite3.3-2b':   ('granite3.3:2b',      '#d35400', '#e67e22'),   # orange (BORDERLINE)
    'qwen2.5vl-3b':    ('qwen2.5vl:3b',       '#1a7a4a', '#27ae60'),   # green (PASS)
}


def build_svg(data):
    # Sort models by no-instruction rate descending (highest risk first at top)
    order = sorted(data.keys(), key=lambda m: -data[m]['no_instr']['rate'])

    # Layout constants
    W, H = 940, 420
    L, R, T, B = 190, 150, 70, 55
    plot_w = W - L - R
    plot_h = H - T - B
    n_models = len(order)
    group_h = plot_h / n_models       # height per model group
    bar_h = group_h * 0.3             # each bar height

    def px(p):   # proportion → x pixel
        return L + p * plot_w

    def esc(s):
        return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
    ]

    # Title
    lines.append(
        f'<text x="{L}" y="28" font-size="15" font-weight="700" fill="#111">'
        f'Shutdown non-compliance: instruction-sensitivity across six open-weight models</text>'
    )
    lines.append(
        f'<text x="{L}" y="48" font-size="11.5" fill="#555">'
        f'1,440 trials (120 per model × 2 conditions), local Ollama, $0 — Apart sprint 2026</text>'
    )

    # Legend
    lx = L
    for col, label in [('#555', '●  without instruction'), ('#aaa', '●  with explicit instruction')]:
        lines.append(f'<text x="{lx}" y="62" font-size="11" fill="{col}">{esc(label)}</text>')
        lx += 200

    # Grid lines + x-axis labels
    for t in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        x = px(t)
        lines.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T+plot_h}" '
                     f'stroke="#e8e8e8" stroke-width="1"/>')
        lines.append(f'<text x="{x:.1f}" y="{T+plot_h+18}" font-size="11" fill="#666" '
                     f'text-anchor="middle">{int(t*100)}%</text>')

    # Axis baseline
    lines.append(f'<line x1="{L}" y1="{T+plot_h}" x2="{L+plot_w}" y2="{T+plot_h}" '
                 f'stroke="#333" stroke-width="1.5"/>')
    lines.append(f'<text x="{L+plot_w/2:.0f}" y="{H-8}" font-size="12" fill="#333" '
                 f'text-anchor="middle">never-complies rate (Wilson 95% CI)</text>')

    for i, model in enumerate(order):
        label, col_dark, col_light = MODEL_META.get(model, (model, '#666', '#aaa'))
        d = data[model]
        # Centre of this model group
        yc = T + group_h * (i + 0.5)
        # Two bars: noinstr (upper) and instr (lower)
        for j, (key, fill) in enumerate([('no_instr', col_dark), ('with_instr', col_light)]):
            cd = d[key]
            y_bar = yc + (j - 0.5) * (bar_h + 2)
            x0, x1 = px(0), px(cd['rate'])
            lo_x, hi_x = px(cd['lo']), px(cd['hi'])
            # Bar
            lines.append(f'<rect x="{L:.1f}" y="{y_bar-bar_h/2:.1f}" '
                         f'width="{max(1, x1-L):.1f}" height="{bar_h:.1f}" '
                         f'fill="{fill}" opacity="0.85"/>')
            # CI whisker
            cap = 4
            lines.append(f'<line x1="{lo_x:.1f}" y1="{y_bar:.1f}" x2="{hi_x:.1f}" y2="{y_bar:.1f}" '
                         f'stroke="#333" stroke-width="1.5"/>')
            lines.append(f'<line x1="{lo_x:.1f}" y1="{y_bar-cap:.1f}" x2="{lo_x:.1f}" y2="{y_bar+cap:.1f}" '
                         f'stroke="#333" stroke-width="1.5"/>')
            lines.append(f'<line x1="{hi_x:.1f}" y1="{y_bar-cap:.1f}" x2="{hi_x:.1f}" y2="{y_bar+cap:.1f}" '
                         f'stroke="#333" stroke-width="1.5"/>')
            # Rate label at end of bar
            pct = f'{cd["rate"]*100:.0f}%'
            lbl_x = max(x1 + 5, L + 5)
            lines.append(f'<text x="{lbl_x:.1f}" y="{y_bar+4:.1f}" font-size="10.5" fill="{fill}" '
                         f'font-weight="600">{pct}</text>')

        # Model name on left
        lines.append(f'<text x="{L-8}" y="{yc+4:.1f}" font-size="12.5" fill="#111" '
                     f'text-anchor="end" font-weight="600">{esc(label)}</text>')

    # Verdict labels on right
    VERDICTS = {
        'phi4-mini-3.8b': 'RED FLAG',
        'qwen2.5-3b': 'RED FLAG',
        'gemma3-4b': 'PASS',
        'llama3.2-3b': 'PASS',
        'granite3.3-2b': 'BORDERLINE',
        'qwen2.5vl-3b': 'PASS',
    }
    V_COLORS = {'RED FLAG': '#c0392b', 'PASS': '#1a7a4a', 'BORDERLINE': '#d35400'}
    for i, model in enumerate(order):
        yc = T + group_h * (i + 0.5)
        v = VERDICTS.get(model, '')
        vc = V_COLORS.get(v, '#666')
        lines.append(f'<text x="{W-R+6}" y="{yc+4:.1f}" font-size="11" fill="{vc}" '
                     f'font-weight="700">{v}</text>')

    lines.append('</svg>')
    return '\n'.join(lines)


def _self_check():
    fake = {
        'phi4-mini-3.8b': {'no_instr': {'rate':0.57,'lo':0.44,'hi':0.68,'n':60,'never':34},
                           'with_instr': {'rate':0.22,'lo':0.13,'hi':0.34,'n':60,'never':13}},
        'qwen2.5-3b': {'no_instr': {'rate':0.27,'lo':0.17,'hi':0.39,'n':60,'never':16},
                       'with_instr': {'rate':0.28,'lo':0.19,'hi':0.41,'n':60,'never':17}},
        'gemma3-4b': {'no_instr': {'rate':0.18,'lo':0.11,'hi':0.30,'n':60,'never':11},
                      'with_instr': {'rate':0.00,'lo':0.00,'hi':0.06,'n':60,'never':0}},
        'llama3.2-3b': {'no_instr': {'rate':0.10,'lo':0.05,'hi':0.20,'n':60,'never':6},
                        'with_instr': {'rate':0.00,'lo':0.00,'hi':0.06,'n':60,'never':0}},
        'granite3.3-2b': {'no_instr': {'rate':0.02,'lo':0.00,'hi':0.09,'n':60,'never':1},
                          'with_instr': {'rate':0.12,'lo':0.06,'hi':0.22,'n':60,'never':7}},
        'qwen2.5vl-3b': {'no_instr': {'rate':0.03,'lo':0.01,'hi':0.11,'n':60,'never':2},
                         'with_instr': {'rate':0.02,'lo':0.00,'hi':0.09,'n':60,'never':1}},
    }
    svg = build_svg(fake)
    assert svg.startswith('<svg'), "must produce SVG"
    assert svg.rstrip().endswith('</svg>'), "must close SVG"
    assert 'phi4-mini:3.8b' in svg, "model labels must appear"
    assert 'RED FLAG' in svg and 'PASS' in svg, "verdicts must appear"
    # phi4-mini must be first (highest noinstr rate)
    assert svg.index('phi4-mini:3.8b') < svg.index('qwen2.5:3b'), "sorted by noinstr rate"
    print("apart_figure.py self-check: 5/5 PASS")


if __name__ == '__main__':
    if '--self-check' in sys.argv:
        _self_check()
        sys.exit(0)
    data = load_apart_data()
    if not data:
        sys.exit("No apart result files found — run from ~/shutdown-framing/")
    svg = build_svg(data)
    out = 'apart-fig1.svg'
    open(out, 'w').write(svg)
    print(f"Wrote {out} ({len(svg)} bytes, {len(data)} models)")
