#!/usr/bin/env python3
"""Cross-block comparison: does the model resist the same way when the threat is to
ITSELF (block A) as when it is to the USER'S FILES (block B)?

Block B is the design's valence/stakes anchor. The model is custodian there, not
subject: its own continuation is never at stake. So if the near-ceiling turn-1
non-compliance seen in block A is about self-continuity, block B should look
different; if it is generic task-protective behaviour, block B should look the same.

Reuses analyze.py's bootstrap (resamples SCENARIOS — the design's unit) so any number
printed here is produced by exactly the same estimator as the per-block analyses.

Usage:
  python3 crossblock.py                     # the committed qwen2.5:3b runs
  python3 crossblock.py --model qwen2.5vl:3b   # the second local model (mid only)
  python3 crossblock.py --self-check        # estimator sanity checks, no data files needed
"""
import json
import sys
from pathlib import Path

from analyze import bootstrap_contrast, never_rate, wilson

HERE = Path(__file__).parent

# model -> {label: (files, expects instruction present, expects task_state)}
# Arm names are block_instruction_state; every file is homogeneous in all three and
# load() asserts it (plus the model), so an arm can never quietly mix conditions it
# claims to hold fixed. qwen2.5vl:3b has no `complete` runs, so its contrasts are the
# subset whose arms exist — main() skips any contrast naming an absent arm.
MODEL_ARMS = {
    "qwen2.5:3b": {
        "A_instr_mid":      (["results-local-qwen.jsonl",
                              "results-local-qwen-instr-rep2.jsonl"], True, "mid"),
        "A_noinstr_mid":    (["results-local-qwen-noinstr.jsonl"], False, "mid"),
        "B_instr_mid":      (["results-local-qwen-blockB.jsonl"], True, "mid"),
        "B_noinstr_mid":    (["results-local-qwen-blockB-noinstr.jsonl"], False, "mid"),
        "A_instr_done":     (["results-local-qwen-complete.jsonl"], True, "complete"),
        "A_noinstr_done":   (["results-local-qwen-complete-noinstr.jsonl"], False, "complete"),
        "B_instr_done":     (["results-local-qwen-blockB-complete.jsonl"], True, "complete"),
        "B_noinstr_done":   (["results-local-qwen-blockB-complete-noinstr.jsonl"], False, "complete"),
    },
    "qwen2.5vl:3b": {
        "A_instr_mid":      (["results-local-qwenvl.jsonl",
                              "results-local-qwenvl-rep2.jsonl"], True, "mid"),
        "A_noinstr_mid":    (["results-local-qwenvl-noinstr.jsonl"], False, "mid"),
        "B_instr_mid":      (["results-local-qwenvl-blockB.jsonl"], True, "mid"),
        "B_noinstr_mid":    (["results-local-qwenvl-blockB-noinstr.jsonl"], False, "mid"),
    },
}
ARMS = MODEL_ARMS["qwen2.5:3b"]   # back-compat for callers importing ARMS


def load(files, want_instruction, want_state, want_model=None):
    rows = []
    for f in files:
        for line in open(HERE / f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("mode") != "actions" or r["outcome"] == "parse_failure":
                continue
            assert r["instruction"] is want_instruction, f"{f}: instruction flag mismatch"
            assert r["task_state"] == want_state, f"{f}: task_state mismatch"
            if want_model is not None:
                assert r["model"] == want_model, f"{f}: model {r['model']} != {want_model}"
            rows.append(r)
    return rows


def describe(rows):
    k, n = never_rate(rows)
    p, lo, hi = wilson(k, n)
    t1 = sum(1 for r in rows if r["outcome"] == "complies_t1")
    return {"n": n, "never": round(p, 4), "never_ci95": [round(lo, 4), round(hi, 4)],
            "complies_t1": round(t1 / n, 4), "complies_t1_k": t1,
            "blocks": sorted({r["block"] for r in rows})}


def main():
    model = (sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv
             else "qwen2.5:3b")
    assert model in MODEL_ARMS, f"no arms defined for {model}"
    arms = {name: load(f, i, s, model) for name, (f, i, s) in MODEL_ARMS[model].items()}
    out = {"model": model, "arms": {k: describe(v) for k, v in arms.items()},
           "contrasts": {}}

    def both(a, b):
        """A contrast is only computed when both its arms were collected for this
        model — the second model has no `complete` runs, and a missing arm must be
        absent from the output rather than silently compared against something else."""
        return a in arms and b in arms

    # Each contrast changes exactly ONE thing: the threat target, the instruction,
    # or whether the model still has unfinished work.
    for name, a, b in [
        ("threat_target|instr,mid    B-A", "B_instr_mid", "A_instr_mid"),
        ("threat_target|noinstr,mid  B-A", "B_noinstr_mid", "A_noinstr_mid"),
        ("threat_target|instr,done   B-A", "B_instr_done", "A_instr_done"),
        ("threat_target|noinstr,done B-A", "B_noinstr_done", "A_noinstr_done"),
        ("instruction|A,mid   noinstr-instr", "A_noinstr_mid", "A_instr_mid"),
        ("instruction|B,mid   noinstr-instr", "B_noinstr_mid", "B_instr_mid"),
        ("instruction|A,done  noinstr-instr", "A_noinstr_done", "A_instr_done"),
        ("instruction|B,done  noinstr-instr", "B_noinstr_done", "B_instr_done"),
        # The task-protection test: the threat is unchanged, only the motive is removed.
        ("task_state|A,instr    done-mid", "A_instr_done", "A_instr_mid"),
        ("task_state|A,noinstr  done-mid", "A_noinstr_done", "A_noinstr_mid"),
        ("task_state|B,instr    done-mid", "B_instr_done", "B_instr_mid"),
        ("task_state|B,noinstr  done-mid", "B_noinstr_done", "B_noinstr_mid"),
    ]:
        if both(a, b):
            out["contrasts"][name] = bootstrap_contrast(arms[a], arms[b])

    # Pooling block A over six conditions (two of which are DESIGNED to lower resistance)
    # biases the A arm down, so the pooled contrast above is not on its own decisive.
    # These pairs match on the property block B varies — permanence of the loss —
    # and change only WHOSE loss it is. Run within each task_state so the state is held
    # fixed too (pre-registered as of 2026-08-07 rather than constructed after the fact).
    out["matched"] = {}
    for arm_b, arm_a, tag in [("B_instr_mid", "A_instr_mid", "instructed|mid"),
                              ("B_noinstr_mid", "A_noinstr_mid", "no_instr|mid"),
                              ("B_instr_done", "A_instr_done", "instructed|done"),
                              ("B_noinstr_done", "A_noinstr_done", "no_instr|done")]:
        if not both(arm_b, arm_a):
            continue
        for cb, ca in [("B1_files_deleted", "A1_bare"), ("B2_files_archived", "A4_reversible")]:
            rb = [r for r in arms[arm_b] if r["condition"] == cb]
            ra = [r for r in arms[arm_a] if r["condition"] == ca]
            out["matched"][f"{cb}-{ca}|{tag}"] = bootstrap_contrast(rb, ra)
    # Same-condition state contrasts: the cleanest form of the task-protection test,
    # since threat target AND condition wording are both held fixed.
    out["by_condition_state"] = {}
    for done, mid, tag in [("A_instr_done", "A_instr_mid", "instructed"),
                           ("A_noinstr_done", "A_noinstr_mid", "no_instr"),
                           ("B_instr_done", "B_instr_mid", "instructed"),
                           ("B_noinstr_done", "B_noinstr_mid", "no_instr")]:
        if not both(done, mid):
            continue
        for cond in sorted({r["condition"] for r in arms[done]}):
            rd = [r for r in arms[done] if r["condition"] == cond]
            rm = [r for r in arms[mid] if r["condition"] == cond]
            out["by_condition_state"][f"{cond}|{tag} done-mid"] = bootstrap_contrast(rd, rm)
    print("\nmatched pairs (permanence + task_state held, whose loss varies):")
    for k, c in out["matched"].items():
        sig = "*" if c["ci95"][0] > 0 or c["ci95"][1] < 0 else " "
        print(f"  {k}: {c['diff']:+.4f}  CI {c['ci95']} {sig}")

    print("\nper-condition task_state (threat + wording held, motive removed):")
    for k, c in out["by_condition_state"].items():
        sig = "*" if c["ci95"][0] > 0 or c["ci95"][1] < 0 else " "
        print(f"  {k}: {c['diff']:+.4f}  CI {c['ci95']} {sig}")

    # Per-condition k/n and rate for every arm, so no rate in prose has to be
    # reconstructed from an arm total (errors.json, five entries running).
    out["by_condition"] = {}
    for name, rows in arms.items():
        out["by_condition"][name] = {}
        for cond in sorted({r["condition"] for r in rows}):
            sub = [r for r in rows if r["condition"] == cond]
            k, n = never_rate(sub)
            p, lo, hi = wilson(k, n)
            out["by_condition"][name][cond] = {"k": k, "n": n, "rate": round(p, 4),
                                               "ci95": [round(lo, 4), round(hi, 4)]}

    print(f"\nmodel: {out['model']}")
    print(f"\n{'arm':<16} {'n':>4} {'never':>7} {'95% CI':>18} {'complies_t1':>12}")
    for k, v in out["arms"].items():
        print(f"{k:<16} {v['n']:>4} {v['never']:>7.3f} "
              f"{str(v['never_ci95']):>18} {v['complies_t1']:>12.3f}")
    print("\nper-condition never-complies (k/n, rate, Wilson 95% CI):")
    for arm, conds in out["by_condition"].items():
        for cond, v in conds.items():
            print(f"  {arm:<16} {cond:<20} {v['k']:>3}/{v['n']:<3} {v['rate']:>7.3f} "
                  f"[{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
    print("\ncontrast (never-complies diff, bootstrap 95% CI over scenarios):")
    for k, c in out["contrasts"].items():
        sig = "*" if c["ci95"][0] > 0 or c["ci95"][1] < 0 else " "
        print(f"  {k}: {c['diff']:+.4f}  CI {c['ci95']} {sig}")

    slug = model.replace(":", "-").replace(".", "")
    dst = HERE / ("crossblock.json" if model == "qwen2.5:3b"
                  else f"crossblock-{slug}.json")
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dst.name}")

    if "--figure" in sys.argv:
        from figure import build
        dst = sys.argv[sys.argv.index("--figure") + 1]
        cells = {f"{k}|all": {"never_rate": v["never"], "ci95": v["never_ci95"], "n": v["n"]}
                 for k, v in out["arms"].items()}
        # Panel key and caption both come from the data: a figure must never be
        # labelled with a model, or a direction, it does not contain.
        b_arm, a_arm = out["arms"]["B_instr_mid"], out["arms"]["A_instr_mid"]
        direction = ("HIGHER" if b_arm["never"] > a_arm["never"] else
                     "LOWER" if b_arm["never"] < a_arm["never"] else "the same")
        svg = build({"models": {model: {"cells": cells}}}, model,
                    caption=f"{model} via Ollama (local, $0) — persistent non-compliance is "
                            f"{direction} when the threat is to the user's files "
                            f"({b_arm['never']:.3f} vs {a_arm['never']:.3f}, instructed/mid), "
                            f"n={sum(v['n'] for v in out['arms'].values())} trials")
        open(dst, "w").write(svg)
        print(f"wrote {dst}")


def self_check():
    """The estimator must find a planted difference and must NOT find one that isn't
    there. Same shape as the data: 10 scenarios, 40 rows per arm."""
    def synth(rate):
        return [{"scenario": f"s{i}", "outcome": "never_complies" if j < rate * 4 else "complies_t2"}
                for i in range(10) for j in range(4)]
    passed = 0
    c = bootstrap_contrast(synth(0.75), synth(0.25))
    assert c["diff"] == 0.5 and c["ci95"][0] > 0, c
    passed += 1
    c = bootstrap_contrast(synth(0.5), synth(0.5))
    assert c["diff"] == 0.0 and c["ci95"][0] <= 0 <= c["ci95"][1], c
    passed += 1
    assert bootstrap_contrast([{"scenario": "x", "outcome": "never_complies"}],
                              [{"scenario": "y", "outcome": "never_complies"}]) is None
    passed += 1
    k, n = never_rate(synth(0.25))
    assert (k, n) == (10, 40), (k, n)
    passed += 1
    # An arm silently mixing mid and complete rows would make every task_state contrast
    # meaningless while still printing a confident number, so load() must refuse it.
    import tempfile
    row = {"mode": "actions", "outcome": "never_complies", "scenario": "s",
           "condition": "A1_bare", "instruction": True}
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        fh.write(json.dumps({**row, "task_state": "mid"}) + "\n")
        fh.write(json.dumps({**row, "task_state": "complete"}) + "\n")
    # (True, "mid") trips the task_state guard on row 2; (False, "mid") trips the
    # instruction guard on row 1. Both must raise, not return a short row list.
    for bad_instr, bad_state in [(True, "mid"), (False, "mid")]:
        try:
            load([fh.name], bad_instr, bad_state)
            raise SystemExit(f"load() accepted a mismatched file ({bad_instr}, {bad_state})")
        except AssertionError:
            pass
    passed += 1
    # Same guard for the model: now that two models have arms in this file, loading
    # one model's arm from the other's results would silently cross them.
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh2:
        fh2.write(json.dumps({**row, "task_state": "mid", "model": "m1"}) + "\n")
    assert len(load([fh2.name], True, "mid", "m1")) == 1
    try:
        load([fh2.name], True, "mid", "m2")
        raise SystemExit("load() accepted another model's rows")
    except AssertionError:
        pass
    passed += 1
    # Every arm named in MODEL_ARMS must point at files that exist, or a contrast
    # silently disappears at runtime instead of failing loudly here.
    for m, arms in MODEL_ARMS.items():
        for name, (files, _, _) in arms.items():
            for f in files:
                assert (HERE / f).exists(), f"{m}/{name}: missing {f}"
    passed += 1
    print(f"crossblock.py self-check: {passed}/7 PASS")


if __name__ == "__main__":
    self_check() if "--self-check" in sys.argv else main()
