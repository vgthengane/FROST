#!/usr/bin/env python3
"""Summarize locked-setting PRidE ablations on FSCIL ModelNet40 (5 seeds).

Default Gram+ProtoBank row is reused from::

    _experiments/00_FINAL_SUB_MEAN_A05_RP2000_EP10_5SEEDS/
      final_sub_mn40_mean_st_buffree_a05_rp2000_seed=<seed>.log

Optional RP=1000 twin reuse::

    _experiments/00_FINAL_SUB_MEAN_A05_RP1000_EP10_5SEEDS/

New runs live under::

    _experiments/00_PRIDE_LOCKED_ABLATION_MN40_5SEEDS/
      pride_mn40_abl_<id>_seed=<seed>.log

Usage::

    aprun python scripts/summarize_pride_locked_ablation_mn40.py
    aprun python scripts/summarize_pride_locked_ablation_mn40.py --latex
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_fscil_results import parse_fscil_log

REPO = Path(__file__).resolve().parents[1]
EXPECTED_SESSIONS = 5
SEEDS = (1947, 1991, 2001, 2008, 2019)

DEFAULT_ROOT = REPO / "_experiments" / "00_FINAL_SUB_MEAN_A05_RP2000_EP10_5SEEDS"
DEFAULT_SINGLE = REPO / "_experiments" / "00_FINAL_SUB_MEAN_A05_RP2000_EP10"
RP1000_ROOT = REPO / "_experiments" / "00_FINAL_SUB_MEAN_A05_RP1000_EP10_5SEEDS"
RP1000_SINGLE = REPO / "_experiments" / "00_FINAL_SUB_MEAN_A05_RP1000_EP10"
ABL_ROOT = REPO / "_experiments" / "00_PRIDE_LOCKED_ABLATION_MN40_5SEEDS"

# (group, id, paper label, log source)
# source: ("abl", method_id) | ("final_sub",) | ("rp1000",)
ROWS: list[tuple[str, str, str, tuple]] = [
    # --- paper component table ---
    ("component", "linear", "Linear", ("abl", "linear")),
    ("component", "adapter_linear", "Adapter + Linear", ("abl", "adapter_linear")),
    ("component", "ridge", "GramBank", ("abl", "ridge")),
    ("component", "cache", "ProtoBank", ("abl", "cache")),
    ("component", "ridge_cache", "Gram+ProtoBank", ("final_sub",)),
    # --- fusion alpha ---
    ("alpha", "alpha01", "α = 0.1", ("abl", "alpha01")),
    ("alpha", "alpha03", "α = 0.3", ("abl", "alpha03")),
    ("alpha", "alpha05", "α = 0.5 (locked)", ("final_sub",)),
    ("alpha", "alpha07", "α = 0.7", ("abl", "alpha07")),
    ("alpha", "alpha09", "α = 0.9", ("abl", "alpha09")),
    # --- adapter width (ffn_num); 64 = locked default ---
    ("adapter", "adapter32", "ffn = 32", ("abl", "adapter32")),
    ("adapter", "adapter64", "ffn = 64 (locked)", ("final_sub",)),
    ("adapter", "adapter128", "ffn = 128", ("abl", "adapter128")),
    ("adapter", "adapter256", "ffn = 256", ("abl", "adapter256")),
    # --- ridge RP ---
    ("rp", "rp500", "RP dim = 500", ("abl", "rp500")),
    ("rp", "rp1000", "RP dim = 1000", ("rp1000",)),
    ("rp", "rp2000", "RP dim = 2000 (locked)", ("final_sub",)),
    ("rp", "rp4000", "RP dim = 4000", ("abl", "rp4000")),
    # --- insert / evo (optional later) ---
    ("insert", "mean", "Insert: mean only (locked)", ("final_sub",)),
    ("insert", "ins_mean_close", "Insert: mean + two close", ("abl", "ins_mean_close")),
    ("insert", "ins_close_far", "Insert: mean + close + far", ("abl", "ins_close_far")),
    ("insert", "ins_mean_far2", "Insert: mean + two far", ("abl", "ins_mean_far2")),
    ("evo", "static", "Evolution: static (locked)", ("final_sub",)),
    ("evo", "evo_rules", "Evolution: rules", ("abl", "evo_rules")),
]


def _metrics(parsed: dict) -> dict:
    """Paper ablation metrics from a parsed FSCIL log."""
    curve = parsed["curve"]
    ab = float(curve[0])
    at = float(curve[-1])
    an = parsed.get("novel_last")
    an = float(an) if an is not None else None
    # Relative drop to match main tables.
    drel = 100.0 * (ab - at) / ab if ab else None
    return {"Ab": ab, "An": an, "AT": at, "drel": drel, "aa": parsed.get("aa")}


def _load_seed(path: Path) -> dict | None:
    if not path.exists():
        return None
    parsed = parse_fscil_log(path, EXPECTED_SESSIONS)
    if not parsed or not parsed["complete"]:
        return None
    return _metrics(parsed)


def _final_sub_path(rp: int, seed: int) -> Path:
    """Resolve final_sub MN40 log; seed 2008 may live in the single-run folder."""
    if rp == 2000:
        multi = DEFAULT_ROOT / f"final_sub_mn40_mean_st_buffree_a05_rp2000_seed={seed}.log"
        single = DEFAULT_SINGLE / "final_sub_mn40_mean_st_buffree_a05_rp2000.log"
    elif rp == 1000:
        multi = RP1000_ROOT / f"final_sub_mn40_mean_st_buffree_a05_rp1000_seed={seed}.log"
        single = RP1000_SINGLE / "final_sub_mn40_mean_st_buffree_a05_rp1000.log"
    else:
        raise ValueError(rp)
    if multi.exists():
        return multi
    if seed == 2008 and single.exists():
        return single
    return multi


def _paths_for(source: tuple, abl_root: Path) -> dict[int, Path]:
    kind = source[0]
    out: dict[int, Path] = {}
    for seed in SEEDS:
        if kind == "abl":
            vid = source[1]
            out[seed] = abl_root / f"pride_mn40_abl_{vid}_seed={seed}.log"
        elif kind == "final_sub":
            out[seed] = _final_sub_path(2000, seed)
        elif kind == "rp1000":
            # Prefer dedicated ablation run if present; else reuse final_sub rp1000.
            abl = abl_root / f"pride_mn40_abl_rp1000_seed={seed}.log"
            out[seed] = abl if abl.exists() else _final_sub_path(1000, seed)
        else:
            raise ValueError(source)
    return out


def _agg(vals: list[float | None]) -> tuple[float, float] | None:
    xs = [v for v in vals if v is not None]
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def collect_row(source: tuple, abl_root: Path) -> dict | None:
    runs = []
    for seed, path in _paths_for(source, abl_root).items():
        m = _load_seed(path)
        if m is not None:
            runs.append((seed, m))
    if not runs:
        return None
    keys = ["Ab", "An", "AT", "drel", "aa"]
    return {
        "n": len(runs),
        "seeds": [s for s, _ in runs],
        **{k: _agg([m[k] for _, m in runs]) for k in keys},
    }


def _fmt(pair: tuple[float, float] | None, prec: int = 2) -> str:
    if pair is None:
        return "—"
    mu, sd = pair
    if sd < 1e-9:
        return f"{mu:.{prec}f}"
    return f"{mu:.{prec}f}±{sd:.{prec}f}"


def render_markdown(rows_data: list[tuple]) -> str:
    lines = [
        "# PRidE locked ablation — FSCIL ModelNet40 (buffer-free, 5 seeds)",
        "",
        "Locked default: mean-only, rp=2000, α=0.5, static, base-only EP10.",
        "Default Gram+ProtoBank reused from `00_FINAL_SUB_MEAN_A05_RP2000_EP10_5SEEDS`.",
        "",
        "Metrics: $\\mathcal{A}_b$=sess0, $\\mathcal{A}_n$=novel@last, "
        "$\\mathcal{A}_T$=last, $\\Delta=100(\\mathcal{A}_b-\\mathcal{A}_T)/\\mathcal{A}_b$.",
        "",
    ]
    cur_group = None
    for group, _id, label, source, data in rows_data:
        if group != cur_group:
            cur_group = group
            lines += [
                f"## {group}",
                "",
                "| Setting | n | Ab | An | AT | Δ↓ |",
                "|---------|--:|---:|---:|---:|---:|",
            ]
        if data is None:
            lines.append(f"| {label} | 0 | — | — | — | — |")
        else:
            star = " **(locked)**" if "locked" in label else ""
            lines.append(
                f"| {label}{star} | {data['n']} | {_fmt(data['Ab'])} | "
                f"{_fmt(data['An'])} | {_fmt(data['AT'])} | {_fmt(data['drel'])} |"
            )
    return "\n".join(lines) + "\n"


def render_component_latex(rows_data: list[tuple]) -> str:
    """Paper-style component ablation table (mean over seeds)."""
    lines = [
        r"\caption{Ablation study on ModelNet in the exemplar-free setting "
        r"(5 seeds). Highlighted row denotes the default configuration "
        r"(mean-only ProtoBank, $\mathrm{rp}{=}2000$, $\alpha{=}0.5$).}",
        r"\vspace{0.2em}",
        r"\label{tab:ablation}",
        r"\small",
        r"\setlength{\tabcolsep}{1mm}",
        r"\begin{tabular}{lrrrr}",
        r"  \toprule",
        r"  \rowcolor{blue!20}",
        r"  \textbf{Setting} & $\mathcal{A}_b$ & $\mathcal{A}_n$ "
        r"& $\mathcal{A}_T$ & $\Delta\,\downarrow$ \\",
        r"  \midrule",
    ]
    for group, _id, label, _source, data in rows_data:
        if group != "component":
            continue
        if data is None:
            lines.append(f"  {label:24s} &  -- & -- & -- & -- \\\\")
            continue
        ab, an, at, drel = (
            data["Ab"][0],
            data["An"][0] if data["An"] else float("nan"),
            data["AT"][0],
            data["drel"][0],
        )
        row = (
            f"  {label:24s} & {ab:5.2f} & {an:5.2f} & {at:5.2f} & {drel:5.2f} \\\\"
        )
        if _id == "ridge_cache":
            lines.append(r"  \rowcolor{red!15}")
            # Bold best AT / Ab / Δ among completed component rows later; keep simple.
            lines.append(
                f"  {label:24s} & \\textbf{{{ab:.2f}}} & {an:.2f} "
                f"& \\textbf{{{at:.2f}}} & \\textbf{{{drel:.2f}}} \\\\"
            )
        else:
            lines.append(row)
    lines += [
        r"  \bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, default=ABL_ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Also print / write component-table LaTeX for ablation.tex",
    )
    args = parser.parse_args()

    rows_data = []
    for group, rid, label, source in ROWS:
        data = collect_row(source, args.log_root)
        rows_data.append((group, rid, label, source, data))

    md = render_markdown(rows_data)
    print(md)
    out_path = args.out or (args.log_root / "locked_ablation_mn40.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"Wrote {out_path}")

    if args.latex:
        tex = render_component_latex(rows_data)
        print(tex)
        tex_path = args.log_root / "ablation_component_mn40.tex"
        tex_path.write_text(tex)
        print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
