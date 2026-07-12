#!/usr/bin/env python3
"""Summarize final base-only prototype × buffer campaigns (EP5 or EP10).

Logs under ``_experiments/00_FINAL_EP5_BASEONLY/`` or ``00_FINAL_EP10_BASEONLY/``::

    final_<protocol>_<regime>_<buf|buffree>.log
    final10_<protocol>_<regime>_<buf|buffree>.log
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_fscil_regime_ablations import parse_rich

REPO = Path(__file__).resolve().parents[1]
METHOD_RE = re.compile(
    r"^final(?:10)?_(?P<protocol>mn40|sn55|co3d|mn_scanobj|sn_scanobj|sn_co3d)_"
    r"(?P<regime>mean_st|pshot_evo5|pshot_evo|pshot_st|hybrid3_st|hybrid4_evo|hybrid_nearmean_st|fps5_st)_(?P<buffer>buffree|buf)$"
)

PROTOCOLS = [
    ("mn40", "MN40 in-domain", 5),
    ("sn55", "SN55 in-domain", 7),
    ("co3d", "CO3D in-domain", 6),
    ("mn_scanobj", "MN40 → ScanObjectNN", 4),
    ("sn_scanobj", "SN55 → ScanObjectNN", 4),
    ("sn_co3d", "SN55 → CO3D", 11),
]

REGIMES = [
    ("mean_st", "Mean + static"),
    ("pshot_evo", "Per-shot + evolve (P3/8)"),
    ("pshot_evo5", "Per-shot + evolve (P5/5)"),
    ("pshot_st", "Per-shot + static"),
    ("hybrid3_st", "Mean+close+far (static, P=3)"),
    ("hybrid4_evo", "Mean+close+far + evolve (P≤4)"),
    ("hybrid_nearmean_st", "Mean+close only (static, P=2)"),
    ("fps5_st", "FPS on 5 shots (static, P=5)"),
]

BUFFERS = [
    ("buf", "Buffer"),
    ("buffree", "Buffer-free"),
]


def discover_final_logs(log_root: Path) -> dict[tuple[str, str, str], Path]:
    out: dict[tuple[str, str, str], Path] = {}
    for path in sorted(log_root.glob("final*.log")):
        stem = path.stem
        if stem.endswith("_seed=2008"):
            stem = stem[: -len("_seed=2008")]
        m = METHOD_RE.match(stem)
        if not m:
            continue
        key = (m.group("protocol"), m.group("regime"), m.group("buffer"))
        out[key] = path
    return out


def fmt(v: float | None) -> str:
    return f"{v:.2f}" if v is not None else "—"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        choices=("ep5", "ep10", "ep5_hybrid", "ep10_hybrid", "auto"),
        default="auto",
        help="ep5=00_FINAL_EP5_BASEONLY, ep10=00_FINAL_EP10_BASEONLY, "
        "ep5_hybrid=00_FINAL_EP5_HYBRID_CLOSEFAR, ep10_hybrid=00_FINAL_EP10_HYBRID_CLOSEFAR",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=None,
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.log_root is None:
        if args.campaign == "ep10_hybrid":
            args.log_root = REPO / "_experiments" / "00_FINAL_EP10_HYBRID_CLOSEFAR"
        elif args.campaign == "ep5_hybrid":
            args.log_root = REPO / "_experiments" / "00_FINAL_EP5_HYBRID_CLOSEFAR"
        elif args.campaign == "ep10":
            args.log_root = REPO / "_experiments" / "00_FINAL_EP10_BASEONLY"
        elif args.campaign == "ep5":
            args.log_root = REPO / "_experiments" / "00_FINAL_EP5_BASEONLY"
        else:
            args.log_root = REPO / "_experiments" / "00_FINAL_EP5_BASEONLY"

    if args.campaign in ("ep5_hybrid", "ep10_hybrid"):
        campaign_label = f"{args.campaign.upper().replace('_', ' ')} close+far"
        regime_list = [
            r for r in REGIMES
            if r[0] in ("hybrid3_st", "hybrid4_evo", "hybrid_nearmean_st", "fps5_st")
        ]
        buffer_list = [("buffree", "Buffer-free")]
    elif "EP10" in str(args.log_root):
        campaign_label = "EP10"
        regime_list = REGIMES
        buffer_list = BUFFERS
    else:
        campaign_label = "EP5"
        regime_list = REGIMES
        buffer_list = BUFFERS
    logs = discover_final_logs(args.log_root)
    lines: list[str] = [
        f"# Final {campaign_label} base-only campaign",
        "",
        f"Log root: `{args.log_root}`",
        "",
    ]

    for proto_id, proto_label, n_sess in PROTOCOLS:
        lines.append(f"## {proto_label}")
        lines.append("")
        header = (
            "| Regime | Buffer | AA | Last | Base@Last | Novel@Last | Harm@Last | Status |"
        )
        sep = "|--------|--------|---:|-----:|----------:|-----------:|----------:|--------|"
        lines.extend([header, sep])
        for regime_id, regime_label in regime_list:
            for buf_id, buf_label in buffer_list:
                path = logs.get((proto_id, regime_id, buf_id))
                if path is None:
                    lines.append(
                        f"| {regime_label} | {buf_label} | — | — | — | — | — | missing |"
                    )
                    continue
                row = parse_rich(path, n_sess)
                if row is None:
                    status = "parse fail"
                    lines.append(
                        f"| {regime_label} | {buf_label} | — | — | — | — | — | {status} |"
                    )
                    continue
                status = "ok" if row["complete"] else f"partial ({row['sessions']}/{n_sess})"
                lines.append(
                    f"| {regime_label} | {buf_label} | {fmt(row['aa'])} | {fmt(row['last'])} | "
                    f"{fmt(row['base_last'])} | {fmt(row['novel_last'])} | "
                    f"{fmt(row['harmonic_last'])} | {status} |"
                )
        lines.append("")

    n_found = len(logs)
    n_expected = len(PROTOCOLS) * len(REGIMES) * len(BUFFERS)
    lines.append(f"**Coverage:** {n_found}/{n_expected} logs found.")
    lines.append("")

    text = "\n".join(lines)
    if args.out:
        args.out.write_text(text)
        print(f"Wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
