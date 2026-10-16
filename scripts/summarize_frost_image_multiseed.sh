#!/usr/bin/env bash
# Summarize PRidE image locked 5-seed campaign.
#
# Usage:
#   bash scripts/summarize_frost_image_multiseed.sh
#   bash scripts/summarize_frost_image_multiseed.sh --log-root _experiments/00_PRIDE_IMAGE_LOCKED_A05_RP2000_5SEEDS
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGROOT="${REPO}/_experiments/00_PRIDE_IMAGE_LOCKED_A05_RP2000_5SEEDS"
INSERT="mean_best2"  # or mean1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --log-root) LOGROOT="${2:?}"; shift 2 ;;
    --insert) INSERT="${2:?}"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -d "${LOGROOT}" ]]; then
  echo "Missing ${LOGROOT}"
  exit 1
fi

# Auto-pick insert from log-root name if still default and dir looks mean-only.
if [[ "${INSERT}" == "mean_best2" && "${LOGROOT}" == *MEANONLY* ]]; then
  INSERT="mean1"
fi

aprun python - "${LOGROOT}" "${INSERT}" <<'PY'
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
insert = sys.argv[2]
pat = re.compile(
    rf"final_img_(cub200|aircraft|inf200)_ep(\d+)_{re.escape(insert)}_a05_rp2000_seed=(\d+)\.log$"
)

def parse(path: Path):
    text = path.read_text(errors="ignore")
    vals = novel = None
    forget = avg = None
    for line in text.splitlines():
        m = re.search(r"CNN top1 curve:\s*\[([0-9.,\s]+)\]", line)
        if m:
            vals = [float(x) for x in m.group(1).split(",")]
        m = re.search(r"FSCIL novel accuracy curve:\s*\[([0-9.,\s]+)\]", line)
        if m:
            novel = [float(x) for x in m.group(1).split(",")]
        m = re.search(r"Average Accuracy \(CNN\):\s*([0-9.]+)", line)
        if m:
            avg = float(m.group(1))
        m = re.search(r"Forgetting \(CNN\):\s*([0-9.]+)", line)
        if m:
            forget = float(m.group(1))
    if not vals or len(vals) < 11:
        return None
    if avg is None:
        avg = sum(vals) / len(vals)
    return {
        "base": vals[0],
        "last": vals[-1],
        "avg": avg,
        "delta": (vals[0] - vals[-1]) / vals[0] * 100,
        "novel": novel[-1] if novel else None,
        "forget": forget,
        "curve": vals,
    }

by = defaultdict(dict)
for f in sorted(root.glob("final_img_*.log")):
    m = pat.match(f.name)
    if not m:
        continue
    proto, ep, seed = m.group(1), int(m.group(2)), int(m.group(3))
    met = parse(f)
    if met:
        by[(proto, ep)][seed] = met

order = [("cub200", 100), ("inf200", 100), ("aircraft", 120)]
print(f"log_root={root}")
print(f"insert={insert}")
print(
    f"{'proto':<10} {'ep':>3} {'n':>2}  "
    f"{'Ab':>11} {'AT':>11} {'Abar':>11} {'Delta':>11} {'Novel':>11} {'Forget':>11}"
)
print("-" * 95)

def fmt_ms(vals):
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return f"{st.mean(vals):.2f}±{st.stdev(vals):.2f}"

for proto, ep in order:
    seeds = by.get((proto, ep), {})
    if not seeds:
        print(f"{proto:<10} {ep:>3}  0  (no complete logs)")
        continue
    rows = [seeds[s] for s in sorted(seeds)]
    print(
        f"{proto:<10} {ep:>3} {len(rows):>2}  "
        f"{fmt_ms([r['base'] for r in rows]):>11} "
        f"{fmt_ms([r['last'] for r in rows]):>11} "
        f"{fmt_ms([r['avg'] for r in rows]):>11} "
        f"{fmt_ms([r['delta'] for r in rows]):>11} "
        f"{fmt_ms([r['novel'] for r in rows if r['novel'] is not None]):>11} "
        f"{fmt_ms([r['forget'] for r in rows if r['forget'] is not None]):>11}"
    )
    for seed in sorted(seeds):
        r = seeds[seed]
        nv = f"{r['novel']:.2f}" if r["novel"] is not None else "n/a"
        fg = f"{r['forget']:.2f}" if r["forget"] is not None else "n/a"
        print(
            f"  seed={seed}: Ab={r['base']:.2f} AT={r['last']:.2f} "
            f"Abar={r['avg']:.2f} Δ={r['delta']:.2f} Novel={nv} Forget={fg}"
        )

done = sum(len(v) for v in by.values())
print()
print(f"complete_logs={done}  expected=15")
PY
