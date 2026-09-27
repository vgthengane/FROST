#!/usr/bin/env bash
# Summarize PRidE image long convergence runs (task-0 warmup peak + FSCIL Acc).
#
# Usage:
#   bash scripts/summarize_pride_image_convergence.sh
#   bash scripts/summarize_pride_image_convergence.sh --log-root _experiments/00_PRIDE_IMAGE_CONVERGENCE_LONG
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGROOT="${REPO}/_experiments/00_PRIDE_IMAGE_CONVERGENCE_LONG"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --log-root) LOGROOT="${2:?}"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -d "${LOGROOT}" ]]; then
  echo "Missing ${LOGROOT}"
  exit 1
fi

echo "log_root=${LOGROOT}"
printf "%-52s %6s %8s %8s %6s %6s %s\n" \
  "method" "bestEp" "bestT0" "finalT0" "last" "avg" "status"
printf "%-52s %6s %8s %8s %6s %6s %s\n" \
  "------" "------" "------" "-------" "----" "---" "------"

shopt -s nullglob
for f in "${LOGROOT}"/conv_*.log; do
  base="$(basename "${f}" .log)"

  # Best / final task-0 adapter test acc from tqdm lines (last hit per epoch wins).
  best_line="$(
    python3 - "${f}" <<'PY'
import re, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(errors="ignore")
pat = re.compile(
    r"Task 0 warmup Epoch (\d+)/(\d+) => Loss [\d.]+, Train_accy ([\d.]+), Test_accy ([\d.]+)"
)
by = {}
for m in pat.finditer(text):
    by[int(m.group(1))] = (int(m.group(2)), float(m.group(3)), float(m.group(4)))
if not by:
    # fallback structured curve
    pat2 = re.compile(
        r"warmup_curve task=0 epoch=(\d+)/(\d+).*train_acc=([\d.]+) test_acc=([\d.]+)"
    )
    for m in pat2.finditer(text):
        by[int(m.group(1))] = (int(m.group(2)), float(m.group(3)), float(m.group(4)))
if not by:
    print("NA NA NA NA")
else:
    best_e, (_, _, best_te) = max(by.items(), key=lambda kv: kv[1][2])
    last_e = max(by)
    _, _, last_te = by[last_e]
    print(f"{best_e} {best_te:.2f} {last_e} {last_te:.2f}")
PY
  )"
  read -r best_ep best_t0 last_ep final_t0 <<< "${best_line}"

  curve_line="$(grep "CNN top1 curve:" "${f}" 2>/dev/null | tail -n 1 || true)"
  if [[ -n "${curve_line}" ]]; then
    curve="${curve_line#*CNN top1 curve: }"
    curve="${curve//[\[\]]/}"
    n="$(echo "${curve}" | tr -cd ',' | wc -c)"
    n=$((n + 1))
    last="$(echo "${curve}" | awk -F',' '{gsub(/ /,"",$NF); print $NF}')"
    avg="$(grep "Average Accuracy (CNN):" "${f}" 2>/dev/null | tail -n 1 | awk '{print $NF}')"
    status="full(${n})"
  else
    last="-"
    avg="-"
    if [[ "${best_ep}" != "NA" ]]; then
      status="task0(${last_ep})"
    else
      status="incomplete"
    fi
  fi

  printf "%-52s %6s %8s %8s %6s %6s %s\n" \
    "${base}" "${best_ep}" "${best_t0}" "${final_t0}" "${last}" "${avg}" "${status}"
done

done_n="$(ls -1 "${LOGROOT}"/conv_*.log 2>/dev/null | wc -l | tr -d ' ')"
echo
echo "logs=${done_n}  expected=9"
echo "Tip: compare bestEp vs tuned_epoch — if bestEp << ep, longer training is wasted."
