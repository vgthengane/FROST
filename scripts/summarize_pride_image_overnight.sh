#!/usr/bin/env bash
# Summarize overnight PRidE image sweep for morning triage.
#
# Usage:
#   bash scripts/summarize_pride_image_overnight.sh
#   bash scripts/summarize_pride_image_overnight.sh --log-root _experiments/00_PRIDE_IMAGE_OVERNIGHT_SWEEP
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGROOT="${REPO}/_experiments/00_PRIDE_IMAGE_OVERNIGHT_SWEEP"
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
printf "%-56s %6s %6s %6s %s\n" "method" "last" "avg" "sess" "curve"
printf "%-56s %6s %6s %6s %s\n" "------" "----" "---" "----" "-----"

shopt -s nullglob
rows=()
for f in "${LOGROOT}"/ovn_*.log; do
  base="$(basename "${f}" .log)"
  curve_line="$(grep "CNN top1 curve:" "${f}" 2>/dev/null | tail -n 1 || true)"
  [[ -n "${curve_line}" ]] || continue
  curve="${curve_line#*CNN top1 curve: }"
  curve="${curve//[\[\]]/}"
  n="$(echo "${curve}" | tr -cd ',' | wc -c)"
  n=$((n + 1))
  last="$(echo "${curve}" | awk -F',' '{gsub(/ /,"",$NF); print $NF}')"
  avg="$(grep "Average Accuracy (CNN):" "${f}" 2>/dev/null | tail -n 1 | awk '{print $NF}')"
  printf "%-56s %6s %6s %6s %s\n" "${base}" "${last}" "${avg}" "${n}" "${curve}"
  rows+=("${avg}|${last}|${base}|${curve}")
done

echo
echo "=== Top 15 by Avg Acc (completed only) ==="
if [[ "${#rows[@]}" -gt 0 ]]; then
  printf "%s\n" "${rows[@]}" | sort -t'|' -k1,1nr | head -15 | while IFS='|' read -r avg last method curve; do
    printf "Avg=%-7s Last=%-7s  %s\n" "${avg}" "${last}" "${method}"
  done
else
  echo "(no completed logs yet)"
fi

done_n="$(ls -1 "${LOGROOT}"/ovn_*.log 2>/dev/null | wc -l | tr -d ' ')"
ok_n=0
for f in "${LOGROOT}"/ovn_*.log; do
  [[ -f "${f}" ]] || continue
  if grep -q "CNN top1 curve:" "${f}"; then
    n="$(grep "CNN top1 curve:" "${f}" | tail -n 1 | tr -cd ',' | wc -c)"
    n=$((n + 1))
    [[ "${n}" -ge 11 ]] && ok_n=$((ok_n + 1))
  fi
done
echo
echo "logs=${done_n}  complete_11sess=${ok_n}  expected=324"
