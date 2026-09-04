#!/bin/bash
# Every test in one command. Each file is a standalone script that prints one line per
# assertion and exits non-zero on failure -- no pytest, no fixtures, no collection magic.
set -u
cd "$(dirname "$0")"
PY="${PYTHON:-.venv/bin/python}"
export BIGRIG_MEM_GB="${BIGRIG_MEM_GB:-9}"
total=0; failed=""
for t in tests/test_*.py; do
  out=$("$PY" "$t" 2>&1); n=$(echo "$out" | grep -c "  PASS  "); total=$((total+n))
  if echo "$out" | grep -q "ALL TESTS PASSED"; then
    printf "  %-26s %3d PASS\n" "$(basename "$t")" "$n"
  else
    printf "  %-26s %3d FAIL\n" "$(basename "$t")" "$n"; failed="$failed $(basename "$t")"
    echo "$out" | grep "  FAIL  " | head -5
  fi
done
printf "  %-26s %3d assertions\n" "TOTAL" "$total"
[ -n "$failed" ] && { echo "  FAILING:$failed"; exit 1; } || { echo "  ALL GREEN"; exit 0; }
