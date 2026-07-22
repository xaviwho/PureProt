#!/usr/bin/env bash
# B1 — ARM determinism matrix (Jetson aarch64).
#
# Runs determinism_harness.py under multiple ONNX Runtime versions, pinning numpy
# per version to MATCH the x86 run so CPU architecture is the only variable in the
# cross-arch comparison. Each venv is deleted right after its run to respect the
# Jetson's tight SD-card space. onnxruntime versions with no aarch64 cp312 wheel
# are recorded as INSTALL_FAILED (a finding), not worked around.
#
# Assumes: ~/pureprotx/py312 (standalone CPython 3.12.10), ~/pureprotx/det/
# holds determinism_harness.py and models_onnx/. 1.18.0 is already done.
set -u
DET="$HOME/pureprotx/det"
PY="$HOME/pureprotx/py312/bin/python3"
VERSIONS="1.16.3 1.17.3 1.19.2 1.20.1 1.22.0"

for ver in $VERSIONS; do
  echo "===== ORT $ver ====="
  case "$ver" in
    1.16.3|1.17.3) NP="numpy==1.26.4" ;;   # numpy1 ABI (matches x86)
    *)             NP="numpy==2.5.1"  ;;   # numpy2 (matches x86)
  esac
  VD="$DET/venv_$ver"
  rm -rf "$VD"; "$PY" -m venv "$VD"
  if ! "$VD/bin/pip" install --quiet --disable-pip-version-check "onnxruntime==$ver" "$NP" 2>"$DET/err_$ver.log"; then
    echo "INSTALL_FAILED $ver :: $(tail -1 "$DET/err_$ver.log")"
    rm -rf "$VD"; continue
  fi
  # numpy ABI safety net: if import crashes, fall back to numpy<2 and record it
  if ! "$VD/bin/python" -c "import onnxruntime" >/dev/null 2>&1; then
    echo "  [abi] numpy retry (<2) for $ver"
    "$VD/bin/pip" install --quiet "numpy<2" >/dev/null 2>&1
    "$VD/bin/python" -c "import onnxruntime" >/dev/null 2>&1 || { echo "IMPORT_FAILED $ver"; rm -rf "$VD"; continue; }
  fi
  "$VD/bin/python" "$DET/determinism_harness.py" \
    --onnx-dir "$DET/models_onnx" \
    --out "$DET/harness_ort${ver}_arm.json" 2>&1 | tail -3
  rm -rf "$VD"   # reclaim SD space immediately
done
echo "ARM MATRIX DONE"
