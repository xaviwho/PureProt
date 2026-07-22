#!/usr/bin/env bash
# Device-side edge-throughput runner (Jetson / Pi4).
# Builds the edge venv (onnxruntime 1.19.2 + rdkit + web3), then runs
# edge_throughput.py with the PureChain mainnet commit enabled.
# Required env: TEST_PRIVATE_KEY, DEVICE_LABEL.
set -u
DET="$HOME/pureprotx/det"
PY="$HOME/pureprotx/py312/bin/python3"
VD="$HOME/pureprotx/edge_venv"

if [ ! -x "$VD/bin/python" ]; then "$PY" -m venv "$VD"; fi
"$VD/bin/pip" install --quiet --disable-pip-version-check \
    "onnxruntime==1.19.2" "rdkit==2025.9.4" "web3==7.14.1" 2>&1 | tail -2
"$VD/bin/python" -c "import onnxruntime,rdkit,web3,numpy as n;print('deps:',onnxruntime.__version__,rdkit.__version__,web3.__version__,n.__version__)" || exit 1

export PURECHAIN_RPC_URL="https://purechainnode.com"
export CONTRACT_ADDRESS="0xb8eb74663c1297825b188D8454a469d02Cc7d56C"
"$VD/bin/python" "$DET/edge_throughput.py" \
  --csv "$DET/chembl243_prepared_data.csv" \
  --onnx-dir "$DET/models_onnx" \
  --device "$DEVICE_LABEL" --storage SD --commit \
  --deploy-json "$DET/purechain_deployment.json" \
  --out "$DET/edge_${DEVICE_LABEL}.json" 2>&1 | tail -12
echo "DEVICE $DEVICE_LABEL DONE"
