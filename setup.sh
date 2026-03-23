#!/bin/bash
set -e

echo "=== GPM Environment Setup ==="

# Check Python version
PYTHON=$(python3 --version 2>&1)
echo "Python: $PYTHON"

# Check CUDA
CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | tr -d ',')
if [ -z "$CUDA_VER" ]; then
    echo "WARNING: nvcc not found. Attempting to detect from nvidia-smi..."
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | awk '{print $9}')
fi
echo "CUDA: $CUDA_VER"

# Map CUDA version to torch index URL
case "$CUDA_VER" in
    12.8*|12.7*|12.6*) TORCH_CUDA="cu126" ;;
    12.5*|12.4*|12.3*) TORCH_CUDA="cu124" ;;
    12.2*|12.1*|12.0*) TORCH_CUDA="cu121" ;;
    11.8*) TORCH_CUDA="cu118" ;;
    *)
        echo "WARNING: Unknown CUDA version '$CUDA_VER', defaulting to cu124"
        TORCH_CUDA="cu124"
        ;;
esac
echo "Using torch index: $TORCH_CUDA"

# Create virtualenv
echo ""
echo "=== Creating virtual environment ==="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip --quiet

# Install torch first
echo ""
echo "=== Installing PyTorch (torch 2.6.0 + $TORCH_CUDA) ==="
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/$TORCH_CUDA

# Install PyG extensions
echo ""
echo "=== Installing PyG extensions ==="
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.6.0+${TORCH_CUDA}.html

# Install DGL
echo ""
echo "=== Installing DGL ==="
TORCH_MAJOR=$(echo $TORCH_CUDA | sed 's/cu//')
pip install dgl -f https://data.dgl.ai/wheels/torch-2.6/${TORCH_CUDA}/repo.html || \
    echo "WARNING: DGL install failed, try manually: pip install dgl -f https://data.dgl.ai/wheels/torch-2.6/${TORCH_CUDA}/repo.html"

# Install remaining requirements (skip googledrivedownloader - not used)
echo ""
echo "=== Installing requirements ==="
pip install -r requirements.txt --quiet

echo ""
echo "=== Setup complete! ==="
echo "Activate with: source .venv/bin/activate"
echo "Run benchmark:  python GPM/main.py --dataset zinc --use_params --debug"
