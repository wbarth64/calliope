#!/bin/bash
# Train poetry LLM on Lambda Labs 8xA100/H100 node
#
# Usage:
#   1. SSH into Lambda Labs instance
#   2. Clone this repo and cd into it
#   3. Run: ./scripts/train_poetry.sh
#
# Prerequisites:
#   - Python 3.10+
#   - uv package manager (curl -LsSf https://astral.sh/uv/install.sh | sh)
#   - NVIDIA GPU with CUDA support

set -e  # Exit on error

echo "=== Poetry LLM Training ==="
echo ""

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install nanochat dependencies (includes torch, wandb, etc.)
echo "Installing nanochat dependencies..."
cd nanochat
uv sync --extra gpu
cd ..

# Prepare poetry data (downloads corpus, creates parquet files)
echo ""
echo "=== Preparing Poetry Data ==="
cd nanochat
uv run python ../scripts/prepare_poetry_data.py

# Train tokenizer on poetry data (32K vocab)
echo ""
echo "=== Training Tokenizer ==="
uv run python -m scripts.tok_train --vocab-size=32768

cd ..

# Launch distributed training on 8 GPUs
echo ""
echo "=== Starting Training ==="
echo "Training configuration:"
echo "  - Model depth: 10 (~25M params)"
echo "  - Sequence length: 512"
echo "  - Batch size per device: 32"
echo "  - Iterations: 10000"
echo ""

cd nanochat
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_train \
    --depth=10 \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --num-iterations=10000 \
    --eval-every=250 \
    --save-every=1000
cd ..

echo ""
echo "=== Training Complete ==="
echo "Checkpoint saved to: ~/.cache/nanochat/base_checkpoints/d10/"
echo ""
echo "To run inference locally:"
echo "  scp -r user@lambda:~/.cache/nanochat/base_checkpoints/d10 ~/.cache/nanochat/base_checkpoints/"
echo "  scp -r user@lambda:~/.cache/nanochat/tokenizer ~/.cache/nanochat/"
echo "  cd nanochat && uv run python -m scripts.chat_cli --checkpoint=d10"
