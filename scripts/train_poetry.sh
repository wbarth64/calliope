#!/bin/bash
# Train poetry LLM on Lambda Labs 8xA100/H100 node
#
# Usage:
#   1. SSH into Lambda Labs instance
#   2. Clone this repo and cd into it
#   3. git submodule update --init
#   4. Run: ./scripts/train_poetry.sh
#
# Prerequisites:
#   - Python 3.10+
#   - uv package manager (curl -LsSf https://astral.sh/uv/install.sh | sh)
#   - NVIDIA GPU with CUDA support
#
# Configurable via env vars:
#   DEPTH          Model depth (default: 6, ~11M params. Use 4 for Chinchilla-optimal)
#   NUM_ITERATIONS Override iteration count (default: auto via param-data ratio)
#   WANDB_RUN      Wandb run name (default: "dummy" = no logging)

set -e  # Exit on error

DEPTH="${DEPTH:-6}"
NUM_ITERATIONS="${NUM_ITERATIONS:-}"
WANDB_RUN="${WANDB_RUN:-dummy}"

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

# Install extra dependencies for dataset downloads
echo ""
echo "=== Installing extra dependencies ==="
cd nanochat
uv pip install kagglehub requests

# Prepare poetry data (downloads all 4 corpora, creates parquet shards)
echo ""
echo "=== Preparing Poetry Data ==="
uv run python ../scripts/prepare_poetry_data.py

# Train tokenizer on poetry data (32K vocab)
echo ""
echo "=== Training Tokenizer ==="
uv run python -m scripts.tok_train --vocab-size=32768

# Build iteration args
ITER_ARGS=""
if [ -n "$NUM_ITERATIONS" ]; then
    ITER_ARGS="--num-iterations=$NUM_ITERATIONS"
else
    # Use Chinchilla-style param-data ratio to auto-compute iterations
    ITER_ARGS="--target-param-data-ratio=10.5"
fi

# Launch distributed training on 8 GPUs
echo ""
echo "=== Starting Training ==="
echo "Training configuration:"
echo "  - Model depth: $DEPTH"
echo "  - Sequence length: 512"
echo "  - Batch size per device: 32"
echo "  - Iterations: ${NUM_ITERATIONS:-auto (param-data ratio 10.5)}"
echo "  - Wandb: $WANDB_RUN"
echo ""

OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_train \
    --run="$WANDB_RUN" \
    --depth="$DEPTH" \
    --max-seq-len=512 \
    --window-pattern=L \
    --device-batch-size=32 \
    $ITER_ARGS \
    --eval-every=250 \
    --save-every=1000

cd ..

echo ""
echo "=== Training Complete ==="
echo "Checkpoint saved to: ~/.cache/nanochat/base_checkpoints/d${DEPTH}/"
echo ""
echo "To run inference locally:"
echo "  scp -r user@lambda:~/.cache/nanochat/base_checkpoints/d${DEPTH} ~/.cache/nanochat/base_checkpoints/"
echo "  scp -r user@lambda:~/.cache/nanochat/tokenizer ~/.cache/nanochat/"
echo "  cd nanochat && uv run python -m scripts.chat_cli --checkpoint=d${DEPTH}"
