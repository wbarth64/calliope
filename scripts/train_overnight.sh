#!/bin/bash
# Overnight batch training sweep on RTX 5080 (16GB VRAM)
#
# Runs 6 configurations varying depth (4/6/10) and sequence length (512/1024).
# s1024 runs use re-chunked data with CHARACTER_BUDGET=3600.
#
# Usage:
#   chmod +x scripts/train_overnight.sh
#   bash scripts/train_overnight.sh
#
# Skips runs whose final checkpoint already exists (idempotent/resume-safe).
# Logs go to logs/{tag}.log, summary table to logs/summary.txt.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
NANOCHAT_DIR="$REPO_DIR/nanochat"
S1024_BASE="$HOME/.cache/nanochat-s1024"

echo "========================================"
echo "  Calliope Overnight Training Sweep"
echo "========================================"
echo ""
echo "Repo:        $REPO_DIR"
echo "nanochat:    $NANOCHAT_DIR"
echo "s1024 data:  $S1024_BASE"
echo ""

# ── Check for uv ──────────────────────────────────────────────────────────
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# ── Install dependencies ──────────────────────────────────────────────────
echo "=== Installing dependencies ==="
cd "$NANOCHAT_DIR"
uv sync --extra gpu
uv pip install kagglehub requests datasets
echo ""

# ── Prepare s512 data (CHARACTER_BUDGET=1800, default) ────────────────────
echo "=== Preparing s512 data (CHARACTER_BUDGET=1800) ==="
uv run python "$SCRIPT_DIR/prepare_poetry_data.py"
echo ""

# ── Train tokenizer (shared by all runs) ──────────────────────────────────
echo "=== Training tokenizer ==="
uv run python -m scripts.tok_train --vocab-size=32768
echo ""

# ── Prepare s1024 data (CHARACTER_BUDGET=3600) ────────────────────────────
echo "=== Preparing s1024 data (CHARACTER_BUDGET=3600) ==="
NANOCHAT_BASE_DIR="$S1024_BASE" uv run python -c "
import importlib.util, sys, os
spec = importlib.util.spec_from_file_location('ppd', os.path.join('..', 'scripts', 'prepare_poetry_data.py'))
ppd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ppd)
ppd.CHARACTER_BUDGET = 3600
ppd.main()
"
echo ""

# ── Copy tokenizer to s1024 base dir ──────────────────────────────────────
echo "=== Copying tokenizer to s1024 base dir ==="
mkdir -p "$S1024_BASE/tokenizer"
cp -r "$HOME/.cache/nanochat/tokenizer/"* "$S1024_BASE/tokenizer/"
echo "  Copied to $S1024_BASE/tokenizer/"
echo ""

# ── Create logs directory ─────────────────────────────────────────────────
mkdir -p "$REPO_DIR/logs"

# ── Experiment matrix ─────────────────────────────────────────────────────
# Format: TAG DEPTH SEQ_LEN BASE_DIR
RUNS=(
    "d4-s512   4  512  $HOME/.cache/nanochat"
    "d6-s512   6  512  $HOME/.cache/nanochat"
    "d10-s512  10 512  $HOME/.cache/nanochat"
    "d4-s1024  4  1024 $S1024_BASE"
    "d6-s1024  6  1024 $S1024_BASE"
    "d10-s1024 10 1024 $S1024_BASE"
)

echo "========================================"
echo "  Starting training sweep (${#RUNS[@]} runs)"
echo "========================================"
echo ""

cd "$NANOCHAT_DIR"

for run_spec in "${RUNS[@]}"; do
    read -r TAG DEPTH SEQ_LEN BASE_DIR <<< "$run_spec"

    CKPT_DIR="$BASE_DIR/base_checkpoints/$TAG"
    LOG_FILE="$REPO_DIR/logs/$TAG.log"

    echo "── Run: $TAG (depth=$DEPTH, seq_len=$SEQ_LEN) ──"

    # Skip if final checkpoint already exists
    if [ -d "$CKPT_DIR" ] && ls "$CKPT_DIR"/model_step_*.pt &>/dev/null; then
        echo "  Checkpoint exists at $CKPT_DIR, skipping."
        echo ""
        continue
    fi

    echo "  Logging to $LOG_FILE"
    echo "  Started at $(date)"

    NANOCHAT_BASE_DIR="$BASE_DIR" \
    OMP_NUM_THREADS=1 \
    uv run torchrun --standalone --nproc_per_node=1 \
        -m scripts.base_train \
        --depth="$DEPTH" \
        --max-seq-len="$SEQ_LEN" \
        --window-pattern=L \
        --device-batch-size=64 \
        --target-param-data-ratio=10.5 \
        --model-tag="$TAG" \
        --eval-every=250 \
        --sample-every=500 \
        --save-every=1000 \
        2>&1 | tee "$LOG_FILE"

    echo "  Finished at $(date)"
    echo ""
done

# ── Evaluation ────────────────────────────────────────────────────────────
echo "========================================"
echo "  Running evaluation on all models"
echo "========================================"
echo ""

for run_spec in "${RUNS[@]}"; do
    read -r TAG DEPTH SEQ_LEN BASE_DIR <<< "$run_spec"

    CKPT_DIR="$BASE_DIR/base_checkpoints/$TAG"
    if [ ! -d "$CKPT_DIR" ]; then
        echo "  No checkpoint for $TAG, skipping eval."
        continue
    fi

    echo "── Eval: $TAG ──"
    NANOCHAT_BASE_DIR="$BASE_DIR" \
    uv run python -m scripts.base_eval \
        --eval=bpb,sample \
        --model-tag="$TAG" \
        2>&1 | tee -a "$REPO_DIR/logs/$TAG.log"
    echo ""
done

# ── Summary table ─────────────────────────────────────────────────────────
echo "========================================"
echo "  Generating summary"
echo "========================================"
echo ""

SUMMARY_FILE="$REPO_DIR/logs/summary.txt"

{
    echo "Calliope Overnight Training Summary"
    echo "Generated: $(date)"
    echo ""
    printf "%-12s  %5s  %7s  %10s\n" "Tag" "Depth" "SeqLen" "Val BPB"
    printf "%-12s  %5s  %7s  %10s\n" "───────────" "─────" "───────" "──────────"

    for run_spec in "${RUNS[@]}"; do
        read -r TAG DEPTH SEQ_LEN BASE_DIR <<< "$run_spec"
        LOG_FILE="$REPO_DIR/logs/$TAG.log"

        BPB="n/a"
        if [ -f "$LOG_FILE" ]; then
            # Extract the last reported val bpb from the log
            LAST_BPB=$(grep -oP 'val_bpb\s*[:=]\s*\K[0-9]+\.[0-9]+' "$LOG_FILE" 2>/dev/null | tail -1)
            if [ -z "$LAST_BPB" ]; then
                # Try alternate format: "val bpb: X.XXX" or "bpb X.XXX"
                LAST_BPB=$(grep -oiP '(?:val\s+)?bpb\s*[:=]?\s*\K[0-9]+\.[0-9]+' "$LOG_FILE" 2>/dev/null | tail -1)
            fi
            if [ -n "$LAST_BPB" ]; then
                BPB="$LAST_BPB"
            fi
        fi

        printf "%-12s  %5s  %7s  %10s\n" "$TAG" "$DEPTH" "$SEQ_LEN" "$BPB"
    done
} | tee "$SUMMARY_FILE"

echo ""
echo "Summary written to $SUMMARY_FILE"
echo ""
echo "========================================"
echo "  Morning review commands:"
echo "========================================"
echo "  cat logs/summary.txt"
echo "  cd nanochat && uv run python -m scripts.chat_cli -i base -g d6-s512"
echo "  NANOCHAT_BASE_DIR=$S1024_BASE uv run python -m scripts.chat_cli -i base -g d6-s1024"
echo ""
echo "Done!"
