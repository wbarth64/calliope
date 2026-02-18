#!/usr/bin/env python3
"""
ASCII dashboard for overnight training results.

Usage:
    python scripts/dashboard.py              # full dashboard
    python scripts/dashboard.py --live       # auto-refresh every 30s (while training)
    python scripts/dashboard.py --run d6-s512  # show single run detail
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────

REPO_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_DIR / "logs"

RUNS = [
    ("d4-s512",   4,  512,  "~/.cache/nanochat"),
    ("d6-s512",   6,  512,  "~/.cache/nanochat"),
    ("d10-s512",  10, 512,  "~/.cache/nanochat"),
    ("d4-s1024",  4,  1024, "~/.cache/nanochat-s1024"),
    ("d6-s1024",  6,  1024, "~/.cache/nanochat-s1024"),
    ("d10-s1024", 10, 1024, "~/.cache/nanochat-s1024"),
]

# ANSI colors
BOLD    = "\033[1m"
DIM     = "\033[2m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
CYAN    = "\033[36m"
RED     = "\033[31m"
MAGENTA = "\033[35m"
RESET   = "\033[0m"

BAR_CHARS = " ▏▎▍▌▋▊▉█"


# ── Log parsing ───────────────────────────────────────────────────────────

def parse_log(tag):
    """Parse a training log file and extract metrics."""
    log_path = LOGS_DIR / f"{tag}.log"
    if not log_path.exists():
        return None

    text = log_path.read_text(errors="replace")
    info = {
        "tag": tag,
        "steps": [],          # (step, loss)
        "val_bpbs": [],       # (step, bpb)
        "samples": [],        # sample text snippets
        "total_steps": None,
        "min_val_bpb": None,
        "final_val_bpb": None,
        "tok_per_sec": None,
        "total_time": None,
        "status": "unknown",
        "last_step": 0,
        "params": None,
    }

    # Extract training steps: "step 00100/02000 ... | loss: 5.123456 | ... | tok/sec: 12,345 | ..."
    for m in re.finditer(
        r"step\s+(\d+)/(\d+).*?loss:\s+([\d.]+).*?tok/sec:\s+([\d,]+).*?total time:\s+([\d.]+)m",
        text
    ):
        step = int(m.group(1))
        total = int(m.group(2))
        loss = float(m.group(3))
        tps = int(m.group(4).replace(",", ""))
        info["steps"].append((step, loss))
        info["total_steps"] = total
        info["last_step"] = max(info["last_step"], step)
        info["tok_per_sec"] = tps
        info["total_time"] = float(m.group(5))

    # Extract val bpb: "Step 00250 | Validation bpb: 1.234567"
    for m in re.finditer(r"Step\s+(\d+)\s+\|\s+Validation bpb:\s+([\d.]+)", text):
        step = int(m.group(1))
        bpb = float(m.group(2))
        info["val_bpbs"].append((step, bpb))

    # Extract min/final val bpb
    m = re.search(r"Minimum validation bpb:\s+([\d.]+)", text)
    if m:
        info["min_val_bpb"] = float(m.group(1))

    if info["val_bpbs"]:
        info["final_val_bpb"] = info["val_bpbs"][-1][1]
        if info["min_val_bpb"] is None:
            info["min_val_bpb"] = min(bpb for _, bpb in info["val_bpbs"])

    # Extract param count from model summary
    m = re.search(r"([\d.]+)M\s+parameters", text)
    if m:
        info["params"] = f"{m.group(1)}M"

    # Determine status
    if info["total_steps"] and info["last_step"] >= info["total_steps"]:
        info["status"] = "done"
    elif info["steps"]:
        info["status"] = "running"
    else:
        info["status"] = "pending"

    return info


# ── Drawing helpers ───────────────────────────────────────────────────────

def bar(value, max_val, width=30, color=CYAN):
    """Render a horizontal bar."""
    if max_val <= 0:
        return " " * width
    ratio = min(value / max_val, 1.0)
    full_blocks = int(ratio * width)
    remainder = (ratio * width) - full_blocks
    partial_idx = int(remainder * 8)
    s = BAR_CHARS[-1] * full_blocks
    if full_blocks < width:
        s += BAR_CHARS[partial_idx]
        s += " " * (width - full_blocks - 1)
    return f"{color}{s}{RESET}"


def sparkline(values, width=40):
    """Render a sparkline from a list of values."""
    if not values:
        return DIM + "no data" + RESET
    sparks = " ▁▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    rng = mx - mn if mx > mn else 1.0
    # Downsample if needed
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values
    line = ""
    for v in sampled:
        idx = int((v - mn) / rng * 8)
        idx = min(idx, 8)
        line += sparks[idx]
    return f"{CYAN}{line}{RESET}"


def status_icon(status):
    if status == "done":
        return f"{GREEN}[done]{RESET}"
    elif status == "running":
        return f"{YELLOW}[running]{RESET}"
    else:
        return f"{DIM}[pending]{RESET}"


def fmt_bpb(val):
    if val is None:
        return f"{DIM}  --  {RESET}"
    return f"{BOLD}{val:.4f}{RESET}"


def progress_bar(current, total, width=20):
    if not total:
        return f"{DIM}{'?' * width}{RESET}"
    pct = min(current / total, 1.0)
    filled = int(pct * width)
    return (
        f"{GREEN}{'█' * filled}{DIM}{'░' * (width - filled)}{RESET}"
        f" {current}/{total} ({pct:.0%})"
    )


# ── Dashboard views ───────────────────────────────────────────────────────

def print_header():
    print()
    print(f"{BOLD}{MAGENTA}  ╔═══════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{MAGENTA}  ║          C A L L I O P E   D A S H B O A R D             ║{RESET}")
    print(f"{BOLD}{MAGENTA}  ╚═══════════════════════════════════════════════════════════╝{RESET}")
    print()


def print_overview():
    """Print the main comparison table."""
    all_info = []
    for tag, depth, seq_len, base_dir in RUNS:
        info = parse_log(tag)
        all_info.append((tag, depth, seq_len, info))

    # Header
    print(f"  {BOLD}{'Tag':<12} {'Depth':>5} {'Seq':>5} {'Params':>7} {'Status':<12} {'Min BPB':>9} {'Final BPB':>10} {'Progress'}{RESET}")
    print(f"  {'─' * 90}")

    best_bpb = None
    for tag, depth, seq_len, info in all_info:
        if info and info["min_val_bpb"] is not None:
            if best_bpb is None or info["min_val_bpb"] < best_bpb:
                best_bpb = info["min_val_bpb"]

    for tag, depth, seq_len, info in all_info:
        if info is None:
            print(f"  {tag:<12} {depth:>5} {seq_len:>5} {'--':>7} {DIM}[no log]{RESET}")
            continue

        params = info["params"] or "--"
        status = status_icon(info["status"])
        min_bpb = fmt_bpb(info["min_val_bpb"])
        final_bpb = fmt_bpb(info["final_val_bpb"])
        prog = progress_bar(info["last_step"], info["total_steps"])

        # Highlight best
        marker = ""
        if best_bpb and info["min_val_bpb"] == best_bpb:
            marker = f" {GREEN}<-- best{RESET}"

        print(f"  {tag:<12} {depth:>5} {seq_len:>5} {params:>7} {status:<22} {min_bpb:>18} {final_bpb:>19}  {prog}{marker}")

    print()


def print_bpb_chart():
    """Print a bar chart comparing final val BPB across runs."""
    entries = []
    for tag, depth, seq_len, base_dir in RUNS:
        info = parse_log(tag)
        if info and info["min_val_bpb"] is not None:
            entries.append((tag, info["min_val_bpb"]))

    if not entries:
        print(f"  {DIM}No BPB data yet.{RESET}")
        return

    print(f"  {BOLD}Val BPB Comparison (lower is better){RESET}")
    print(f"  {'─' * 55}")

    max_bpb = max(b for _, b in entries)
    min_bpb = min(b for _, b in entries)

    for tag, bpb in entries:
        is_best = bpb == min_bpb
        color = GREEN if is_best else CYAN
        label = f"  {tag:<12} {bpb:.4f} "
        b = bar(bpb, max_bpb * 1.05, width=35, color=color)
        star = f" {GREEN}*{RESET}" if is_best else ""
        print(f"{label}{b}{star}")

    print()


def print_loss_curves():
    """Print sparkline loss curves for each run."""
    print(f"  {BOLD}Training Loss Curves{RESET}")
    print(f"  {'─' * 55}")

    for tag, depth, seq_len, base_dir in RUNS:
        info = parse_log(tag)
        if info is None or not info["steps"]:
            print(f"  {tag:<12} {DIM}no data{RESET}")
            continue

        losses = [loss for _, loss in info["steps"]]
        spark = sparkline(losses, width=40)
        first = losses[0] if losses else 0
        last = losses[-1] if losses else 0
        print(f"  {tag:<12} {spark}  {DIM}{first:.3f} -> {last:.3f}{RESET}")

    print()


def print_val_curves():
    """Print sparkline val BPB curves for each run."""
    print(f"  {BOLD}Validation BPB Curves{RESET}")
    print(f"  {'─' * 55}")

    for tag, depth, seq_len, base_dir in RUNS:
        info = parse_log(tag)
        if info is None or not info["val_bpbs"]:
            print(f"  {tag:<12} {DIM}no data{RESET}")
            continue

        bpbs = [bpb for _, bpb in info["val_bpbs"]]
        spark = sparkline(bpbs, width=40)
        first = bpbs[0] if bpbs else 0
        last = bpbs[-1] if bpbs else 0
        print(f"  {tag:<12} {spark}  {DIM}{first:.4f} -> {last:.4f}{RESET}")

    print()


def print_throughput():
    """Print throughput comparison."""
    entries = []
    for tag, depth, seq_len, base_dir in RUNS:
        info = parse_log(tag)
        if info and info["tok_per_sec"]:
            entries.append((tag, info["tok_per_sec"], info.get("total_time")))

    if not entries:
        return

    print(f"  {BOLD}Throughput{RESET}")
    print(f"  {'─' * 55}")

    max_tps = max(t for _, t, _ in entries)
    for tag, tps, total_time in entries:
        b = bar(tps, max_tps * 1.05, width=25, color=CYAN)
        time_str = f"  ({total_time:.1f}m)" if total_time else ""
        print(f"  {tag:<12} {tps:>8,} tok/s  {b}{time_str}")

    print()


def print_seq_len_comparison():
    """Side-by-side comparison: s512 vs s1024 for each depth."""
    print(f"  {BOLD}Sequence Length Impact (s512 vs s1024){RESET}")
    print(f"  {'─' * 55}")

    for depth in [4, 6, 10]:
        tag_512 = f"d{depth}-s512"
        tag_1024 = f"d{depth}-s1024"
        info_512 = parse_log(tag_512)
        info_1024 = parse_log(tag_1024)

        bpb_512 = info_512["min_val_bpb"] if info_512 else None
        bpb_1024 = info_1024["min_val_bpb"] if info_1024 else None

        label = f"  depth={depth:<3}"
        if bpb_512 is not None and bpb_1024 is not None:
            delta = bpb_1024 - bpb_512
            pct = (delta / bpb_512) * 100
            direction = f"{GREEN}better{RESET}" if delta < 0 else f"{RED}worse{RESET}"
            print(f"{label}  s512={bpb_512:.4f}  s1024={bpb_1024:.4f}  "
                  f"delta={delta:+.4f} ({pct:+.1f}% {direction})")
        else:
            v512 = f"{bpb_512:.4f}" if bpb_512 else "--"
            v1024 = f"{bpb_1024:.4f}" if bpb_1024 else "--"
            print(f"{label}  s512={v512}  s1024={v1024}")

    print()


def print_detail(tag):
    """Print detailed view for a single run."""
    info = parse_log(tag)
    if info is None:
        print(f"  {RED}No log found for {tag}{RESET}")
        return

    print(f"  {BOLD}Detail: {tag}{RESET}  {status_icon(info['status'])}")
    print(f"  {'─' * 55}")
    print(f"  Params:        {info['params'] or 'unknown'}")
    print(f"  Total steps:   {info['total_steps'] or 'unknown'}")
    print(f"  Current step:  {info['last_step']}")
    print(f"  Min val BPB:   {fmt_bpb(info['min_val_bpb'])}")
    print(f"  Final val BPB: {fmt_bpb(info['final_val_bpb'])}")
    print(f"  Throughput:    {info['tok_per_sec']:,} tok/s" if info["tok_per_sec"] else "")
    print(f"  Total time:    {info['total_time']:.1f}m" if info["total_time"] else "")
    print()

    if info["val_bpbs"]:
        print(f"  {BOLD}Val BPB over training:{RESET}")
        bpbs = [bpb for _, bpb in info["val_bpbs"]]
        mn, mx = min(bpbs), max(bpbs)

        # Vertical chart
        chart_h = 12
        chart_w = min(len(bpbs), 60)
        if len(bpbs) > chart_w:
            step_size = len(bpbs) / chart_w
            sampled = [(info["val_bpbs"][int(i * step_size)]) for i in range(chart_w)]
        else:
            sampled = info["val_bpbs"]

        rng = mx - mn if mx > mn else 0.001
        for row in range(chart_h):
            threshold = mx - (row / (chart_h - 1)) * rng
            line = "  "
            if row == 0:
                line += f"{mx:.3f} │"
            elif row == chart_h - 1:
                line += f"{mn:.3f} │"
            else:
                line += "       │"
            for step, bpb in sampled:
                if bpb >= threshold:
                    line += f"{CYAN}█{RESET}"
                else:
                    line += " "
            print(line)
        print(f"         └{'─' * len(sampled)}")
        steps_range = f"step {sampled[0][0]}..{sampled[-1][0]}"
        print(f"          {steps_range}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────

def render_dashboard(detail_tag=None):
    # Clear screen
    print("\033[2J\033[H", end="")
    print_header()

    if detail_tag:
        print_detail(detail_tag)
        return

    print_overview()
    print_bpb_chart()
    print_loss_curves()
    print_val_curves()
    print_seq_len_comparison()
    print_throughput()

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"  {DIM}Last updated: {timestamp}{RESET}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Calliope training dashboard")
    parser.add_argument("--live", action="store_true", help="Auto-refresh every 30s")
    parser.add_argument("--interval", type=int, default=30, help="Refresh interval in seconds")
    parser.add_argument("--run", type=str, default=None, help="Show detail for a single run")
    parser.add_argument("--no-color", action="store_true", help="Disable colors")
    args = parser.parse_args()

    if args.no_color:
        global BOLD, DIM, GREEN, YELLOW, CYAN, RED, MAGENTA, RESET
        BOLD = DIM = GREEN = YELLOW = CYAN = RED = MAGENTA = RESET = ""

    if not LOGS_DIR.exists():
        print(f"No logs directory found at {LOGS_DIR}")
        print("Run train_overnight.sh first.")
        sys.exit(1)

    if args.live:
        try:
            while True:
                render_dashboard(detail_tag=args.run)
                print(f"  {DIM}Refreshing every {args.interval}s (Ctrl+C to quit){RESET}")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n  Goodbye!")
    else:
        render_dashboard(detail_tag=args.run)


if __name__ == "__main__":
    main()
