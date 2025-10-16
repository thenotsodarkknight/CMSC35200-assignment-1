#!/usr/bin/env python3
"""
Task 5: Fine-tune nanoGPT on Shakespeare and plot the learning curve.

This script optionally runs the nanoGPT training loop (using the same
configuration executed manually earlier) and then parses the training
log to produce a plot of loss vs. tokens seen.
"""
from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
NANOGPT_DIR = PROJECT_ROOT / "nanoGPT"
LOG_PATH = NANOGPT_DIR / "out" / "training_log.txt"
PLOT_PATH = PROJECT_ROOT / "outputs" / "nanogpt_learning_curve.png"
SAMPLE_PATH = PROJECT_ROOT / "outputs" / "nanogpt_sample.txt"


TRAIN_CMD = [
    "python3",
    "train.py",
    "--device",
    "mps",
    "--dtype",
    "float32",
    "--compile",
    "False",
    "--dataset",
    "shakespeare",
    "--n_layer",
    "4",
    "--n_head",
    "4",
    "--n_embd",
    "64",
    "--block_size",
    "64",
    "--batch_size",
    "8",
    "--init_from",
    "gpt2",
    "--eval_interval",
    "100",
    "--eval_iters",
    "100",
    "--max_iters",
    "300",
    "--bias",
    "True",
]


@dataclass
class IterRecord:
    iteration: int
    tokens_seen: int
    train_loss: float


@dataclass
class EvalRecord:
    step: int
    tokens_seen: int
    train_loss: float
    val_loss: float


def run_training(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with subprocess.Popen(
        TRAIN_CMD,
        cwd=NANOGPT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc, log_path.open("w", encoding="utf-8") as log_file:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"Training command failed with exit code {ret}")


def parse_log(log_path: Path) -> Tuple[int, List[IterRecord], List[EvalRecord]]:
    text = log_path.read_text(encoding="utf-8")
    token_match = re.search(r"tokens per iteration will be: ([\d,]+)", text)
    if not token_match:
        raise ValueError("Could not find tokens-per-iteration line in log.")
    tokens_per_iter = int(token_match.group(1).replace(",", ""))

    iter_records: List[IterRecord] = []
    for match in re.finditer(r"iter (\d+): loss ([\d.]+)", text):
        iteration = int(match.group(1))
        loss = float(match.group(2))
        tokens_seen = (iteration + 1) * tokens_per_iter
        iter_records.append(IterRecord(iteration, tokens_seen, loss))

    eval_records: List[EvalRecord] = []
    for match in re.finditer(
        r"step (\d+): train loss ([\d.]+), val loss ([\d.]+)", text
    ):
        step = int(match.group(1))
        train_loss = float(match.group(2))
        val_loss = float(match.group(3))
        tokens_seen = (step + 1) * tokens_per_iter
        eval_records.append(EvalRecord(step, tokens_seen, train_loss, val_loss))

    return tokens_per_iter, iter_records, eval_records


def make_plot(iter_records: List[IterRecord], eval_records: List[EvalRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    df_iters = pd.DataFrame([r.__dict__ for r in iter_records])
    df_evals = pd.DataFrame([r.__dict__ for r in eval_records])

    plt.figure(figsize=(10, 6))
    if not df_iters.empty:
        plt.plot(
            df_iters["tokens_seen"],
            df_iters["train_loss"],
            label="Train loss (per iter)",
            linewidth=1,
            alpha=0.7,
        )
    if not df_evals.empty:
        plt.scatter(
            df_evals["tokens_seen"],
            df_evals["val_loss"],
            label="Validation loss (eval checkpoints)",
            color="orange",
        )
    plt.xlabel("Tokens seen")
    plt.ylabel("Loss")
    plt.title("nanoGPT Shakespeare Fine-tuning Learning Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved learning curve plot to {output}")


def write_sample(sample_path: Path, start: str = "to be") -> None:
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Use nanoGPT's sample.py with the latest checkpoint in out/
        out = subprocess.check_output(
            [
                "python3",
                "sample.py",
                "--dtype",
                "float32",
                "--num_samples",
                "3",
                "--max_new_tokens",
                "40",
                "--start",
                start,
            ],
            cwd=NANOGPT_DIR,
            text=True,
        )
        sample_path.write_text(out, encoding="utf-8")
        print(f"Saved text sample to {sample_path}")
    except Exception as e:  # noqa: BLE001
        print(f"Sampling failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run nanoGPT fine-tuning and plot the learning curve.")
    parser.add_argument("--skip-train", action="store_true", help="Skip running training and just parse the existing log.")
    args = parser.parse_args()

    if not NANOGPT_DIR.exists():
        raise FileNotFoundError(f"nanoGPT directory not found at {NANOGPT_DIR}")

    if not args.skip_train:
        print("Starting nanoGPT training (this may take a while)...")
        run_training(LOG_PATH)
    elif not LOG_PATH.exists():
        raise FileNotFoundError(f"{LOG_PATH} not found. Run without --skip-train to generate it.")

    tokens_per_iter, iter_records, eval_records = parse_log(LOG_PATH)
    print(f"Parsed {len(iter_records)} iteration records and {len(eval_records)} evaluation checkpoints.")
    print(f"Tokens per iteration: {tokens_per_iter}")
    make_plot(iter_records, eval_records, PLOT_PATH)
    write_sample(SAMPLE_PATH)


if __name__ == "__main__":
    main()
