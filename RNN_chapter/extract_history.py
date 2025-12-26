#!/usr/bin/env python3
"""Extract history JSON from NumPy checkpoint (.npz)."""

import argparse
import json
import os
from typing import Any, Dict

import numpy as np


def extract_history_from_checkpoint(ckpt_path: str, output_path: str) -> None:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint from {ckpt_path}...")
    data = np.load(ckpt_path, allow_pickle=True)
    history_json = data.get("history")

    if history_json is None:
        print("No history found; creating placeholder entry.")
        epoch = int(data.get("epoch", np.array([0]))[0])
        best_acc = float(data.get("best_acc", np.array([0.0]))[0])
        history = {
            "epochs": list(range(1, epoch + 2)),
            "train_loss": [0.0] * (epoch + 1),
            "train_acc": [0.0] * (epoch + 1),
            "val_loss": [0.0] * (epoch + 1),
            "val_acc": [best_acc] * (epoch + 1),
            "learning_rates": [0.0] * (epoch + 1),
        }
    else:
        history = json.loads(str(history_json[0]))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"History saved to {output_path}")
    if history["epochs"]:
        print(f"Epochs recorded: {len(history['epochs'])}")
        print(f"Last val acc: {history['val_acc'][-1]:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract training history from .npz checkpoint")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best.npz")
    parser.add_argument("--output", type=str, default="checkpoints/history.json")
    args = parser.parse_args()

    extract_history_from_checkpoint(args.ckpt, args.output)


if __name__ == "__main__":
    main()
