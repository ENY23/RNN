import argparse
import json
import os
import time
from typing import Dict

import numpy as np

from data_utils import BatchIterator, load_cifar10
from models.cnn_rnn_numpy import (
    CNNRNNClassifier,
    accuracy,
    softmax_cross_entropy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NumPy CNN+RNN CIFAR-10 Trainer (with history log)")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default="")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def evaluate(model: CNNRNNClassifier, images: np.ndarray, labels: np.ndarray,
             batch_size: int = 256) -> Dict[str, float]:
    iterator = BatchIterator(images, labels, batch_size, shuffle=False, augment=False, seed=123)
    all_logits = []
    for x_batch, y_batch in iterator:
        logits, _ = model.forward(x_batch, retain_cache=False)
        all_logits.append(logits)
    logits = np.concatenate(all_logits, axis=0)
    loss, _ = softmax_cross_entropy(logits, labels)
    acc = accuracy(logits, labels)
    return {"loss": loss, "acc": acc}


def save_history(history: Dict[str, list], ckpt_dir: str) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, "history.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"History written to {path}")


def train() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_images, train_labels, test_images, test_labels, mean, std = load_cifar10(args.data_dir)
    model = CNNRNNClassifier(seed=args.seed)

    start_epoch = 0
    best_acc = 0.0
    history = {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rates": [],
    }

    if args.resume and os.path.isfile(args.resume):
        model, start_epoch, best_acc, hist = CNNRNNClassifier.load(args.resume)
        if hist is not None:
            history = hist
        print(f"Resumed from {args.resume} at epoch {start_epoch}, best_acc={best_acc:.2f}")

    iterator = BatchIterator(
        train_images,
        train_labels,
        args.batch_size,
        shuffle=True,
        augment=True,
        seed=args.seed,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        count = 0
        start = time.time()

        for x_batch, y_batch in iterator:
            logits, cache = model.forward(x_batch, retain_cache=True)
            loss, grad = softmax_cross_entropy(logits, y_batch)
            batch_acc = accuracy(logits, y_batch)
            grads = model.backward(grad, cache)
            grad_norm = model.apply_gradients(
                grads,
                lr=args.lr,
                weight_decay=args.weight_decay,
                grad_clip=args.grad_clip,
            )
            bs = x_batch.shape[0]
            epoch_loss += loss * bs
            epoch_acc += batch_acc * bs
            count += bs

        epoch_loss /= count
        epoch_acc /= count
        val_metrics = evaluate(model, test_images, test_labels)
        elapsed = time.time() - start

        history["epochs"].append(epoch + 1)
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["learning_rates"].append(args.lr)

        msg = (
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {epoch_loss:.3f} Acc: {epoch_acc:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.3f} Acc: {val_metrics['acc']:.2f}% | "
            f"GradNorm: {grad_norm:.2f} | Time: {elapsed:.1f}s"
        )
        print(msg)

        is_best = val_metrics["acc"] > best_acc
        if is_best:
            best_acc = val_metrics["acc"]

        ckpt_path = os.path.join(args.checkpoint_dir, "last.npz")
        model.save(ckpt_path, epoch=epoch, best_acc=best_acc, history=history)
        if is_best:
            model.save(os.path.join(args.checkpoint_dir, "best.npz"), epoch=epoch, best_acc=best_acc, history=history)

        if (epoch + 1) % 5 == 0 or is_best:
            save_history(history, args.checkpoint_dir)

        iterator = BatchIterator(
            train_images,
            train_labels,
            args.batch_size,
            shuffle=True,
            augment=True,
            seed=int(np.random.randint(0, 1e9)),
        )

    save_history(history, args.checkpoint_dir)
    print(f"Training complete. Best Acc: {best_acc:.2f}%")


if __name__ == "__main__":
    train()
