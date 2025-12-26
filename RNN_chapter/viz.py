#!/usr/bin/env python3
"""Visualization utilities for the NumPy CNN+RNN classifier."""

import argparse
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix

from data_utils import denormalize, load_cifar10
from models.cnn_rnn_numpy import CNNRNNClassifier, accuracy, softmax_cross_entropy

CLASS_NAMES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


def load_checkpoint(ckpt_path: str) -> CNNRNNClassifier:
    model, _, _, _ = CNNRNNClassifier.load(ckpt_path)
    return model


def collect_predictions(model: CNNRNNClassifier, images: np.ndarray,
                        labels: np.ndarray, batch_size: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = []
    feats = []
    iterator = range(0, images.shape[0], batch_size)
    for start in iterator:
        x = images[start:start + batch_size]
        logits, cache = model.forward(x, retain_cache=False)
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        prob = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs.append(prob)
        feats.append(cache['last_hidden'])
    probs_np = np.concatenate(probs, axis=0)
    feats_np = np.concatenate(feats, axis=0)
    preds = np.argmax(probs_np, axis=1)
    return probs_np, preds, feats_np


def plot_confusion(cm: np.ndarray, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap='viridis')
    ax.set_xticks(range(10))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.set_yticks(range(10))
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_reliability(max_probs: np.ndarray, correct: np.ndarray, save_path: str) -> None:
    bins = np.linspace(0, 1, 11)
    mids = (bins[:-1] + bins[1:]) / 2
    accs, confs = [], []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (max_probs >= b0) & (max_probs < b1)
        if mask.any():
            accs.append(correct[mask].mean())
            confs.append(max_probs[mask].mean())
        else:
            accs.append(0)
            confs.append(0)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], 'k--')
    ax.bar(mids, accs, width=0.09, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title('Reliability Diagram')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def tsne_plot(features: np.ndarray, labels: np.ndarray, save_path: str) -> None:
    max_points = min(3000, features.shape[0])
    idx = np.random.choice(features.shape[0], max_points, replace=False)
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    embed = tsne.fit_transform(features[idx])
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in range(10):
        mask = labels[idx] == c
        ax.scatter(embed[mask, 0], embed[mask, 1], s=6, label=CLASS_NAMES[c])
    ax.legend(fontsize=7, ncol=2)
    ax.set_title('t-SNE of RNN Hidden States')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualization for NumPy CNN+RNN model")
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--ckpt', type=str, default='checkpoints/best.npz')
    parser.add_argument('--outdir', type=str, default='viz_outputs')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    train_images, train_labels, test_images, test_labels, mean, std = load_cifar10(args.data_dir)
    model = load_checkpoint(args.ckpt)

    probs, preds, feats = collect_predictions(model, test_images, test_labels, args.batch_size)
    loss, _ = softmax_cross_entropy(np.log(probs + 1e-9), test_labels)
    acc = accuracy(np.log(probs + 1e-9), test_labels)
    print(f"Test Loss: {loss:.3f} | Test Acc: {acc:.2f}%")

    cm = confusion_matrix(test_labels, preds)
    plot_confusion(cm, os.path.join(args.outdir, 'confusion_matrix.png'))

    max_probs = probs.max(axis=1)
    correct = (preds == test_labels).astype(np.float32)
    plot_reliability(max_probs, correct, os.path.join(args.outdir, 'reliability.png'))

    tsne_plot(feats, test_labels, os.path.join(args.outdir, 'tsne.png'))

    print(f"Visualizations saved to {args.outdir}")


if __name__ == '__main__':
    main()
