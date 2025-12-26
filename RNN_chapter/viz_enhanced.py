#!/usr/bin/env python3
"""Enhanced visualization suite for the NumPy CNN+RNN model."""

import argparse
import json
import os
from typing import Dict, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix

from data_utils import BatchIterator, denormalize, load_cifar10
from models.cnn_rnn_numpy import CNNRNNClassifier, accuracy, softmax_cross_entropy

# Use colorblind-friendly palettes (avoiding red-blue as per user preference)
COLORS = {
    'train': '#FFA500',      # Orange
    'test': '#9370DB',       # Purple
    'accent1': '#2E8B57',    # SeaGreen
    'accent2': '#FFD700',    # Gold
    'accent3': '#8B4789',    # DarkOrchid
}

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def setup_plot_style():
    """Setup matplotlib style for publication-quality figures"""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 13,
        'font.family': 'DejaVu Sans',
    })

def load_model(ckpt_path: str) -> Tuple[CNNRNNClassifier, Dict[str, float]]:
    model, epoch, best_acc, history = CNNRNNClassifier.load(ckpt_path)
    meta = {"epoch": epoch, "best_acc": best_acc}
    return model, meta

def collect_predictions(model: CNNRNNClassifier, images: np.ndarray,
                        labels: np.ndarray, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    iterator = BatchIterator(images, labels, batch_size, shuffle=False, augment=False, seed=123)
    logits_all, feats_all = [], []
    for x, _ in iterator:
        logits, cache = model.forward(x, retain_cache=False)
        logits_all.append(logits)
        feats_all.append(cache['last_hidden'])
    logits = np.concatenate(logits_all, axis=0)
    feats = np.concatenate(feats_all, axis=0)
    return logits, feats, labels

# ==================== Training Dynamics ====================

def plot_training_curves(history_file, save_path):
    """Plot loss and accuracy vs epoch for train/test"""
    if not os.path.exists(history_file):
        print(f"Warning: {history_file} not found, skipping training curves")
        return
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = history['epochs']
    train_loss = history['train_loss']
    train_acc = history['train_acc']
    val_loss = history['val_loss']
    val_acc = history['val_acc']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(epochs, train_loss, label='Train', color=COLORS['train'], linewidth=2)
    ax1.plot(epochs, val_loss, label='Test', color=COLORS['test'], linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_acc, label='Train', color=COLORS['train'], linewidth=2)
    ax2.plot(epochs, val_acc, label='Test', color=COLORS['test'], linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_learning_rate_schedule(history_file, save_path):
    """Plot learning rate vs epoch"""
    if not os.path.exists(history_file):
        print(f"Warning: {history_file} not found, skipping LR schedule")
        return
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = history['epochs']
    lrs = history.get('learning_rates', [])
    
    if not lrs:
        print("No learning rate history found")
        return
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, lrs, color=COLORS['accent1'], linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# ==================== Performance Analysis ====================

def plot_confusion_matrix(y_true, y_pred, save_path, normalize=True):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    
    if normalize:
        cm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, aspect='auto', cmap='YlOrBr')
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_yticklabels(CLASS_NAMES)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Proportion' if normalize else 'Count', rotation=270, labelpad=15)
    
    # Add text annotations
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            text = ax.text(j, i, f'{cm[i, j]:.2f}' if normalize else f'{int(cm[i, j])}',
                          ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black",
                          fontsize=8)
    
    ax.set_title('Confusion Matrix (Normalized)' if normalize else 'Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_per_class_metrics(y_true, y_pred, save_path):
    """Plot precision, recall, F1 for each class"""
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, 
                                   output_dict=True, digits=4, zero_division=0)
    
    classes = CLASS_NAMES
    precision = [report[c]['precision'] for c in classes]
    recall = [report[c]['recall'] for c in classes]
    f1 = [report[c]['f1-score'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, precision, width, label='Precision', color='#FFA500', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', color='#9370DB', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', color='#2E8B57', alpha=0.8)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    
    # Also save as CSV
    csv_path = save_path.replace('.png', '.csv')
    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
        for c in classes:
            writer.writerow([c, report[c]['precision'], report[c]['recall'], 
                           report[c]['f1-score'], report[c]['support']])
        avg = report['weighted avg']
        writer.writerow(['Weighted Avg', avg['precision'], avg['recall'], 
                        avg['f1-score'], avg['support']])
    print(f"Saved: {csv_path}")

def plot_topk_accuracy(probs, y_true, save_path, k_values=[1, 3, 5]):
    """Plot top-k accuracy bar chart"""
    topk_accs = []
    
    for k in k_values:
        if k <= 0 or k > probs.shape[1]:
            raise ValueError("k must be between 1 and the number of classes")
        topk_idx = np.argpartition(-probs, kth=k-1, axis=1)[:, :k]
        correct = np.any(topk_idx == y_true[:, None], axis=1)
        acc = correct.mean() * 100
        topk_accs.append(acc)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(k_values)), topk_accs, color=['#FFA500', '#9370DB', '#2E8B57'], 
                  alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Top-K')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Top-K Accuracy Performance')
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels([f'Top-{k}' for k in k_values])
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, topk_accs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_misclassified_gallery(raw_images, y_true, y_pred, probs, save_path, n=16):
    wrong_idx = np.where(y_true != y_pred)[0]
    if wrong_idx.size == 0:
        print("No misclassifications to plot.")
        return
    conf = probs[wrong_idx, y_pred[wrong_idx]]
    order = np.argsort(-conf)[:n]
    pick = wrong_idx[order]
    cols = 8
    rows = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4))
    axes = axes.flatten()
    for i, idx in enumerate(pick):
        img = raw_images[idx].transpose(1, 2, 0)
        ax = axes[i]
        ax.imshow(np.clip(img, 0, 1))
        ax.axis('off')
        pred_conf = probs[idx, y_pred[idx]]
        ax.set_title(
            f"Pred: {CLASS_NAMES[y_pred[idx]]}\n"
            f"True: {CLASS_NAMES[y_true[idx]]}\n"
            f"Conf: {pred_conf:.2f}", fontsize=7
        )
    for j in range(i + 1, rows * cols):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# ==================== Calibration & Confidence ====================

def plot_reliability_diagram(probs, y_true, y_pred, save_path, n_bins=10):
    """Reliability diagram and ECE calculation"""
    max_probs = probs.max(axis=1)
    correct = (y_true == y_pred).astype(np.float32)
    
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(max_probs, bins) - 1
    
    bin_acc, bin_conf, bin_counts = [], [], []
    for b in range(n_bins):
        mask = inds == b
        if mask.sum() == 0:
            bin_acc.append(0.0)
            bin_conf.append(0.0)
            bin_counts.append(0)
        else:
            bin_acc.append(correct[mask].mean())
            bin_conf.append(max_probs[mask].mean())
            bin_counts.append(mask.sum())
    
    bin_acc = np.array(bin_acc)
    bin_conf = np.array(bin_conf)
    bin_counts = np.array(bin_counts)
    
    ece = np.sum(np.abs(bin_acc - bin_conf) * (bin_counts / max(1, bin_counts.sum())))
    
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    ax.bar((bins[:-1] + bins[1:]) / 2.0, bin_acc, width=1.0/n_bins, 
           alpha=0.7, color=COLORS['train'], edgecolor='black', label='Model Output')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Reliability Diagram\nECE = {ece:.4f}', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path} (ECE={ece:.4f})")

def plot_confidence_distribution(probs, y_true, y_pred, save_path):
    """Confidence distribution histogram for correct vs incorrect"""
    max_probs = probs.max(axis=1)
    correct_conf = max_probs[y_true == y_pred]
    incorrect_conf = max_probs[y_true != y_pred]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(correct_conf, bins=30, alpha=0.6, label=f'Correct (n={len(correct_conf)})', 
            color=COLORS['accent1'], edgecolor='black')
    ax.hist(incorrect_conf, bins=30, alpha=0.6, label=f'Incorrect (n={len(incorrect_conf)})', 
            color=COLORS['train'], edgecolor='black')
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Confidence Distribution: Correct vs Incorrect Predictions', fontsize=13)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics in bottom right
    textstr = f'Correct: μ={correct_conf.mean():.3f}, σ={correct_conf.std():.3f}\n'
    textstr += f'Incorrect: μ={incorrect_conf.mean():.3f}, σ={incorrect_conf.std():.3f}'
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# ==================== Representation & Interpretability ====================

def plot_tsne(features, labels, save_path, max_points=5000, seed=42):
    """t-SNE visualization of penultimate layer features"""
    X = features
    y = labels
    
    if X.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=max_points, replace=False)
        X = X[idx]
        y = y[idx]
    
    print(f"Running t-SNE on {X.shape[0]} samples...")
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', 
                init='pca', random_state=seed, verbose=1)
    emb = tsne.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use colorblind-friendly palette
    palette = sns.color_palette("tab10", 10)
    
    for c in range(10):
        mask = y == c
        ax.scatter(emb[mask, 0], emb[mask, 1], s=20, alpha=0.6, 
                  label=CLASS_NAMES[c], color=palette[c])
    
    ax.legend(markerscale=1.5, fontsize=9, ncol=2, loc='best')
    ax.set_title('t-SNE Visualization of Penultimate Layer Features', fontsize=13)
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_raw_gallery(raw_images, y_true, y_pred, save_path, n=16):
    cols = 8
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4))
    axes = axes.flatten()
    for i in range(n):
        idx = i % raw_images.shape[0]
        img = raw_images[idx].transpose(1, 2, 0)
        ax = axes[i]
        ax.imshow(np.clip(img, 0, 1))
        ax.axis('off')
        ax.set_title(f"T:{CLASS_NAMES[y_true[idx]]}\nP:{CLASS_NAMES[y_pred[idx]]}", fontsize=7)
    for j in range(i + 1, rows * cols):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# ==================== Model Statistics ====================

def plot_model_statistics(model: CNNRNNClassifier, save_path: str) -> None:
    params = model.state_dict()
    total = sum(p.size for p in params.values())
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(['Parameters'], [total / 1e6], color=COLORS['train'], edgecolor='black')
    ax.set_ylabel('Count (Millions)')
    ax.set_title('Model Parameter Count')
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{bar.get_height():.2f}M",
                ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# ==================== Main Function ====================

def main():
    parser = argparse.ArgumentParser(description="Enhanced Visualization for NumPy CNN+RNN")
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--ckpt', type=str, default='checkpoints/best.npz')
    parser.add_argument('--history', type=str, default='checkpoints/history.json')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--outdir', type=str, default='visualizations')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    setup_plot_style()
    np.random.seed(args.seed)

    print(f"Loading model from {args.ckpt}...")
    model, meta = load_model(args.ckpt)
    print(f"Model ready. Best accuracy: {meta['best_acc']:.2f}%")

    train_images, train_labels, test_images, test_labels, mean, std = load_cifar10(args.data_dir)
    # Store raw images for misclassification gallery
    raw_test = denormalize(test_images, mean, std)

    print("Collecting predictions...")
    logits, feats, labels = collect_predictions(model, test_images, test_labels, args.batch_size)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)
    acc = accuracy(logits, labels)
    loss, _ = softmax_cross_entropy(logits, labels)
    print(f"Test Loss: {loss:.3f} | Test Acc: {acc:.2f}%")

    print("Generating visualizations...")
    plot_training_curves(args.history, os.path.join(args.outdir, '1_training_curves.png'))
    plot_learning_rate_schedule(args.history, os.path.join(args.outdir, '1_lr_schedule.png'))

    plot_confusion_matrix(labels, preds, os.path.join(args.outdir, '2_confusion_matrix.png'))
    plot_per_class_metrics(labels, preds, os.path.join(args.outdir, '2_per_class_metrics.png'))
    plot_topk_accuracy(probs, labels, os.path.join(args.outdir, '2_topk_accuracy.png'))
    plot_misclassified_gallery(raw_test, labels, preds, probs, os.path.join(args.outdir, '2_misclassified.png'))

    plot_reliability_diagram(probs, labels, preds, os.path.join(args.outdir, '3_reliability_diagram.png'))
    plot_confidence_distribution(probs, labels, preds, os.path.join(args.outdir, '3_confidence_distribution.png'))

    plot_tsne(feats, labels, os.path.join(args.outdir, '4_tsne.png'))
    plot_raw_gallery(raw_test, labels, preds, os.path.join(args.outdir, '4_raw_gallery.png'))

    plot_model_statistics(model, os.path.join(args.outdir, '5_model_statistics.png'))

    print(f"Done. Outputs saved to {args.outdir}")

if __name__ == '__main__':
    main()
