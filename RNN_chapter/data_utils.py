import os
import pickle
import tarfile
import urllib.request
from dataclasses import dataclass
from typing import Iterator, Tuple, Optional

import numpy as np

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCHIVE_NAME = "cifar-10-python.tar.gz"
EXTRACTED_FOLDER = "cifar-10-batches-py"


def _download_cifar10(data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    archive_path = os.path.join(data_dir, ARCHIVE_NAME)
    if not os.path.exists(archive_path):
        print(f"Downloading CIFAR-10 to {archive_path}...")
        urllib.request.urlretrieve(CIFAR10_URL, archive_path)
    extracted_path = os.path.join(data_dir, EXTRACTED_FOLDER)
    if not os.path.exists(extracted_path):
        print("Extracting CIFAR-10 archive...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=data_dir)


def _load_batch(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    images = data["data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    labels = np.array(data["labels"], dtype=np.int64)
    return images, labels


def load_cifar10(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _download_cifar10(data_dir)
    base = os.path.join(data_dir, EXTRACTED_FOLDER)
    train_images, train_labels = [], []
    for i in range(1, 6):
        imgs, lbls = _load_batch(os.path.join(base, f"data_batch_{i}"))
        train_images.append(imgs)
        train_labels.append(lbls)
    train_images = np.concatenate(train_images, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    test_images, test_labels = _load_batch(os.path.join(base, "test_batch"))

    mean = train_images.mean(axis=(0, 2, 3), keepdims=True)
    std = train_images.std(axis=(0, 2, 3), keepdims=True) + 1e-7
    train_images = (train_images - mean) / std
    test_images = (test_images - mean) / std
    return train_images, train_labels, test_images, test_labels, mean, std


def denormalize(images: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return images * std + mean


def random_crop_flip(batch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n, c, h, w = batch.shape
    padded = np.pad(batch, ((0, 0), (0, 0), (4, 4), (4, 4)), mode="reflect")
    augmented = np.empty_like(batch)
    for i in range(n):
        top = rng.integers(0, 9)
        left = rng.integers(0, 9)
        crop = padded[i, :, top:top + h, left:left + w]
        if rng.random() < 0.5:
            crop = crop[:, :, ::-1]
        augmented[i] = crop
    return augmented


@dataclass
class BatchIterator:
    images: np.ndarray
    labels: np.ndarray
    batch_size: int
    shuffle: bool = True
    augment: bool = False
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        indices = np.arange(self.images.shape[0])
        if self.shuffle:
            self.rng.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            excerpt = indices[start:start + self.batch_size]
            batch_x = self.images[excerpt].copy()
            if self.augment:
                batch_x = random_crop_flip(batch_x, self.rng)
            yield batch_x.astype(np.float32, copy=False), self.labels[excerpt]
