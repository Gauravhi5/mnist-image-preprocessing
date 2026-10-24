"""
MNIST Image Preprocessing
Performs normalization and Gaussian filtering without TensorFlow or Keras.
"""

import gzip
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download_file(filename):
    """Download an original compressed MNIST file if it is not present."""
    path = Path(filename)
    if not path.exists():
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(BASE_URL + filename, path)
    return path


def load_images(filename):
    """Read MNIST IDX image data into an array shaped (N, 28, 28)."""
    with gzip.open(filename, "rb") as file:
        data = file.read()
    return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(-1, 28, 28)


def load_labels(filename):
    """Read MNIST IDX label data into an array."""
    with gzip.open(filename, "rb") as file:
        data = file.read()
    return np.frombuffer(data, dtype=np.uint8, offset=8)


def main():
    paths = {key: download_file(name) for key, name in FILES.items()}

    train_images = load_images(paths["train_images"])
    train_labels = load_labels(paths["train_labels"])
    test_images = load_images(paths["test_images"])
    test_labels = load_labels(paths["test_labels"])

    print("Training images:", train_images.shape)
    print("Training labels:", train_labels.shape)
    print("Testing images:", test_images.shape)
    print("Testing labels:", test_labels.shape)

    # Normalize pixel intensities from [0, 255] to [0, 1].
    train_normalized = train_images.astype(np.float32) / 255.0
    test_normalized = test_images.astype(np.float32) / 255.0

    # Apply Gaussian smoothing to one example image.
    index = 0
    original = train_images[index]
    normalized = train_normalized[index]
    gaussian_result = gaussian_filter(normalized, sigma=0.8)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"Original (label {train_labels[index]})")
    axes[1].imshow(normalized, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Normalized")
    axes[2].imshow(gaussian_result, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Gaussian Filter (sigma=0.8)")

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()
    plt.savefig("preprocessing_results.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("Normalized training pixel range:",
          float(train_normalized.min()), "to", float(train_normalized.max()))
    print("Saved comparison figure: preprocessing_results.png")


if __name__ == "__main__":
    main()
