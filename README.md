# MNIST Image Preprocessing with Python

## Overview
This project demonstrates basic preprocessing on the MNIST handwritten-digit dataset without importing TensorFlow or Keras.

## Preprocessing steps
1. **Dataset loading:** Reads the original compressed MNIST IDX files directly.
2. **Normalization:** Converts pixel values from the range 0–255 to 0–1 by dividing by 255.
3. **Gaussian filtering:** Applies a Gaussian filter to one image to demonstrate smoothing.
4. **Visualization:** Displays the original, normalized, and filtered image side by side.

## Requirements
- Python 3
- NumPy
- SciPy
- Matplotlib

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run
```bash
python mnist_preprocessing.py
```

The script downloads the four original MNIST compressed files if they are not already present, then loads the training and testing data. It saves the comparison figure as `preprocessing_results.png`.

## Notes
- The script does not use TensorFlow or Keras.
- Gaussian smoothing can reduce small intensity variations, but excessive smoothing may blur digit edges.
- The MNIST dataset contains 60,000 training images and 10,000 test images, each 28 × 28 pixels.
