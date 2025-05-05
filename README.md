# Image Focus Detector

[![PyPI version](https://badge.fury.io/py/image-focus-detector.svg)](https://badge.fury.io/py/image-focus-detector) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

A Python package to detect salient or important regions (Regions of Interest - ROIs) within an image using a pre-trained Vision Transformer (ViT) model executed via ONNX Runtime.

This package analyzes an input image and outputs a list of potential regions that are considered visually important, along with their bounding boxes and saliency scores.

![example](overlay.png "Example") 

## Features

* Detects salient regions in images.
* Uses a Vision Transformer (ViT) based model.
* Leverages ONNX Runtime for efficient, cross-platform inference.
* Provides bounding boxes and saliency scores for detected ROIs.
* Allows adjustment of detection thresholds (`roi_threshold`, `score_threshold`, `min_contour_area`).

## Installation

You can install `img-focus` using pip. Ensure you have Python 3.8+ installed.

```bash
pip install img-focus
```

This will also install necessary dependencies, including onnxruntime, numpy, opencv-python-headless, Pillow, and albumentations.

(Note: If you haven't published the package to PyPI yet, users will need to install it from source or a local build file, e.g., pip install . or pip install -e . for editable mode.)

Usage
Here's a basic example of how to use the package to find ROIs in an image fetched from a URL:

```python
# -*- coding: utf-8 -*-
import numpy as np
import requests
from PIL import Image
import io
import cv2 # OpenCV est nécessaire pour le redimensionnement
from matplotlib import patches

from img_focus import Salency
import matplotlib.pyplot as plt # Importation de Matplotlib

# --- 1. Get an image (Example: Fetch from URL) ---
image_url = "https://upload.wikimedia.org/wikipedia/commons/b/bb/Knight-Daniel-Ridgway-Women-Washing-Clothes-by-a-Stream.jpg"

response = requests.get(image_url, timeout=15) # Ajout d'un timeout
response.raise_for_status()  # Lève une exception pour les erreurs HTTP

image_pil = Image.open(io.BytesIO(response.content)).convert("RGB")
image_np = np.array(image_pil)
original_h, original_w = image_np.shape[:2]

# --- 2. Initialize the detector ---
detector = Salency()

# --- 3. Find Regions of Interest (ROIs) & Get Heatmap ---
rois = detector.find_regions_of_interest(
    image_pil,
    score_threshold=0.7
)
heatmap_raw = detector.get_heatmap(image_pil)

# --- 4. Process and display results ---
if rois:
    print(f"\nFound {len(rois)} region(s) of interest:")
    for i, roi in enumerate(rois):
        print(f"  --- ROI {i+1} ---")
        print(f"    Max Score: {roi['score_max']:.4f}")
        print(f"    Bounding Box (x, y, width, height): [{roi['x']}, {roi['y']}, {roi['w']}, {roi['h']}]")
        x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']
else:
    print("\nNo significant regions of interest found with the current thresholds.")


# --- 5. Visualization with Matplotlib ---
heatmap_resized = cv2.resize(heatmap_raw, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
heatmap_normalized = cv2.normalize(heatmap_resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

ax.imshow(image_np)

im = ax.imshow(heatmap_normalized, cmap='jet', alpha=0.5, vmin= 0.5)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
if rois:
    for i, roi in enumerate(rois):
        x = int(roi.get('x')*original_w)
        y = int(roi.get('y')*original_h)
        w = int(roi.get('w')*original_w)
        h = int(roi.get('h')*original_h)

        if all(v is not None for v in [x, y, w, h]):
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            ax.text(x + 5, y + 20, f"ROI {i + 1}", color='lime', fontsize=10, weight='bold')

ax.axis('off')
plt.tight_layout()
plt.show()
```
