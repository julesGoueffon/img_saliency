from typing import Union, List, Dict

import albumentations as A
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from importlib_resources import files


TARGET_SIZE = 224 # Expected input size
IMG_MEAN = (0.485, 0.456, 0.406) # ImageNet mean
IMG_STD = (0.229, 0.224, 0.225) # ImageNet standard deviation

# Ignore areas smaller than this (in square pixels)
# Note: This constant seems defined by comment only, check if used or intended for find_and_rois's min_area

NUM_CLASSES = 1 # Number of output classes (1 for the saliency map)

INPUT_NAME = "input_image"
OUTPUT_NAME = "output_importance" # Renamed from "output_saliency" if that was intended, keeping as is for direct translation

class Salency:
    _MODEL_SUBPATH = "models/vit_s_focus.onnx"

    def __init__(self):

        providers = ['CPUExecutionProvider']

        model_path = files("img_focus").joinpath(self._MODEL_SUBPATH)

        # Convert the pathlib.Path object to a string, as ONNX Runtime expects a string.

        # --- Optional: Keep the debug print if useful ---
        # ONNXRuntime needs the path as a string
        self.session = ort.InferenceSession(str(model_path), providers=providers)


    def find_regions_of_interest(self, img:Image,
                                 roi_threshold:float = 0.8,
                                 score_threshold:Union[float,None]= 0.9,
                                 min_contour_area:int = 100) -> List[Dict]:
        """
        Finds regions of interest in an image based on a saliency heatmap.

        Args:
            img: The input PIL Image.
            roi_threshold: The threshold used to binarize the heatmap for contour finding.
            score_threshold: Optional minimum score_max for a ROI to be included in the results.
            min_contour_area: Minimum area (in pixels of the resized heatmap) for a contour to be considered an ROI.

        Returns:
            A list of dictionaries, each representing an ROI with its properties.
        """
        final_heatmap = self.get_heatmap(img)
        image_with_rois = find_and_extract_rois( # Changed function name to reflect its action better
            final_heatmap,
            roi_threshold,
            min_contour_area
        )
        if score_threshold is not None: # Check explicitly for None
            image_with_rois = list(filter(lambda x: x["score_max"] >= score_threshold, image_with_rois))
        return image_with_rois

    def get_heatmap(self, img:Image) -> np.ndarray:
        """
        Generates the saliency heatmap for the input image.

        Args:
            img: The input PIL Image.

        Returns:
            A numpy array representing the saliency heatmap, resized to match the
            image's aspect ratio before padding.
        """
        image_array, _, _, pre_padding_size_hw = preprocess_image(
            img, TARGET_SIZE, IMG_MEAN, IMG_STD
        )
        # The model expects input shape (batch_size, channels, height, width)
        results = self.session.run([OUTPUT_NAME], {INPUT_NAME: image_array})
        output_data = results[0]

        final_heatmap = get_prediction_heatmap(
            output_data,
            pre_padding_size_hw
        )
        return final_heatmap


# --- Utility Functions ---

def preprocess_image(image: Image, target_size: int, img_mean: tuple, img_std: tuple):
    """
    Preprocesses the input image for the model.

    Args:
        image: The input PIL Image.
        target_size: The target size (height and width) for model input.
        img_mean: The mean values for normalization.
        img_std: The standard deviation values for normalization.

    Returns:
        A tuple containing:
            - image_np_trans: The processed image as a numpy array (BCHW).
            - original_size_wh: Original width and height of the image.
            - original_image_np: Original image as a numpy array (HWC).
            - pre_padding_size_hw: Height and width after resizing but before padding.
    """
    original_size_wh = image.size
    original_image_np = np.array(image)
    original_h, original_w = original_image_np.shape[:2]

    # Define image transformations
    transforms = A.Compose([
        # Resize the longest side to target_size, maintaining aspect ratio
        A.LongestMaxSize(max_size=target_size, interpolation=cv2.INTER_AREA),
        # Pad the image to be target_size x target_size
        A.PadIfNeeded(min_height=target_size, min_width=target_size,
                      border_mode=cv2.BORDER_CONSTANT), # Use value=0 for padding
        # Normalize using ImageNet stats
        A.Normalize(mean=img_mean, std=img_std),
    ])

    processed = transforms(image=original_image_np)
    image_np_trans = processed['image'] # HWC format after albumentations

    # Calculate size *before* padding was applied
    ratio = min(target_size / original_h, target_size / original_w)
    resized_h, resized_w = int(original_h * ratio), int(original_w * ratio)
    pre_padding_size_hw = (resized_h, resized_w) # Store H, W before padding

    # Add batch dimension and transpose to CHW format expected by PyTorch/ONNX models
    image_np_trans = np.expand_dims(image_np_trans, axis=0) # Add batch dimension -> BHWC
    image_np_trans = image_np_trans.transpose(0, 3, 1, 2) # Transpose to BCHW

    return image_np_trans.astype(np.float32), original_size_wh, original_image_np, pre_padding_size_hw # Ensure float32


def get_prediction_heatmap(scores: np.ndarray, pre_padding_size_hw: tuple) -> np.ndarray:
    """
    Processes the raw model output scores into a usable heatmap,
    correcting for padding and resizing.

    Args:
        scores: Raw output from the ONNX model (expected shape [1, 1, H, W]).
        pre_padding_size_hw: The height and width of the image *before* padding was added
                             during preprocessing (but after resizing).

    Returns:
        The final heatmap, cropped to remove padding effects and representing
        the saliency on the aspect-ratio-preserved image.
    """

    # Validate score shape
    if scores.shape[0] != 1 or scores.shape[1] != NUM_CLASSES:
         raise ValueError(f"Unexpected output shape: {scores.shape}. Expected (1, {NUM_CLASSES}, H, W)")

    # Remove batch and channel dimensions (assuming NUM_CLASSES=1)
    heatmap_pred_np = scores.squeeze()
    if heatmap_pred_np.ndim != 2:
        raise ValueError(f"Heatmap not 2D after squeeze: {heatmap_pred_np.shape}")

    # --- Padding Correction and Resizing ---
    # Resize the heatmap from model output size to the target input size (e.g., 224x224)
    heatmap_target_size = cv2.resize(heatmap_pred_np.astype(np.float32),
                                     (TARGET_SIZE, TARGET_SIZE), # Resize to square target size
                                     interpolation=cv2.INTER_LINEAR)

    # Calculate the padding that was added during preprocessing
    resized_h, resized_w = pre_padding_size_hw # H, W *before* padding
    pad_h_total = TARGET_SIZE - resized_h
    pad_w_total = TARGET_SIZE - resized_w
    pad_top = pad_h_total // 2
    pad_left = pad_w_total // 2

    # Crop the heatmap to remove the padded areas, effectively reversing the PadIfNeeded step
    # The slicing indices correspond to the *actual content* area within the padded heatmap
    heatmap_cropped = heatmap_target_size[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w]

    if heatmap_cropped.size == 0:
        # This could happen if pre_padding_size_hw is invalid (e.g., contains zeros)
        raise ValueError(f"Cropping the heatmap resulted in an empty image. "
                         f"Target size: {TARGET_SIZE}x{TARGET_SIZE}, "
                         f"Pre-padding size: {pre_padding_size_hw}, "
                         f"Pads (top, left): ({pad_top}, {pad_left})")

    return heatmap_cropped


# --- New Function: ROI Detection and Calculation ---
# (Replace the old find_and_rois function with this one in your img_focus.py file or equivalent)

def find_and_extract_rois(heatmap_np: np.ndarray, threshold_value: float, min_area: int) -> List[Dict]:
    """
    Finds regions of interest (ROIs) on the provided heatmap, filters by minimum area,
    and returns a list of dictionaries containing the ROI data, including coordinates
    (bounding box and max point) relative to the dimensions of the input heatmap (heatmap_np).

    Args:
        heatmap_np: The input saliency heatmap (numpy array, expected to be 2D).
                    This should be the heatmap *after* padding removal (from get_prediction_heatmap).
        threshold_value: A value between 0 and 1 used for thresholding the normalized heatmap.
                         Pixels above this threshold (relative to max) form the binary mask.
        min_area: The minimum area (in pixels of the heatmap_np) for a contour to be considered an ROI.

    Returns:
        A list of dictionaries, where each dictionary represents an ROI and contains:
            - 'score_mean': Mean heatmap value within the ROI contour.
            - 'score_max': Maximum heatmap value within the ROI contour.
            - 'x': Relative x-coordinate of the bounding box top-left corner (0-1).
            - 'y': Relative y-coordinate of the bounding box top-left corner (0-1).
            - 'w': Relative width of the bounding box (0-1).
            - 'h': Relative height of the bounding box (0-1).
            - 'x_max': Relative x-coordinate of the point with the maximum score (0-1).
            - 'y_max': Relative y-coordinate of the point with the maximum score (0-1).
        The list is sorted by 'score_max' in descending order.
    """
    # 1. Check if the heatmap is valid
    if heatmap_np is None or heatmap_np.size == 0 or heatmap_np.ndim != 2:
        print(f"Warning: Invalid heatmap provided to find_and_extract_rois. Shape: {heatmap_np.shape if heatmap_np is not None else 'None'}")
        return []

    # Get the actual dimensions of the (potentially cropped) heatmap
    h_heat, w_heat = heatmap_np.shape[:2]
    if h_heat == 0 or w_heat == 0:
        print(f"Warning: Invalid heatmap dimensions ({h_heat}x{w_heat}).")
        return []

    # 2. Normalize and threshold the heatmap to create a binary mask
    # Handle potential non-finite values (though less likely after cropping valid data)
    if not np.all(np.isfinite(heatmap_np)):
        print("Warning: Non-finite values found in heatmap, replacing with 0.")
        heatmap_np = np.nan_to_num(heatmap_np, nan=0.0, posinf=0.0, neginf=0.0)

    min_h_val, max_h_val = np.min(heatmap_np), np.max(heatmap_np)
    # Avoid division by zero if the heatmap is flat (constant value)
    if min_h_val == max_h_val:
        print("Warning: Heatmap is constant, no contours will be found.")
        return []

    # Normalize heatmap to 0-255 range for thresholding and contour finding
    # NORM_MINMAX scales the range [min_h_val, max_h_val] to [0, 255]
    heatmap_norm_u8 = cv2.normalize(heatmap_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Binary thresholding
    # Ensure threshold_value is clamped between 0 and 1
    threshold_norm = min(1.0, max(0.0, threshold_value))
    # Calculate the pixel value corresponding to the relative threshold
    thresh_pixel_val = int(threshold_norm * 255) # Threshold applied on the 0-255 normalized map
    # Pixels >= thresh_pixel_val become 255 (white), others 0 (black)
    _, binary_mask = cv2.threshold(heatmap_norm_u8, thresh_pixel_val, 255, cv2.THRESH_BINARY)

    # 3. Find contours in the binary mask
    # RETR_EXTERNAL finds only the outer contours.
    # CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments, leaving only their end points.
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roi_data = [] # Changed variable name for clarity

    # 4. Analyze each found contour
    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter by minimum area
        if area >= min_area:
            # Create a mask specific to this contour to sample heatmap values
            contour_mask = np.zeros_like(binary_mask, dtype=np.uint8) # Same size as binary mask, initialized to 0
            # Draw the current contour filled with white (255) on the contour mask
            cv2.drawContours(contour_mask, [contour], contourIdx=-1, color=255, thickness=cv2.FILLED)

            # Find the coordinates (indices) within the heatmap where the contour mask is > 0
            # np.where returns a tuple of arrays (row_indices, col_indices)
            indices_in_mask_yx = np.where(contour_mask > 0)

            # Check if the mask actually covers any pixels (it should if area > 0, but good practice)
            if len(indices_in_mask_yx[0]) == 0:
                continue # Skip if no pixels are covered by this contour mask

            # Extract the original heatmap values corresponding to this contour region
            # Use the indices found above to index into the original heatmap_np
            values_in_mask = heatmap_np[indices_in_mask_yx]

            # Verify that we successfully extracted values
            if values_in_mask.size == 0:
                continue # Skip if no values were extracted

            # Find the index of the maximum score *within the extracted values*
            idx_of_max_in_filtered_array = np.argmax(values_in_mask)

            # Get the coordinates (y, x) of the maximum point IN THE HEATMAP's coordinate system
            # Use the index found above to get the corresponding y and x from indices_in_mask_yx
            max_y_heat = indices_in_mask_yx[0][idx_of_max_in_filtered_array]
            max_x_heat = indices_in_mask_yx[1][idx_of_max_in_filtered_array]

            # Get the maximum and mean score within the contour using the extracted values
            score_max_inside_contour = values_in_mask[idx_of_max_in_filtered_array] # Direct access to max value
            score_mean_inside_contour = np.mean(values_in_mask)

            # Calculate the bounding box (x, y, width, height) of the contour in the heatmap's coordinates
            x_heat, y_heat, w_box, h_box = cv2.boundingRect(contour)

            # Store ROI data with coordinates normalized relative to the heatmap dimensions (h_heat, w_heat)
            roi_data.append({
                "score_mean": float(score_mean_inside_contour), # Ensure float type
                "score_max": float(score_max_inside_contour),  # Ensure float type
                "x": x_heat / w_heat,           # Relative x-coordinate (top-left)
                "y": y_heat / h_heat,           # Relative y-coordinate (top-left)
                "w": w_box / w_heat,            # Relative width
                "h": h_box / h_heat,            # Relative height
                "x_max": max_x_heat / w_heat,   # Relative max x-coordinate
                "y_max": max_y_heat / h_heat    # Relative max y-coordinate
            })

    # Sort ROIs by maximum score in descending order
    return list(sorted(roi_data, key=lambda item: -item["score_max"]))
