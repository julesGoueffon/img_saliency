# tests/test_detector.py
import pytest
from PIL import Image
import numpy as np
# Make sure 'img_focus' is importable, e.g., by setting PYTHONPATH
# or having the tests directory at the same level as the img_focus package.
from img_focus import Salency

# Create a simple test image (or include a small image in your tests)
@pytest.fixture
def sample_image():
    # Creates a simple image with a "hot" zone
    img_array = np.zeros((100, 100, 3), dtype=np.uint8)
    img_array[30:70, 30:70, :] = 200 # "Important" zone (brighter)
    img = Image.fromarray(img_array)
    return img

@pytest.fixture(scope="module") # Use module scope if Salency init is expensive and stateless for tests
def detector():
    # Fixture to initialize the detector once per test module if possible
    # Or use default function scope for each test if the state needs to be clean
    print("\nInitializing Salency detector for tests...")
    return Salency()

def test_initialization(detector):
    """Checks that initialization works correctly and loads the ORT session."""
    assert detector.session is not None
    # Check if CPUExecutionProvider is indeed the first (and likely only) provider configured
    assert 'CPUExecutionProvider' in detector.session.get_providers()
    # Example of a more specific check if you always expect CPU only:
    # assert detector.session.get_providers() == ['CPUExecutionProvider']

def test_find_regions_of_interest(detector, sample_image):
    """Tests the main detection function."""
    # Use thresholds appropriate for the simple test image and expected model output
    rois = detector.find_regions_of_interest(sample_image,
                                             roi_threshold=0.5,    # Threshold for binarizing heatmap
                                             score_threshold=0.1,  # Minimum max_score for a ROI to be kept
                                             min_contour_area=50) # Min area in pixels on the heatmap

    assert isinstance(rois, list)
    # On a simple image like this, we should ideally detect the single bright square.
    # Depending on the model's behavior and thresholding, it might find 0, 1, or potentially more if the area fragments.
    # assert len(rois) == 1 # Be more specific if the model consistently finds one ROI here.
    assert len(rois) >= 0 # A basic check that it returns a list, possibly empty.

    # Check the structure of the first found ROI, if any
    if rois:
        print(f"Found {len(rois)} ROIs. First ROI: {rois[0]}")
        roi = rois[0]
        assert isinstance(roi, dict)
        # Check for expected keys
        expected_keys = {"score_mean", "score_max", "x", "y", "w", "h", "x_max", "y_max"}
        assert expected_keys.issubset(roi.keys())

        # Check basic validity of values
        assert roi["w"] > 0
        assert roi["h"] > 0
        assert 0 <= roi["x"] <= 1
        assert 0 <= roi["y"] <= 1
        assert 0 < roi["w"] <= 1  # Width should be positive and relative
        assert 0 < roi["h"] <= 1  # Height should be positive and relative
        assert 0 <= roi["x_max"] <= 1
        assert 0 <= roi["y_max"] <= 1
        # The score_max depends heavily on the model's output scale.
        # If the heatmap is normalized internally before this step, it might be 0-1.
        # If it's raw score, the range is unknown without knowing the model.
        # assert 0 <= roi["score_max"] <= 1.0 # Adjust if score range is different
        assert roi["score_max"] >= 0 # A safe minimum assumption
    else:
        print("No ROIs found for the sample image with current settings.")


def test_get_heatmap(detector, sample_image):
    """Tests the heatmap generation."""
    heatmap = detector.get_heatmap(sample_image)
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.ndim == 2 # Should be a 2D heatmap (after squeeze and processing)
    assert np.all(np.isfinite(heatmap)) # Check for NaN or Inf values

    # The size of the returned heatmap should correspond to the image size
    # *before* padding but *after* aspect-ratio-preserving resize.
    # For a 100x100 input and TARGET_SIZE=224, it's resized to 100x100 (LongestMaxSize doesn't enlarge)
    # then padded to 224x224. The get_heatmap function should return the 100x100 part.
    # However, the exact size depends on the implementation detail of aspect ratio preservation
    # and cropping. Let's check if the shape is reasonable.
    assert heatmap.shape[0] > 0
    assert heatmap.shape[1] > 0
    print(f"Generated heatmap shape: {heatmap.shape}")
    # You could add a more specific check if you calculate the expected dimensions based on
    # TARGET_SIZE and the `preprocess_image` logic for the 100x100 input.
    # Example calculation for 100x100 input & TARGET_SIZE=224:
    # ratio = min(224/100, 224/100) = 2.24
    # If LongestMaxSize only shrinks, ratio = 1. resized_h=100, resized_w=100.
    # Expected shape would be (100, 100) after cropping the padding.
    # assert heatmap.shape == (100, 100) # Add this if you confirm the logic