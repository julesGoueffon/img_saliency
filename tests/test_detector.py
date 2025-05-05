# tests/test_detector.py
import pytest
from PIL import Image
import numpy as np
from img_focus import Salency

# Créez une image de test simple (ou incluez une petite image dans vos tests)
@pytest.fixture
def sample_image():
    # Crée une image simple avec une zone "chaude"
    img_array = np.zeros((100, 100, 3), dtype=np.uint8)
    img_array[30:70, 30:70, :] = 200 # Zone "importante"
    img = Image.fromarray(img_array)
    return img

@pytest.fixture
def detector():
    # Fixture pour initialiser le détecteur une seule fois par session de test si possible
    # Ou pour chaque test si l'état doit être propre
    return Salency()

def test_initialization(detector):
    """Vérifie que l'initialisation se passe bien et charge la session ORT."""
    assert detector.session is not None
    assert detector.session.get_providers()[0] == 'CPUExecutionProvider' # Ou ce que vous attendez

def test_find_regions_of_interest(detector, sample_image):
    """Teste la fonction principale de détection."""
    rois = detector.find_regions_of_interest(sample_image, roi_threshold=0.5, score_threshold=0.1, min_contour_area=50)

    assert isinstance(rois, list)
    # Sur une image simple, on pourrait s'attendre à 1 ROI principale
    assert len(rois) >= 1 # Soyez flexible ou plus précis si vous connaissez l'output exact

    # Vérifiez la structure du premier ROI trouvé
    if rois:
        roi = rois[0]
        assert isinstance(roi, dict)
        assert "score_mean" in roi
        assert "score_max" in roi
        assert "x" in roi
        assert "y" in roi
        assert "w" in roi
        assert "h" in roi
        assert roi["w"] > 0
        assert roi["h"] > 0
        assert 0 <= roi["score_max"] <= 1.0 # Le score max devrait être dans cette plage après normalisation interne potentielle ou être le score brut

def test_get_heatmap(detector, sample_image):
    """Teste la génération de la heatmap."""
    heatmap = detector.get_heatmap(sample_image)
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.ndim == 2 # Doit être une heatmap 2D
    # La taille de la heatmap dépend du redimensionnement interne avant le retour
    # assert heatmap.shape == (expected_h, expected_w) # Ajoutez si vous connaissez la taille attendue