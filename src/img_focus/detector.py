from typing import Union, List, Dict

import albumentations as A
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from importlib_resources import files


TARGET_SIZE = 224 # Taille d'entrée attendue
IMG_MEAN = (0.485, 0.456, 0.406) # Moyenne ImageNet
IMG_STD = (0.229, 0.224, 0.225) # Écart-type ImageNet

 # Ignorer les zones plus petites que ça (en pixels carrés)

NUM_CLASSES = 1 # Nombre de classes de sortie (1 pour la carte d'importance)

INPUT_NAME = "input_image"
OUTPUT_NAME = "output_importance"

class Salency:
    _MODEL_SUBPATH = "models/vit_s_focus.onnx"

    def __init__(self):

        providers = ['CPUExecutionProvider']

        model_path = files("img_focus").joinpath(self._MODEL_SUBPATH)

        # Convertir l'objet pathlib.Path en string, car ONNX Runtime attend une chaîne.

        # --- Optionnel : Garder le print de débogage si utile ---
        # ONNXRuntime a besoin du chemin en tant que string
        self.session = ort.InferenceSession(str(model_path), providers=providers)

        print(f"Session ONNX chargée. Fournisseur utilisé : {self.session.get_providers()}")


    def find_regions_of_interest(self, img:Image,
                                 roi_threshold:float = 0.8,
                                 score_threshold:Union[float,None]= 0.9,
                                 min_contour_area:int = 100) -> List[Dict]:
        heatmap_final = self.get_heatmap(img)
        image_with_rois = find_and_rois(
            heatmap_final,
            roi_threshold,
            min_contour_area
        )
        if score_threshold:
            image_with_rois  = list(filter(lambda x:x["score_max"]>=score_threshold, image_with_rois))
        return image_with_rois

    def get_heatmap(self, img:Image):
        image_array, original_size_wh, original_image_np, pre_padding_size_hw = preprocess_image(
            img, TARGET_SIZE, IMG_MEAN, IMG_STD
        )
        results = self.session.run([OUTPUT_NAME], {INPUT_NAME: image_array})
        output_data = results[0]
        print("output_data max", np.max(output_data))

        heatmap_final = get_prediction_heatmap(
            output_data,
            pre_padding_size_hw
        )
        return heatmap_final


# --- Fonctions (simplifiées du script précédent) ---

def preprocess_image(image, target_size, img_mean, img_std):
    original_size_wh = image.size
    image_np = np.array(image)
    original_h, original_w = image_np.shape[:2]

    transforms = A.Compose([
        A.LongestMaxSize(max_size=target_size, interpolation=cv2.INTER_AREA),
        A.PadIfNeeded(min_height=target_size, min_width=target_size,
                      border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=img_mean, std=img_std),
    ])

    processed = transforms(image=image_np)
    image_np_trans = processed['image']
    image_np_trans = np.expand_dims(image_np_trans, axis=0)
    image_np_trans = image_np_trans.transpose(0, 3, 1, 2)

    ratio = min(target_size / original_h, target_size / original_w)
    resized_h, resized_w = int(original_h * ratio), int(original_w * ratio)
    pre_padding_size_hw = (resized_h, resized_w)

    return image_np_trans, original_size_wh, image_np, pre_padding_size_hw


def get_prediction_heatmap(scores, pre_padding_size_hw):
    """Retourne la heatmap finale, corrigée du padding et redimensionnée à la taille originale."""
    print("score max", np.max(scores))

    if scores.shape[0] != 1 or scores.shape[1] != NUM_CLASSES:
         raise ValueError(f"Shape de sortie inattendue: {scores.shape}. Attendu (1, {NUM_CLASSES}, H, W)")
    heatmap_pred_np = scores.squeeze()
    if heatmap_pred_np.ndim != 2:
        raise ValueError(f"Heatmap non 2D après squeeze: {heatmap_pred_np.shape}")

    # --- Correction Padding et Redimensionnement ---
    heatmap_target_size = cv2.resize(heatmap_pred_np.astype(np.float32),
                                     (TARGET_SIZE, TARGET_SIZE),
                                     interpolation=cv2.INTER_LINEAR)
    resized_h, resized_w = pre_padding_size_hw
    pad_h = TARGET_SIZE - resized_h
    pad_w = TARGET_SIZE - resized_w
    pad_top = pad_h // 2
    pad_left = pad_w // 2

    heatmap_cropped = heatmap_target_size[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w]
    if heatmap_cropped.size == 0:
        raise ValueError("Découpage de la heatmap a produit une image vide.")

    return heatmap_cropped

# --- Nouvelle Fonction: Détection et Dessin des ROIs (modifiée) ---

# --- Nouvelle Fonction: Détection et Calcul des ROIs ---
# (Remplacez l'ancienne fonction find_and_rois par celle-ci dans votre fichier img_focus.py ou équivalent)

def find_and_rois(heatmap_np, threshold_value, min_area):
    """
    Trouve les zones d'intérêt sur la heatmap fournie, filtre par aire minimale,
    et retourne une liste de dictionnaires contenant les données des ROI,
    y compris les coordonnées (bounding box et point max) relatives
    aux dimensions de la heatmap d'entrée (heatmap_np).
    """
    # 1. Vérifier si la heatmap est valide
    if heatmap_np is None or heatmap_np.size == 0:
        print("Warning: Heatmap vide fournie à find_and_rois.")
        return []

    # Obtenir les dimensions réelles de la heatmap traitée
    h_heat, w_heat = heatmap_np.shape[:2]
    if h_heat == 0 or w_heat == 0:
        print(f"Warning: Dimensions de la heatmap invalides ({h_heat}x{w_heat}).")
        return []

    # 2. Normaliser et seuiller la heatmap pour créer un masque binaire
    if not np.all(np.isfinite(heatmap_np)):
        heatmap_np = np.nan_to_num(heatmap_np, nan=0.0, posinf=0.0, neginf=0.0) # Gérer NaN/inf

    min_h_val, max_h_val = np.min(heatmap_np), np.max(heatmap_np)
    if min_h_val == max_h_val: # Heatmap constante, pas de contours à trouver
        print("Warning: Heatmap constante, aucun contour ne sera trouvé.")
        return []

    # Normalisation en 0-255 pour le seuillage
    heatmap_norm_u8 = cv2.normalize(heatmap_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Seuillage binaire
    # S'assurer que threshold_value est entre 0 et 1
    threshold_norm = min(1.0, max(0.0, threshold_value))
    thresh_pixel_val = int(255 * threshold_norm)
    _, binary_mask = cv2.threshold(heatmap_norm_u8, thresh_pixel_val, 255, cv2.THRESH_BINARY)

    # 3. Trouver les contours dans le masque binaire
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Nombre de contours trouvés avant filtrage: {len(contours)}")

    contours_data = []

    # 4. Analyser chaque contour trouvé
    for contour in contours:
        area = cv2.contourArea(contour)

        # Filtrer par aire minimale
        if area >= min_area:
            # Créer un masque spécifique pour ce contour
            contour_mask = np.zeros_like(binary_mask, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

            # Trouver les indices (y, x) des pixels appartenant au contour dans la heatmap
            indices_in_mask_yx = np.where(contour_mask > 0)

            # Vérifier si le masque contient des pixels (peut être vide pour des contours très petits/fins)
            if len(indices_in_mask_yx[0]) == 0:
                continue

            # Extraire les valeurs de la heatmap originale correspondantes à ce contour
            values_in_mask = heatmap_np[indices_in_mask_yx]

            # Vérifier si on a bien extrait des valeurs
            if values_in_mask.size == 0:
                continue

            # Trouver l'index du score maximum parmi ces valeurs
            idx_of_max_in_filtered_array = np.argmax(values_in_mask)

            # Obtenir les coordonnées (y, x) du point maximum DANS LA HEATMAP
            max_y = indices_in_mask_yx[0][idx_of_max_in_filtered_array]
            max_x = indices_in_mask_yx[1][idx_of_max_in_filtered_array]

            # Obtenir le score maximum et le score moyen dans le contour
            score_max_inside_contour = values_in_mask[idx_of_max_in_filtered_array] # Plus direct
            score_mean_inside_contour = np.mean(values_in_mask)

            # Calculer le rectangle englobant (bounding box) du contour
            x, y, w, h = cv2.boundingRect(contour)

            # Stocker les données du ROI avec coordonnées relatives aux dimensions de la heatmap (h_heat, w_heat)
            contours_data.append({
                "score_mean": score_mean_inside_contour,
                "score_max": score_max_inside_contour,
                "x": x / w_heat,       # Coordonnée x relative
                "y": y / h_heat,       # Coordonnée y relative
                "w": w / w_heat,       # Largeur relative
                "h": h / h_heat,       # Hauteur relative
                "x_max": max_x / w_heat, # Coordonnée x max relative
                "y_max": max_y / h_heat  # Coordonnée y max relative
            })

    # Trier les ROIs par score maximum décroissant
    return list(sorted(contours_data, key=lambda x: -x["score_max"]))

# --- Script Principal ---
if __name__ == "__main__":
    print("--- Test local de détection ROI ---")
    # Assurez-vous que le chemin vers l'image est correct pour CE script de tests
    # Il est préférable d'utiliser un chemin relatif ou de le passer en argument
    image = Image.open("/Users/julesgoueffon/PycharmProjects/img_focus_point/img.png").convert("RGB")

    s = Salency() # Initialise et charge le modèle
    rois = s.find_regions_of_interest(image, score_threshold=0.85) # Exemple avec seuil
    if rois:
        print(f"ROIs trouvées ({len(rois)}):")
        for i, roi in enumerate(rois):
            print(f"  ROI {i+1}: Score Max={roi['score_max']:.3f}, Box=[{roi['x']},{roi['y']},{roi['w']},{roi['h']}]")
    else:
        print("Aucune ROI trouvée avec les seuils donnés.")
