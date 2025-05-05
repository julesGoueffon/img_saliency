from typing import Union, List, Dict

import albumentations as A
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


ONNX_MODEL_PATH = "./models/vit_s_focus.onnx"  # Chemin vers votre fichier .onnx


TARGET_SIZE = 224 # Taille d'entrée attendue
IMG_MEAN = (0.485, 0.456, 0.406) # Moyenne ImageNet
IMG_STD = (0.229, 0.224, 0.225) # Écart-type ImageNet

 # Ignorer les zones plus petites que ça (en pixels carrés)

NUM_CLASSES = 1 # Nombre de classes de sortie (1 pour la carte d'importance)

INPUT_NAME = "input_image"
OUTPUT_NAME = "output_importance"

class Salency:
    def __init__(self):
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
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

def find_and_rois(heatmap_np, threshold_value, min_area):
    """
    Trouve les zones d'intérêt sur la heatmap, les filtre et dessine
    le contour, le rectangle et le cercle englobants sur l'image fournie.
    """
    # 1. Normaliser la heatmap en 0-255 et convertir en uint8
    if not np.all(np.isfinite(heatmap_np)):
        heatmap_np = np.nan_to_num(heatmap_np, nan=0.0, posinf=0.0, neginf=0.0)
    min_h, max_h = np.min(heatmap_np), np.max(heatmap_np)
    if min_h == max_h:
        return None


    heatmap_norm_u8 = cv2.normalize(heatmap_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    thresh_val, binary_mask = cv2.threshold(heatmap_norm_u8, int(255*min(1,max(0,threshold_value))), 255, cv2.THRESH_BINARY)


    # 3. Trouver les contours dans le masque binaire
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Nombre de contours trouvés avant filtrage: {len(contours)}")



    #4.find max
    max_idx_flat = np.argmax(heatmap_np)
    max_point_yx = np.unravel_index(max_idx_flat, heatmap_np.shape) # (row, col) ou (y, x)
    xmax, ymax = (max_point_yx[1], max_point_yx[0])

    contours_data= []

    # 4. Filtrer les contours et dessiner les 3 formes
    for contour in contours:
        area = cv2.contourArea(contour)
        # Filtrer par aire minimale
        if area >= min_area:
            point_to_test = (float(xmax), float(ymax))
            contour_mask = np.zeros_like(binary_mask, dtype=np.uint8)

            # 2. Dessiner le contour actuel sur le masque vide, en le remplissant (-1 ou cv2.FILLED)
            #    Note: cv2.drawContours attend une liste de contours, d'où [contour]
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
            mask_values_norm = heatmap_norm_u8[contour_mask > 0]
            mask_values = heatmap_np[contour_mask > 0]

            if len(mask_values) == 0:
                continue
            scores_inside_contour = np.mean(mask_values)
            scores_max_inside_contour = np.max(mask_values)

            # b) Calculer et dessiner le rectangle minimum englobant (Vert)
            x, y, w, h = cv2.boundingRect(contour)
            contours_data.append({"score_mean": scores_inside_contour,
                                  "score_max": scores_max_inside_contour,
                                  "x": x,
                                  "y": y,
                                  "w": w,
                                  "h": h})

    # Convertir l'image finale en RGB pour l'affichage avec Matplotlib
    return list(sorted(contours_data, key = lambda x: -x["score_max"]))


# --- Script Principal ---
if __name__ == "__main__":
    print("--- Détection ROI sur Heatmap ViT (Contour, Rectangle, Cercle) ---")
    s = Salency()
    image = Image.open("/Users/julesgoueffon/PycharmProjects/img_focus_point/img.png").convert("RGB")

    rois = s.find_regions_of_interest(image)
    print(rois)