import cv2
import numpy as np

# -------------------------------------------------
# Utility functions
# -------------------------------------------------

def crop_bottom_roi(img, bottom_ratio=0.4):
    """
    Crop bottom portion of image where biomass is present
    """
    h, w = img.shape[:2]
    start_y = int(h * (1 - bottom_ratio))
    return img[start_y:h, :]


def mask_side_walls(img, side_ratio=0.12):
    """
    Remove left and right curved glass wall regions
    """
    h, w = img.shape[:2]
    left = int(w * side_ratio)
    right = int(w * (1 - side_ratio))
    return img[:, left:right]


def preprocess_image(img):
    """
    Full preprocessing pipeline
    """
    # Step 1: Crop bottom ROI
    roi = crop_bottom_roi(img, bottom_ratio=0.4)

    # Step 2: Mask side walls
    roi = mask_side_walls(roi, side_ratio=0.12)

    # Step 3: Convert to LAB
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Step 4: Mild smoothing (preserve pellets)
    L_blur = cv2.GaussianBlur(L, (3, 3), 0)

    return roi, L_blur, B


# -------------------------------------------------
# Feature extractors
# -------------------------------------------------

def compute_blob_features(L_img):
    """
    Pellet-based features (mainly Day-3)
    """
    # Invert because pellets are bright
    _, thresh = cv2.threshold(L_img, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 20]

    if len(areas) == 0:
        return {
            "blob_count": 0,
            "mean_blob_area": 0,
            "blob_area_fraction": 0,
            "blob_size_std": 0
        }

    total_area = L_img.shape[0] * L_img.shape[1]

    return {
        "blob_count": len(areas),
        "mean_blob_area": np.mean(areas),
        "blob_area_fraction": np.sum(areas) / total_area,
        "blob_size_std": np.std(areas)
    }


def compute_sediment_features(L_img):
    """
    Sedimentation geometry features (mainly Day-6)
    """
    # Dark regions = sediment
    _, thresh = cv2.threshold(L_img, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = thresh.shape

    # Sediment height estimation
    row_sum = np.sum(thresh > 0, axis=1)
    sediment_rows = np.where(row_sum > 0.3 * w)[0]

    if len(sediment_rows) == 0:
        sediment_height = 0
    else:
        sediment_height = (sediment_rows[-1] - sediment_rows[0]) / h

    sediment_area_fraction = np.sum(thresh > 0) / (h * w)

    return {
        "sediment_height_ratio": sediment_height,
        "sediment_area_fraction": sediment_area_fraction
    }


def compute_universal_features(L_img):
    """
    Features useful for both Day-3 and Day-6
    """
    h, w = L_img.shape

    mean_L = np.mean(L_img)
    std_L = np.std(L_img)

    # Dark pixel fraction
    dark_pixels = np.sum(L_img < mean_L)
    dark_fraction = dark_pixels / (h * w)

    # Vertical centroid of intensity
    y_indices = np.arange(h).reshape(-1, 1)
    centroid_y = np.sum(y_indices * L_img) / np.sum(L_img)
    centroid_y_norm = centroid_y / h

    # Texture / sharpness
    laplacian_var = cv2.Laplacian(L_img, cv2.CV_64F).var()

    return {
        "mean_L_bottom": mean_L,
        "std_L_bottom": std_L,
        "dark_pixel_fraction": dark_fraction,
        "vertical_centroid": centroid_y_norm,
        "laplacian_variance": laplacian_var
    }


# -------------------------------------------------
# MASTER FUNCTION
# -------------------------------------------------

def extract_features(image_path):
    """
    Final feature extraction entry point
    """
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    roi, L_img, B_img = preprocess_image(img)

    features = {}

    # Universal
    features.update(compute_universal_features(L_img))

    # Pellet morphology
    features.update(compute_blob_features(L_img))

    # Sedimentation geometry
    features.update(compute_sediment_features(L_img))

    return features


# -------------------------------------------------
# TEST (optional)
# -------------------------------------------------

if __name__ == "__main__":
    test_img = "images/1.2/day3/image_1.jpeg"
    feats = extract_features(test_img)
    for k, v in feats.items():
        print(f"{k}: {v}")
