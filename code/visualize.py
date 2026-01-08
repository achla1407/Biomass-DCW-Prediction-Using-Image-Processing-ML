import cv2
import numpy as np
import matplotlib.pyplot as plt

# Preprocessing helpers (same as extract_features)

def crop_bottom_roi(img, bottom_ratio=0.4):
    h, w = img.shape[:2]
    start_y = int(h * (1 - bottom_ratio))
    return img[start_y:h, :]


def mask_side_walls(img, side_ratio=0.12):
    h, w = img.shape[:2]
    left = int(w * side_ratio)
    right = int(w * (1 - side_ratio))
    return img[:, left:right]


def preprocess(img):
    roi = crop_bottom_roi(img, 0.4)
    roi = mask_side_walls(roi, 0.12)

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    L_blur = cv2.GaussianBlur(L, (3, 3), 0)
    return roi, L_blur


# Segmentation visualizer


def visualize_segmentation(image_path):
    img = cv2.imread(image_path)
    roi, L = preprocess(img)

  
    # Pellet (blob) mask
  
    _, pellet_mask = cv2.threshold(
        L, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)
    pellet_mask = cv2.morphologyEx(
        pellet_mask, cv2.MORPH_OPEN, kernel
    )

    
    # Sediment mask
    
    _, sediment_mask = cv2.threshold(
        L, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Plotting everything

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    plt.title("ROI (Bottom + Side masked)")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(pellet_mask, cmap="gray")
    plt.title("Pellet / Blob Segmentation (Day-3)")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(sediment_mask, cmap="gray")
    plt.title("Sediment Segmentation (Day-6)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# TEST

if __name__ == "__main__":
    visualize_segmentation("images/1.2/day3/image_1.jpeg")

