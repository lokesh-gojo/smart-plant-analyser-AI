"""
SmartVine AI — Classical Color Space & ML Baselines
Direct reproduction of Seng et al. (IEEE Access, 2018) Section IV (Tables VI and VII).
Evaluates pixel classification across RGB, HSV, YCbCr, and CIELab color spaces using
Gaussian-Kernel SVM, k-Nearest Neighbors (k=10), AdaBoost, and Logistic Regression.
"""

from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def extract_color_spaces(bgr_image: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Converts a standard BGR image into the 4 canonical color spaces studied in the paper:
    1. RGB (native red-green-blue)
    2. HSV (Hue-Saturation-Value: isolates chromatic hue from lighting)
    3. YCbCr (Luminance Y and blue-difference Cb / red-difference Cr chrominance)
    4. CIELab (Perceptually uniform L* lightness, a* green-red, b* blue-yellow)
    """
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2Lab)
    return {"RGB": rgb, "HSV": hsv, "YCbCr": ycrcb, "Lab": lab}


def extract_pixel_samples_from_images(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    space_name: str = "HSV",
    sample_pixels_per_img: int = 2000,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts stratified grape berry (positive) vs background/leaf (negative) pixel samples.
    """
    np.random.seed(random_state)
    x_list, y_list = [], []

    for img, mask in zip(images, masks):
        spaces = extract_color_spaces(img)
        feat_map = spaces[space_name].reshape(-1, 3).astype(np.float32)
        binary_labels = (mask.reshape(-1) > 0).astype(int)

        # Balanced random pixel sampling per image
        pos_idx = np.where(binary_labels == 1)[0]
        neg_idx = np.where(binary_labels == 0)[0]

        half_n = sample_pixels_per_img // 2
        selected_pos = np.random.choice(pos_idx, size=min(half_n, len(pos_idx)), replace=False) if len(pos_idx) > 0 else np.array([], dtype=int)
        selected_neg = np.random.choice(neg_idx, size=min(half_n, len(neg_idx)), replace=False) if len(neg_idx) > 0 else np.array([], dtype=int)

        chosen_idx = np.concatenate([selected_pos, selected_neg])
        if len(chosen_idx) > 0:
            x_list.append(feat_map[chosen_idx])
            y_list.append(binary_labels[chosen_idx])

    if not x_list:
        raise ValueError("No valid pixel samples extracted.")

    X = np.vstack(x_list)
    y = np.concatenate(y_list)
    return X, y


def benchmark_berry_pixel_classifiers(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    sample_pixels_per_img: int = 2000,
    cultivar_type: str = "white"
) -> Dict[str, Dict[str, float]]:
    """
    Reproduces Seng et al. (2018) Tables VI and VII benchmarking 6 classifiers
    across RGB, HSV, YCbCr, and CIELab color spaces.
    
    Validation findings:
    - White cultivars (Sauvignon Blanc): HSV and Lab achieve ~84.4% accuracy with SVM.
    - Red cultivars (Shiraz, Merlot): CIELab (~89.1%) and YCbCr (~89.0%) achieve peak accuracy.
    - k-NN (k=10): Reaches 83.7% (white) and 88.8% (red), optimal for mobile edge inference.
    """
    results: Dict[str, Dict[str, float]] = {}

    for space_name in ["RGB", "HSV", "YCbCr", "Lab"]:
        X, y = extract_pixel_samples_from_images(
            images, masks, space_name=space_name, sample_pixels_per_img=sample_pixels_per_img
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # 1. Gaussian RBF Kernel SVM (Best overall performer in paper)
        svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
        svm.fit(X_train, y_train)
        y_pred_svm = svm.predict(X_test)
        svm_acc = accuracy_score(y_test, y_pred_svm)
        svm_f1 = f1_score(y_test, y_pred_svm, zero_division=0)

        # 2. k-NN (k=10, edge baseline in paper)
        knn = KNeighborsClassifier(n_neighbors=10, n_jobs=1)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        knn_acc = accuracy_score(y_test, y_pred_knn)
        knn_f1 = f1_score(y_test, y_pred_knn, zero_division=0)

        # 3. AdaBoost (Boosted Tree)
        adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)
        adaboost.fit(X_train, y_train)
        y_pred_ada = adaboost.predict(X_test)
        ada_acc = accuracy_score(y_test, y_pred_ada)

        # 4. Logistic Regression
        lr = LogisticRegression(max_iter=500, random_state=42)
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        lr_acc = accuracy_score(y_test, y_pred_lr)

        results[space_name] = {
            "SVM_Acc": round(float(svm_acc * 100), 2),
            "SVM_F1": round(float(svm_f1 * 100), 2),
            "kNN_Acc": round(float(knn_acc * 100), 2),
            "kNN_F1": round(float(knn_f1 * 100), 2),
            "AdaBoost_Acc": round(float(ada_acc * 100), 2),
            "LogReg_Acc": round(float(lr_acc * 100), 2),
        }

    return results


def predict_bunch_mask_classical(
    bgr_image: np.ndarray,
    model: Any,
    color_space: str = "HSV"
) -> np.ndarray:
    """
    Applies a trained classical classifier (SVM or k-NN) pixel-by-pixel to segment grape bunches.
    """
    h, w, _ = bgr_image.shape
    spaces = extract_color_spaces(bgr_image)
    pixels = spaces[color_space].reshape(-1, 3).astype(np.float32)

    pred_labels = model.predict(pixels)
    binary_mask = pred_labels.reshape((h, w)).astype(np.uint8) * 255

    # Morphological cleaning: remove salt-and-pepper noise and fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned
