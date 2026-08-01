import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

def leaf_present_bgr(img: np.ndarray) -> float:
    """
    Checks leaf presence based on HSV green color thresholding.
    Returns fraction of green pixels in the image.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([25, 40, 30])
    upper = np.array([100, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    green_ratio = float(np.count_nonzero(mask) / (img.shape[0] * img.shape[1]))
    return green_ratio

def preprocess_image(img: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    """
    Preprocesses OpenCV BGR image for EfficientNet CNN model.
    Converts to RGB, resizes, applies EfficientNet preprocessing, adds batch dimension.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)
    arr = img_resized.astype("float32")
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr
