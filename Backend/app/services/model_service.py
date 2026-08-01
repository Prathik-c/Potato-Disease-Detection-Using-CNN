import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from app.core.config import settings
from app.core.logging import logger

class ModelService:
    def __init__(self):
        self.model = None
        self.class_names = settings.CLASS_NAMES

    def load(self, model_path: str = None):
        path = model_path or settings.MODEL_PATH
        if not os.path.exists(path):
            logger.error(f"Model file not found at path: {path}")
            raise FileNotFoundError(f"Model file not found at {path}")
        
        logger.info(f"Loading Keras model from {path}...")
        self.model = load_model(path)
        logger.info("Model loaded successfully.")

    def predict(self, input_tensor: np.ndarray):
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load() first.")
        
        predictions = self.model.predict(input_tensor)[0]
        class_id = int(np.argmax(predictions))
        label = self.class_names[class_id]
        confidence = float(predictions[class_id])
        return label, confidence

model_service = ModelService()
