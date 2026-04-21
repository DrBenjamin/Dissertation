"""Custom TensorFlow Lite Image Classification utilities.

This module provides functionality to load and use custom TensorFlow Lite
image classification models, adapted from the logic in image_classification.ipynb.
"""

import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)

TFLITE_INTERPRETER_CLS: Optional[object] = None
TFLITE_RUNTIME: Optional[str] = None

try:
    import tensorflow as tf  # type: ignore
    from tensorflow.lite import Interpreter as _TfLiteInterpreter  # type: ignore
    TFLITE_INTERPRETER_CLS = _TfLiteInterpreter
    TFLITE_RUNTIME = "tensorflow"
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter as _RuntimeInterpreter  # type: ignore
        tf = None  # type: ignore
        TFLITE_INTERPRETER_CLS = _RuntimeInterpreter
        TFLITE_RUNTIME = "tflite_runtime"
    except ImportError:
        TFLITE_RUNTIME = None

if TFLITE_RUNTIME is None:
    logger.warning(
        "TensorFlow Lite runtime not available. Custom TensorFlow Lite image classification models will not work."
    )
elif TFLITE_RUNTIME == "tflite_runtime":
    logger.info("Using TensorFlow Lite runtime via tflite_runtime interpreter.")

try:
    import mediapipe as mp
    MP_AVAILABLE = True
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError:
    MP_AVAILABLE = False
    logger.warning("MediaPipe not available.")


@dataclass
class CustomTFLiteImageClassifierConfig:
    """Configuration for custom TensorFlow Lite image classification model."""
    model_path: str
    confidence_threshold: float = 0.0
    max_results: int = 4


class CustomTFLiteImageClassifier:
    """Custom TensorFlow Lite image classifier.
    
    This class provides an interface to use custom TensorFlow Lite image
    classification models, adapted from the MediaPipe image classification notebook.
    """
    
    def __init__(self, config: CustomTFLiteImageClassifierConfig):
        """Initializing custom TensorFlow Lite image classifier.
        
        Args:
            config: Configuration object for the classifier
        """
        self.config = config
        self.classifier = None
        
        self._load_model()
    
    def _load_model(self):
        """Loading the custom TensorFlow Lite image classification model."""
        if not MP_AVAILABLE:
            logger.error("MediaPipe Tasks runtime not available")
            return
        
        if not os.path.exists(self.config.model_path):
            logger.error(f"Model file not found: {self.config.model_path}")
            return
        
        try:
            base_options = mp_python.BaseOptions(model_asset_path=self.config.model_path)
            options_kwargs = {
                "base_options": base_options,
                "max_results": self.config.max_results,
            }

            # Avoid forcing an aggressive threshold at task level, otherwise
            # low-confidence but still valid top classes may be dropped.
            if self.config.confidence_threshold > 0:
                options_kwargs["score_threshold"] = self.config.confidence_threshold

            options = vision.ImageClassifierOptions(**options_kwargs)
            self.classifier = vision.ImageClassifier.create_from_options(options)
            logger.info("Custom image classification model ready: %s", self.config.model_path)
        except Exception as e:
            logger.error(f"Failed to load custom image classification model: {e}")
            self.classifier = None
    
    def classify(self, image: np.ndarray) -> Optional[Dict]:
        """Classifying image using the custom model.
        
        Args:
            image: Input image as BGR numpy array
            
        Returns:
            Dictionary with classification results or None if classification failed
        """
        if self.classifier is None:
            logger.error("Model not loaded properly")
            return None
        
        try:
            # Preprocessing image to MediaPipe format
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            classification_result = self.classifier.classify(mp_image)

            if not classification_result.classifications:
                return {
                    'classifications': [],
                    'model_path': self.config.model_path
                }

            categories = classification_result.classifications[0].categories
            results: List[Dict[str, float]] = []

            for category in categories[: self.config.max_results]:
                score = float(category.score)
                if score < self.config.confidence_threshold:
                    continue
                name = category.category_name or f"Class_{category.index}"
                results.append({
                    'category_name': name,
                    'score': score,
                    'index': int(category.index),
                })

            if not results and categories:
                # Returning top-k categories even when scores are below threshold
                # to match notebook behaviour and prevent empty UI output.
                for category in categories[: self.config.max_results]:
                    name = category.category_name or f"Class_{category.index}"
                    results.append({
                        'category_name': name,
                        'score': float(category.score),
                        'index': int(category.index),
                    })

            if results and all(item['category_name'].startswith('Class_') for item in results):
                posture_labels = {
                    0: "normal posture",
                    1: "slumped posture",
                    2: "tensed posture",
                }
                for item in results:
                    index = item['index']
                    item['category_name'] = posture_labels.get(index, item['category_name'])
            
            return {
                'classifications': results,
                'model_path': self.config.model_path
            }
            
        except Exception as e:
            logger.error(f"Error classifying image with custom model: {e}")
            return None


def create_custom_image_classifier(model_path: str, **kwargs) -> Optional[CustomTFLiteImageClassifier]:
    """Creating a custom TensorFlow Lite image classifier.
    
    Args:
        model_path: Path to the custom TensorFlow Lite model file
        **kwargs: Additional configuration parameters
        
    Returns:
        CustomTFLiteImageClassifier instance or None if creation failed
    """
    if not is_custom_image_classification_model_available(model_path):
        return None
    
    config = CustomTFLiteImageClassifierConfig(
        model_path=model_path,
        confidence_threshold=kwargs.get('confidence_threshold', 0.0),
        max_results=kwargs.get('max_results', 4)
    )
    
    return CustomTFLiteImageClassifier(config)


def is_custom_image_classification_model_available(model_path: str) -> bool:
    """Checking if a custom image classification model file is available and valid.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        True if model is available and can be loaded, False otherwise
    """
    if not model_path or not os.path.exists(model_path):
        return False
    
    if not model_path.endswith('.tflite'):
        return False
    
    if not MP_AVAILABLE:
        logger.warning("MediaPipe Tasks runtime unavailable; classification disabled.")
        return False

    try:
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.ImageClassifierOptions(base_options=base_options)
        classifier = vision.ImageClassifier.create_from_options(options)
        classifier.close()
        return True
    except Exception as e:
        logger.warning(f"Custom image classification model validation failed: {e}")
        return False


def detect_model_type(model_path: str) -> str:
    """Detecting whether a TensorFlow Lite model is for pose detection or image classification.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        'pose' for pose detection models, 'classification' for image classification models,
        'unknown' if unable to determine
    """
    if not os.path.exists(model_path) or TFLITE_INTERPRETER_CLS is None:
        return 'unknown'
    
    try:
        interpreter = TFLITE_INTERPRETER_CLS(model_path=model_path)
        interpreter.allocate_tensors()
        
        output_details = interpreter.get_output_details()
        
        if len(output_details) == 0:
            return 'unknown'
        
        output_shape = output_details[0]['shape']
        
        # Heuristic detection based on output shape
        # Pose models typically output landmark coordinates (e.g., [1, 33, 3] for 33 landmarks)
        # Classification models typically output class probabilities (e.g., [1, num_classes])
        
        if len(output_shape) >= 3 and output_shape[-1] in [2, 3, 4]:
            # Likely pose model with landmark coordinates
            return 'pose'
        elif len(output_shape) == 2 and output_shape[-1] >= 2:
            # Likely classification model with multiple classes
            return 'classification'
        else:
            return 'unknown'
            
    except Exception as e:
        logger.warning(f"Error detecting model type: {e}")
        return 'unknown'