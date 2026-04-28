"""Custom TensorFlow Lite Pose Detection utilities.

This module provides functionality to load and use custom TensorFlow Lite
pose landmarker models trained with MediaPipe Model Maker, alongside the
existing MediaPipe pose processing capabilities.
"""

import os
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Union
import numpy as np
import cv2

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
        "TensorFlow Lite runtime not available. Custom TensorFlow Lite models will not work."
    )
elif TFLITE_RUNTIME == "tflite_runtime":
    logger.info("Using TensorFlow Lite runtime via tflite_runtime interpreter.")

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    logger.warning("MediaPipe not available.")


@dataclass
class CustomTFLiteConfig:
    """Configuration for custom TensorFlow Lite pose model."""
    model_path: str
    input_size: Tuple[int, int] = (256, 256)  # Default input size for pose models
    confidence_threshold: float = 0.5
    num_landmarks: int = 33  # Standard pose landmarks


class CustomTFLitePoseDetector:
    """Custom TensorFlow Lite pose detector.
    
    This class provides an interface to use custom TensorFlow Lite pose
    landmarker models trained with MediaPipe Model Maker.
    """
    
    def __init__(self, config: CustomTFLiteConfig):
        """Initialize the custom TensorFlow Lite pose detector.
        
        Args:
            config: Configuration for the custom model
        """
        self.config = config
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
        if TFLITE_INTERPRETER_CLS is None:
            raise ImportError(
                "TensorFlow Lite runtime is required for custom TensorFlow Lite models"
            )
        
        self._load_model()
    
    def _load_model(self):
        """Loading the custom TensorFlow Lite model."""
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Model file not found: {self.config.model_path}")
        
        try:
            # Loading TensorFlow Lite interpreter
            self.interpreter = TFLITE_INTERPRETER_CLS(model_path=self.config.model_path)
            self.interpreter.allocate_tensors()
            
            # Getting input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            logger.info(f"Custom TensorFlow Lite model loaded: {self.config.model_path}")
            logger.info(f"Input shape: {self.input_details[0]['shape']}")
            logger.info(f"Output shape: {self.output_details[0]['shape']}")
            
        except Exception as e:
            logger.error(f"Failed to load custom TensorFlow Lite model: {e}")
            raise
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocessing image for the custom model.
        
        Args:
            image: Input image as BGR numpy array
            
        Returns:
            Preprocessed image ready for model input
        """
        # Converting BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resizing to model input size
        input_shape = self.input_details[0]['shape']
        height, width = input_shape[1], input_shape[2]
        resized_image = cv2.resize(rgb_image, (width, height))
        
        # Normalizing to [0, 1] range
        normalized_image = resized_image.astype(np.float32) / 255.0
        
        # Adding batch dimension
        input_tensor = np.expand_dims(normalized_image, axis=0)
        
        return input_tensor
    
    def _postprocess_output(self, output: np.ndarray, original_shape: Tuple[int, int]) -> Optional[Dict]:
        """Postprocessing model output to extract pose landmarks.
        
        Args:
            output: Raw model output
            original_shape: Original image dimensions (height, width)
            
        Returns:
            Dictionary with pose landmarks or None if no pose detected
        """
        try:
            # The exact postprocessing depends on your custom model's output format
            # This is a generic implementation that may need adjustment
            
            # Assuming output contains landmark coordinates
            # Shape might be [1, 33, 3] for 33 landmarks with x, y, z coordinates
            if len(output.shape) >= 2 and output.shape[-1] >= 2:
                landmarks = output[0] if len(output.shape) == 3 else output
                
                # Converting normalized coordinates to pixel coordinates
                orig_height, orig_width = original_shape
                
                pose_landmarks = []
                for i, landmark in enumerate(landmarks):
                    if len(landmark) >= 2:
                        x = landmark[0] * orig_width
                        y = landmark[1] * orig_height
                        z = landmark[2] if len(landmark) > 2 else 0.0
                        visibility = landmark[3] if len(landmark) > 3 else 1.0
                        
                        # Only include landmarks above confidence threshold
                        if visibility >= self.config.confidence_threshold:
                            pose_landmarks.append({
                                'x': float(x),
                                'y': float(y),
                                'z': float(z),
                                'visibility': float(visibility)
                            })
                        else:
                            pose_landmarks.append(None)
                
                return {
                    'pose_landmarks': pose_landmarks,
                    'pose_world_landmarks': None  # World landmarks not available in this implementation
                }
            
        except Exception as e:
            logger.error(f"Error postprocessing model output: {e}")
        
        return None
    
    def process(self, image: np.ndarray) -> Optional[Dict]:
        """Processing image to detect pose landmarks.
        
        Args:
            image: Input image as BGR numpy array
            
        Returns:
            Dictionary with pose detection results or None if detection failed
        """
        if self.interpreter is None:
            logger.error("Model not loaded properly")
            return None
        
        try:
            # Storing original shape
            original_shape = image.shape[:2]
            
            # Preprocessing image
            input_tensor = self._preprocess_image(image)
            
            # Running inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()
            
            # Getting output
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Postprocessing to extract landmarks
            results = self._postprocess_output(output, original_shape)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing image with custom model: {e}")
            return None


def create_custom_tflite_detector(model_path: str, **kwargs) -> Optional[CustomTFLitePoseDetector]:
    """Creating a custom TensorFlow Lite pose detector.
    
    Args:
        model_path: Path to the custom TensorFlow Lite model file
        **kwargs: Additional configuration parameters
        
    Returns:
        CustomTFLitePoseDetector instance or None if creation failed
    """
    try:
        config = CustomTFLiteConfig(model_path=model_path, **kwargs)
        detector = CustomTFLitePoseDetector(config)
        return detector
    except Exception as e:
        logger.error(f"Failed to create custom TensorFlow Lite detector: {e}")
        return None


def is_custom_model_available(model_path: str) -> bool:
    """Checking if a custom TensorFlow Lite model is available and valid.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        True if model is available and can be loaded, False otherwise
    """
    if TFLITE_INTERPRETER_CLS is None:
        return False
    
    if not os.path.exists(model_path):
        return False
    
    try:
        # Attempting to load the model to verify it's valid
        interpreter = TFLITE_INTERPRETER_CLS(model_path=model_path)
        interpreter.allocate_tensors()
        return True
    except Exception as e:
        logger.debug(f"Model validation failed for {model_path}: {e}")
        return False


def convert_custom_results_to_mediapipe_format(custom_results: Dict) -> Optional[object]:
    """Converting custom model results to MediaPipe-compatible format.
    
    This function helps maintain compatibility with existing MediaPipe-based
    pose processing code when using custom TensorFlow Lite models.
    
    Args:
        custom_results: Results from custom TensorFlow Lite model
        
    Returns:
        Object with MediaPipe-compatible interface or None
    """
    if not custom_results or 'pose_landmarks' not in custom_results:
        return None
    
    try:
        # Creating a simple object that mimics MediaPipe results structure
        class CustomPoseResults:
            def __init__(self, landmarks):
                self.pose_landmarks = landmarks
                self.pose_world_landmarks = None
                self.segmentation_mask = None
        
        # Converting landmark format if needed
        landmarks = custom_results['pose_landmarks']
        
        # Creating MediaPipe-compatible results object
        results = CustomPoseResults(landmarks)
        return results
        
    except Exception as e:
        logger.error(f"Error converting custom results to MediaPipe format: {e}")
        return None