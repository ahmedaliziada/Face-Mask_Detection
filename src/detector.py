"""Face mask detection logic."""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from typing import Tuple, List
import config


class MaskDetector:
    """Face mask detector using CNN model."""
    
    def __init__(self, model_path: str = None):
        """Initialize the mask detector.
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path or str(config.MODEL_PATH)
        self.model = None
        self.face_cascade = None
        
    def load_model(self):
        """Load the trained mask detection model."""
        try:
            self.model = load_model(self.model_path)
            return True
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def load_face_cascade(self):
        """Load Haar Cascade for face detection."""
        cascade_path = cv2.data.haarcascades + config.FACE_CASCADE_PATH
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise Exception("Failed to load face cascade classifier")
    
    def detect_faces(self, image: np.ndarray) -> np.ndarray:
        """Detect faces in an image.
        
        Args:
            image: Input image array
            
        Returns:
            Array of face coordinates (x, y, width, height)
        """
        if self.face_cascade is None:
            self.load_face_cascade()
            
        faces = self.face_cascade.detectMultiScale(
            image,
            scaleFactor=config.FACE_DETECTION_SCALE_FACTOR,
            minNeighbors=config.FACE_DETECTION_MIN_NEIGHBORS
        )
        return faces
    
    def preprocess_face(self, face: np.ndarray) -> np.ndarray:
        """Preprocess face image for model prediction.
        
        Args:
            face: Face image array
            
        Returns:
            Preprocessed face array
        """
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, config.MODEL_INPUT_SIZE)
        face = np.array(face) / 255.0
        face = np.expand_dims(face, axis=0)
        return face
    
    def predict_mask(self, image: np.ndarray) -> Tuple[List, List]:
        """Detect faces and predict mask presence.
        
        Args:
            image: Input image array
            
        Returns:
            Tuple of (faces, predictions)
        """
        if self.model is None:
            self.load_model()
            
        faces = self.detect_faces(image)
        predictions = []
        
        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]
            processed_face = self.preprocess_face(face)
            prediction = self.model.predict(processed_face, verbose=0)
            predictions.append(prediction)
        
        return faces, predictions
    
    def annotate_image(self, image: np.ndarray, faces: List, 
                      predictions: List) -> Tuple[np.ndarray, dict]:
        """Annotate image with detection results.
        
        Args:
            image: Input image array (RGB)
            faces: List of face coordinates
            predictions: List of model predictions
            
        Returns:
            Tuple of (annotated image, statistics dictionary)
        """
        result_image = image.copy()
        mask_count = 0
        no_mask_count = 0
        
        for i, (x, y, w, h) in enumerate(faces):
            (mask, withoutMask) = predictions[i][0]
            
            # Determine label and color
            label = config.LABEL_NO_MASK if mask > withoutMask else config.LABEL_MASK
            confidence = max(mask, withoutMask) * 100
            
            if label == config.LABEL_MASK:
                mask_count += 1
                color = config.COLOR_MASK
                bg_color = (40, 200, 40)  # Lighter green for background
            else:
                no_mask_count += 1
                color = config.COLOR_NO_MASK
                bg_color = (220, 40, 40)  # Lighter red for background
            
            # Draw rounded rectangle with thicker border
            thickness = 3
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, thickness)
            
            # Draw small corner accents for modern look
            corner_length = min(20, w // 5, h // 5)
            # Top-left corner
            cv2.line(result_image, (x, y), (x + corner_length, y), color, thickness + 2)
            cv2.line(result_image, (x, y), (x, y + corner_length), color, thickness + 2)
            # Top-right corner
            cv2.line(result_image, (x + w, y), (x + w - corner_length, y), color, thickness + 2)
            cv2.line(result_image, (x + w, y), (x + w, y + corner_length), color, thickness + 2)
            # Bottom-left corner
            cv2.line(result_image, (x, y + h), (x + corner_length, y + h), color, thickness + 2)
            cv2.line(result_image, (x, y + h), (x, y + h - corner_length), color, thickness + 2)
            # Bottom-right corner
            cv2.line(result_image, (x + w, y + h), (x + w - corner_length, y + h), color, thickness + 2)
            cv2.line(result_image, (x + w, y + h), (x + w, y + h - corner_length), color, thickness + 2)
            
            # Prepare label text
            text = f"{label} ({confidence:.1f}%)"
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.7
            font_thickness = 2
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            
            # Calculate label background position (above the face box)
            label_y_pos = y - 15 if y - 15 > text_height + 10 else y + h + text_height + 15
            
            # Draw shadow for text background (offset)
            shadow_offset = 3
            cv2.rectangle(
                result_image,
                (x - 2 + shadow_offset, label_y_pos - text_height - 8 + shadow_offset),
                (x + text_width + 12 + shadow_offset, label_y_pos + 8 + shadow_offset),
                (0, 0, 0),
                -1
            )
            
            # Draw label background with rounded effect
            cv2.rectangle(
                result_image,
                (x - 2, label_y_pos - text_height - 8),
                (x + text_width + 12, label_y_pos + 8),
                bg_color,
                -1
            )
            
            # Draw border around label
            cv2.rectangle(
                result_image,
                (x - 2, label_y_pos - text_height - 8),
                (x + text_width + 12, label_y_pos + 8),
                color,
                2
            )
            
            # Draw text shadow
            cv2.putText(
                result_image,
                text,
                (x + 5 + 2, label_y_pos + 2),
                font,
                font_scale,
                (0, 0, 0),
                font_thickness + 1
            )
            
            # Draw main text
            cv2.putText(
                result_image,
                text,
                (x + 5, label_y_pos),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness
            )
        
        stats = {
            'total_faces': len(faces),
            'mask_count': mask_count,
            'no_mask_count': no_mask_count,
            'compliance_rate': (mask_count / len(faces) * 100) if len(faces) > 0 else 0
        }
        
        return result_image, stats
