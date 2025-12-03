"""Configuration settings for the Face Mask Detection application."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "face_mask_detection_model.h5"
DATA_PATH = PROJECT_ROOT / "data"

# Model configuration
MODEL_INPUT_SIZE = (128, 128)
CONFIDENCE_THRESHOLD = 0.5

# Face detection configuration
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
FACE_DETECTION_SCALE_FACTOR = 1.1
FACE_DETECTION_MIN_NEIGHBORS = 5

# UI configuration
PAGE_TITLE = "Face Mask Detection"
PAGE_ICON = "😷"
LAYOUT = "wide"

# Colors (BGR format for OpenCV)
COLOR_MASK = (0, 255, 0)      # Green
COLOR_NO_MASK = (255, 0, 0)   # Red

# Labels
LABEL_MASK = "Mask"
LABEL_NO_MASK = "No Mask"
