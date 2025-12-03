"""Utility functions for the application."""

import cv2
import numpy as np
from typing import Tuple


def load_image_from_upload(uploaded_file) -> np.ndarray:
    """Load image from Streamlit file uploader.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Image array in BGR format
    """
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    return image


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to RGB.
    
    Args:
        image: Image array in BGR format
        
    Returns:
        Image array in RGB format
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def get_compliance_status(compliance_rate: float) -> Tuple[str, str]:
    """Get compliance status message and type.
    
    Args:
        compliance_rate: Percentage of people wearing masks
        
    Returns:
        Tuple of (message, status_type)
    """
    if compliance_rate == 100:
        return (
            "✅ 100% Mask Compliance - All people are wearing masks!",
            "success"
        )
    elif compliance_rate >= 50:
        return (
            f"⚠️ {compliance_rate:.1f}% Mask Compliance - Some people not wearing masks",
            "warning"
        )
    else:
        return (
            f"❌ {compliance_rate:.1f}% Mask Compliance - Most people not wearing masks",
            "error"
        )
