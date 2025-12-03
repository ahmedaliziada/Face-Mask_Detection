"""Main application entry point."""

import streamlit as st
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import config
from detector import MaskDetector
from utils import load_image_from_upload, bgr_to_rgb, get_compliance_status
from ui import (
    apply_custom_css, 
    render_header, 
    render_about_section, 
    render_sidebar,
    display_statistics,
    display_compliance_status
)


# Page Configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_detector():
    """Load and cache the mask detector."""
    try:
        detector = MaskDetector()
        detector.load_model()
        detector.load_face_cascade()
        return detector
    except Exception as e:
        st.error(f"Error initializing detector: {str(e)}")
        return None


def main():
    """Main application function."""
    # Apply styling and render UI components
    apply_custom_css()
    render_header()
    render_about_section()
    st.markdown("---")
    render_sidebar()
    st.markdown("---")
    
    # Load detector
    detector = load_detector()
    
    if detector is None:
        st.error("❌ Failed to load the model. Please check the model path.")
        return
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            'Choose an image file',
            type=['png', 'jpeg', 'jpg'],
            help="Upload a clear image with visible faces"
        )
    
    # Process uploaded image
    if uploaded_file is not None:
        with col1:
            image = load_image_from_upload(uploaded_file)
            rgb_image = bgr_to_rgb(image)
            st.image(rgb_image, caption="Original Image", width=600)
        
        with col2:
            st.markdown("#### 🔍 Detection Results")
            
            with st.spinner("🔄 Detecting faces and analyzing masks..."):
                faces, predictions = detector.predict_mask(image)
                
                if len(faces) == 0:
                    st.warning("⚠️ No faces detected in the image. Please upload an image with visible faces.")
                else:
                    # Annotate image and get statistics
                    result_image, stats = detector.annotate_image(rgb_image, faces, predictions)
                    
                    st.image(result_image, caption="Detection Results", width=600)
                    
                    # Display statistics
                    display_statistics(stats)
                    
                    # Display compliance status
                    message, status_type = get_compliance_status(stats['compliance_rate'])
                    display_compliance_status(stats['compliance_rate'], message, status_type)
    else:
        with col2:
            st.markdown("#### 🔍 Detection Results")
            st.info("👈 Upload an image to start detection")


if __name__ == "__main__":
    main()
