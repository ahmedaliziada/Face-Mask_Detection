"""Streamlit UI components and styling."""

import streamlit as st


def apply_custom_css():
    """Apply custom CSS styling to the application."""
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            font-weight: bold;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            border: none;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .result-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render the application header."""
    st.title('😷 Face Mask Detection Application')
    st.markdown("### AI-Powered COVID-19 Safety Compliance System")


def render_about_section():
    """Render the about/project description section."""
    with st.expander("ℹ️ About This Project", expanded=False):
        st.markdown("""
        #### Computer Vision & Deep Learning for Mask Detection
        
        This application uses **Convolutional Neural Networks (CNNs)** to detect whether people in images 
        are wearing face masks. Built during the COVID-19 pandemic era, this demonstrates how AI can be 
        applied to public health and safety monitoring.
        
        **Key Technologies:**
        
        - **Deep Learning**: Custom CNN model trained on mask/no-mask datasets
        - **Computer Vision**: OpenCV for face detection using Haar Cascade classifiers
        - **TensorFlow/Keras**: Deep learning framework for model training and inference
        - **Image Processing**: Real-time face detection and classification
        
        **How It Works:**
        1. **Face Detection**: Haar Cascade identifies faces in the image
        2. **Preprocessing**: Detected faces are resized and normalized
        3. **Classification**: CNN model predicts mask presence
        4. **Visualization**: Results displayed with bounding boxes and labels
        
        **Real-World Applications:**
        - Public space monitoring (malls, airports, offices)
        - Automated compliance checking
        - Safety protocol enforcement
        - Healthcare facility management
        - Educational institution monitoring
        """)


def render_sidebar():
    """Render the sidebar with configuration and information."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("#### Model Information")
        st.info("""
        **Model Type**: CNN
        
        **Input Size**: 128x128 RGB
        
        **Classes**: 
        - ✅ Mask
        - ❌ No Mask
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Detection Process")
        st.markdown("""
        1. Upload image
        2. Face detection
        3. Mask classification
        4. Results visualization
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 Tips")
        st.info("""
        - Use clear, front-facing photos
        - Good lighting improves accuracy
        - Multiple faces supported
        - Works with various mask types
        """)


def display_statistics(stats: dict):
    """Display detection statistics.
    
    Args:
        stats: Dictionary containing detection statistics
    """
    st.markdown("---")
    st.markdown("##### 📊 Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Faces", stats['total_faces'])
    with col2:
        st.metric("😷 With Mask", stats['mask_count'])
    with col3:
        st.metric("❌ No Mask", stats['no_mask_count'])


def display_compliance_status(compliance_rate: float, message: str, status_type: str):
    """Display compliance status message.
    
    Args:
        compliance_rate: Compliance percentage
        message: Status message
        status_type: Type of status ('success', 'warning', 'error')
    """
    if status_type == "success":
        st.success(message)
    elif status_type == "warning":
        st.warning(message)
    else:
        st.error(message)
