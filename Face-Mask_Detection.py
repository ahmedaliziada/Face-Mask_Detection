
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Page Configuration
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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

st.title('😷 Face Mask Detection Application')
st.markdown("### AI-Powered COVID-19 Safety Compliance System")

# Project Description
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

st.markdown("---")

# Sidebar
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

st.markdown("---")

# Load model with error handling
@st.cache_resource
def load_mask_model():
    try:
        model = load_model(r"D:\Work\TECH\Route\Projects\CV\Face-Mask_Detection\face_mask_detection_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_mask_model()

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### 📤 Upload Image")
    upload = st.file_uploader(
        'Choose an image file',
        type=['png', 'jpeg', 'jpg'],
        help="Upload a clear image with visible faces"
    )

# Define function for face detection and mask detection
def detect_and_predict_mask(image, model):
    """
    Detect faces and predict mask presence.
    
    Args:
        image: Input image array
        model: Trained mask detection model
    
    Returns:
        faces: Detected face coordinates
        predictions: Mask/no-mask predictions
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1)
    
    predictions = []
    
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (128, 128))
        face = np.array(face) / 255.0
        face = np.expand_dims(face, axis=0)
        
        # Predict mask/no mask
        predictions.append(model.predict(face, verbose=0))
    
    return faces, predictions

if upload is not None and model is not None:
    with col1:
        file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        st.image(original_image, caption="Original Image", width=600)
    
    with col2:
        st.markdown("#### 🔍 Detection Results")
        
        with st.spinner("🔄 Detecting faces and analyzing masks..."):
            faces, predictions = detect_and_predict_mask(image, model)
            
            if len(faces) == 0:
                st.warning("⚠️ No faces detected in the image. Please upload an image with visible faces.")
            else:
                # Draw results on image
                result_image = original_image.copy()
                
                mask_count = 0
                no_mask_count = 0
                
                for i, (x, y, w, h) in enumerate(faces):
                    (mask, withoutMask) = predictions[i][0]
                    # Swap the logic - the model output order seems reversed
                    label = 'No Mask' if mask > withoutMask else 'Mask'
                    confidence = max(mask, withoutMask) * 100
                    
                    if label == 'Mask':
                        mask_count += 1
                        color = (0, 255, 0)  # Green for mask
                        bg_color = (40, 200, 40)  # Lighter green
                    else:
                        no_mask_count += 1
                        color = (255, 0, 0)  # Red for no mask
                        bg_color = (220, 40, 40)  # Lighter red
                    
                    # Draw thicker rectangle
                    thickness = 3
                    cv2.rectangle(result_image, (x, y), (x + w, y + h), color, thickness)
                    
                    # Draw corner accents for modern look
                    corner_length = min(20, w // 5, h // 5)
                    # Top-left
                    cv2.line(result_image, (x, y), (x + corner_length, y), color, thickness + 2)
                    cv2.line(result_image, (x, y), (x, y + corner_length), color, thickness + 2)
                    # Top-right
                    cv2.line(result_image, (x + w, y), (x + w - corner_length, y), color, thickness + 2)
                    cv2.line(result_image, (x + w, y), (x + w, y + corner_length), color, thickness + 2)
                    # Bottom-left
                    cv2.line(result_image, (x, y + h), (x + corner_length, y + h), color, thickness + 2)
                    cv2.line(result_image, (x, y + h), (x, y + h - corner_length), color, thickness + 2)
                    # Bottom-right
                    cv2.line(result_image, (x + w, y + h), (x + w - corner_length, y + h), color, thickness + 2)
                    cv2.line(result_image, (x + w, y + h), (x + w, y + h - corner_length), color, thickness + 2)
                    
                    # Prepare label
                    text = f"{label} ({confidence:.1f}%)"
                    font = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 0.7
                    font_thickness = 2
                    
                    # Get text size
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
                    
                    # Position label
                    label_y_pos = y - 15 if y - 15 > text_height + 10 else y + h + text_height + 15
                    
                    # Draw shadow
                    shadow_offset = 3
                    cv2.rectangle(
                        result_image,
                        (x - 2 + shadow_offset, label_y_pos - text_height - 8 + shadow_offset),
                        (x + text_width + 12 + shadow_offset, label_y_pos + 8 + shadow_offset),
                        (0, 0, 0),
                        -1
                    )
                    
                    # Draw label background
                    cv2.rectangle(
                        result_image,
                        (x - 2, label_y_pos - text_height - 8),
                        (x + text_width + 12, label_y_pos + 8),
                        bg_color,
                        -1
                    )
                    
                    # Draw label border
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
                
                st.image(result_image, caption="Detection Results", width=600)
                
                # Summary statistics
                st.markdown("---")
                st.markdown("##### 📊 Summary")
                
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Total Faces", len(faces))
                with col_stat2:
                    st.metric("😷 With Mask", mask_count)
                with col_stat3:
                    st.metric("❌ No Mask", no_mask_count)
                
                # Compliance status
                compliance_rate = (mask_count / len(faces)) * 100
                if compliance_rate == 100:
                    st.success(f"✅ 100% Mask Compliance - All people are wearing masks!")
                elif compliance_rate >= 50:
                    st.warning(f"⚠️ {compliance_rate:.1f}% Mask Compliance - Some people not wearing masks")
                else:
                    st.error(f"❌ {compliance_rate:.1f}% Mask Compliance - Most people not wearing masks")

elif upload is None:
    with col2:
        st.markdown("#### 🔍 Detection Results")
        st.info("👈 Upload an image to start detection")
elif model is None:
    st.error("❌ Failed to load the model. Please check the model path.")