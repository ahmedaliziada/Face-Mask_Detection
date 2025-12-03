# Architecture Documentation

## System Overview

The Face Mask Detection System is a modular, production-ready application that combines classical computer vision techniques with deep learning to detect and classify face mask usage in images.

## 🏛️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface (Streamlit)               │
│                          app.py                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │   UI     │  │  Utils   │  │  Config  │  │ Detector │   │
│  │  (ui.py) │  │(utils.py)│  │(config.py)│  │(detector │   │
│  │          │  │          │  │          │  │   .py)   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Processing Pipeline                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Face         │→ │ Preprocessing │→ │ Classification│     │
│  │ Detection    │  │ & Resize     │  │ (CNN Model)   │     │
│  │ (Haar)       │  │              │  │               │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    External Libraries                        │
│   OpenCV  │  TensorFlow/Keras  │  NumPy  │  Streamlit      │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Component Architecture

### 1. **app.py** - Main Entry Point
- **Responsibility**: Application orchestration
- **Functions**:
  - Initialize Streamlit configuration
  - Coordinate between UI and detection modules
  - Handle user interactions
  - Manage application state
- **Dependencies**: All src modules

### 2. **src/config.py** - Configuration Module
- **Responsibility**: Centralized configuration
- **Contains**:
  - Path configurations
  - Model parameters
  - UI settings
  - Detection thresholds
- **Pattern**: Configuration as code
- **Benefits**: Easy modification without code changes

### 3. **src/detector.py** - Detection Engine
- **Responsibility**: Core detection logic
- **Class**: `MaskDetector`
- **Key Methods**:
  ```python
  load_model()              # Load trained CNN model
  load_face_cascade()       # Initialize face detector
  detect_faces()            # Find faces in image
  preprocess_face()         # Prepare face for model
  predict_mask()            # Classify mask presence
  annotate_image()          # Draw results on image
  ```
- **Design Pattern**: Facade pattern (simplified interface)
- **Dependencies**: TensorFlow, OpenCV, NumPy

### 4. **src/ui.py** - User Interface Components
- **Responsibility**: UI rendering and styling
- **Functions**:
  - `apply_custom_css()`: Apply styling
  - `render_header()`: Display title/header
  - `render_about_section()`: Project information
  - `render_sidebar()`: Configuration sidebar
  - `display_statistics()`: Show detection stats
  - `display_compliance_status()`: Compliance alerts
- **Design Pattern**: Component-based UI
- **Dependencies**: Streamlit

### 5. **src/utils.py** - Utility Functions
- **Responsibility**: Helper functions
- **Functions**:
  - Image loading from uploads
  - Color space conversions
  - Status message generation
- **Pattern**: Utility/Helper module
- **Dependencies**: OpenCV, NumPy

## 🔄 Data Flow

### Request Flow
```
1. User uploads image
   ↓
2. Image loaded via utils.load_image_from_upload()
   ↓
3. detector.predict_mask() called
   ↓
4. Face detection (Haar Cascade)
   ↓
5. Each face preprocessed
   ↓
6. CNN model prediction
   ↓
7. Results annotated on image
   ↓
8. Statistics calculated
   ↓
9. UI displays results
```

### Detection Pipeline
```
Input Image (Any size, RGB)
   ↓
[Face Detection - Haar Cascade]
   ↓
Multiple face regions (x, y, w, h)
   ↓
[For each face]
   ├─ Extract face region
   ├─ Convert BGR → RGB
   ├─ Resize to 128×128
   ├─ Normalize (divide by 255)
   ├─ Add batch dimension
   ↓
[CNN Model Inference]
   ↓
[Mask, No Mask] probabilities
   ↓
[Post-processing]
   ├─ Determine label
   ├─ Calculate confidence
   ├─ Draw bounding box
   ├─ Add label text
   ↓
Annotated Output Image
```

## 🧠 Model Architecture

### CNN Model (Simplified)
```
Input Layer (128, 128, 3)
   ↓
[Convolutional Blocks]
   ├─ Conv2D + ReLU
   ├─ BatchNormalization
   ├─ MaxPooling
   ├─ Dropout
   ↓
[Flatten Layer]
   ↓
[Dense Layers]
   ├─ Dense(128) + ReLU
   ├─ Dropout
   ↓
Output Layer (2 units, Softmax)
   ├─ [0]: Mask probability
   └─ [1]: No Mask probability
```

## 🔧 Design Patterns Used

### 1. **Singleton Pattern** (Implicit)
- `@st.cache_resource` ensures single model instance
- Reduces memory usage and load time

### 2. **Facade Pattern**
- `MaskDetector` class provides simple interface
- Hides complex TensorFlow/OpenCV operations

### 3. **Separation of Concerns**
- UI logic separated from business logic
- Configuration isolated in dedicated module
- Utility functions in separate module

### 4. **Dependency Injection**
- Model path configurable via constructor
- Easy testing with mock models

## 🔐 Security Considerations

1. **Input Validation**
   - File type checking (png, jpg, jpeg)
   - File size limits (Streamlit default: 200MB)

2. **Error Handling**
   - Try-catch blocks for model loading
   - Graceful degradation on errors
   - User-friendly error messages

3. **Resource Management**
   - Model cached to prevent memory leaks
   - Images processed in memory
   - No persistent storage of uploads

## 📈 Scalability Considerations

### Current Architecture
- **Deployment**: Single-instance Streamlit app
- **Processing**: Synchronous, sequential
- **Suitable for**: Small to medium workloads

### Scaling Options

1. **Horizontal Scaling**
   - Deploy behind load balancer
   - Use container orchestration (Docker + Kubernetes)

2. **Async Processing**
   - Add message queue (RabbitMQ, Redis)
   - Background workers for batch processing

3. **Model Optimization**
   - Convert to TensorFlow Lite
   - Use ONNX for cross-platform inference
   - Implement model quantization

## 🧪 Testing Strategy

### Unit Tests
- `test_detector.py`: Test detection logic
- `test_utils.py`: Test utility functions
- `test_config.py`: Validate configuration

### Integration Tests
- End-to-end image processing
- UI component rendering
- Model inference pipeline

### Performance Tests
- Inference latency
- Memory usage
- Concurrent user handling

## 🚀 Deployment Options

### 1. **Streamlit Cloud**
```bash
streamlit run app.py
```

### 2. **Docker Container**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

### 3. **Cloud Platforms**
- AWS EC2 + Docker
- Google Cloud Run
- Azure Container Instances
- Heroku

## 📊 Performance Metrics

### Expected Performance
- **Inference Time**: ~100-300ms per face
- **Memory Usage**: ~500MB-1GB (model loaded)
- **Throughput**: 5-10 images/minute (single instance)

### Optimization Opportunities
1. Batch processing for multiple faces
2. GPU acceleration for faster inference
3. Model pruning for reduced size
4. Caching for repeated images

## 🔄 Future Enhancements

1. **Real-time Video Processing**
   - Webcam integration
   - Video file upload support

2. **Advanced Features**
   - Face mask type classification
   - Proper mask fit detection
   - Crowd density monitoring

3. **API Development**
   - REST API for integration
   - Webhook support for notifications

4. **Analytics Dashboard**
   - Historical compliance tracking
   - Time-series analysis
   - Export reports

## 📚 References

- **Face Detection**: Viola-Jones algorithm (Haar Cascades)
- **Deep Learning**: CNN architecture for image classification
- **Framework**: TensorFlow/Keras documentation
- **UI**: Streamlit best practices

---

**Document Version**: 1.0  
**Last Updated**: December 2025  
**Maintained By**: Development Team
