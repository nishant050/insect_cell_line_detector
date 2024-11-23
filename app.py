
import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
import io
from typing import Optional, Tuple, List, Dict, Generator
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import joblib
import numpy as np
from skimage import color
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
import os



###################
# Data Handling
###################
class DataLoader:
    """Handles all data loading operations (images and ground truth)"""
    
    # Supported image formats
    SUPPORTED_FORMATS = {
        'jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp', 'gif', 'webp'
    }
    
    @staticmethod
    def get_file_extension(file) -> str:
        """
        Get file extension from filename
        
        Args:
            file: Streamlit uploaded file object
        Returns:
            file extension in lowercase
        """
        return file.name.split('.')[-1].lower()

    @classmethod
    def load_image(cls, image_file) -> Optional[np.ndarray]:
        """
        Load and preprocess uploaded image
        Supports multiple formats including TIFF
        
        Args:
            image_file: Streamlit uploaded file object
        Returns:
            numpy array of RGB image or None if loading fails
        """
        if image_file is not None:
            try:
                # Get file extension
                file_ext = cls.get_file_extension(image_file)
                
                # Verify file format
                if file_ext not in cls.SUPPORTED_FORMATS:
                    st.error(f"Unsupported file format: {file_ext}")
                    st.info(f"Supported formats: {', '.join(cls.SUPPORTED_FORMATS)}")
                    return None
                
                # Read image bytes
                image_bytes = image_file.getvalue()
                
                # Handle TIFF files separately
                if file_ext in ['tif', 'tiff']:
                    try:
                        # Use PIL for TIFF files
                        img = Image.open(io.BytesIO(image_bytes))
                        # Convert to RGB if necessary
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        # Convert to numpy array
                        image = np.array(img)
                    except Exception as e:
                        st.error(f"Error loading TIFF file: {str(e)}")
                        return None
                else:
                    # Handle other formats using OpenCV
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError("Failed to decode image")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Validate image
                if not isinstance(image, np.ndarray):
                    raise ValueError("Failed to convert image to numpy array")
                
                if len(image.shape) != 3:
                    st.warning("Image is not in RGB format. Converting...")
                    # Handle grayscale images
                    if len(image.shape) == 2:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    else:
                        raise ValueError("Invalid image dimensions")
                
                # Validate image dimensions and content
                if image.size == 0:
                    raise ValueError("Image is empty")
                if np.max(image) == np.min(image):
                    st.warning("Image has no contrast (single value)")
                
                return image
                
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                st.info("Please try another image or format")
                return None
        return None

    @staticmethod
    def load_csv(txt_file) -> Optional[np.ndarray]:
        """
        Load ground truth boxes from YOLO format text file
        
        Args:
            txt_file: Streamlit uploaded file object
        Returns:
            numpy array of bounding boxes [x1, y1, x2, y2] or None if loading fails
        """
        if txt_file is not None:
            try:
                # Read text file content
                content = txt_file.getvalue().decode('utf-8')
                
                # Get image dimensions from the current displayed image
                # We need to access these through Streamlit's session state
                if 'current_image' not in st.session_state:
                    st.error("Please upload an image first")
                    return None
                
                image = st.session_state.current_image
                image_height, image_width = image.shape[:2]
                
                boxes = []
                # Process each line
                for line in content.splitlines():
                    if line.strip():  # Skip empty lines
                        # Parse YOLO format: class_id x y w h
                        class_id, x, y, w, h = map(float, line.strip().split())
                        
                        # Convert YOLO format to absolute coordinates
                        x1 = int((x - w/2) * image_width)
                        y1 = int((y - h/2) * image_height)
                        x2 = int((x + w/2) * image_width)
                        y2 = int((y + h/2) * image_height)
                        
                        boxes.append([x1, y1, x2, y2])
                
                return np.array(boxes)
                
            except Exception as e:
                st.error(f"Error loading ground truth file: {str(e)}")
                return None
        return None

#############
# HOG + SVM detector
#############

class SimpleDetector:
    """
    Object detector using HOG features and sliding windows.
    
    Attributes:
        model: Trained classifier (typically sklearn.svm.SVC)
        scaler: Feature scaler (typically sklearn.preprocessing.StandardScaler)
        window_size (tuple): Size of detection window (width, height)
        step_size (int): Pixels to move window in each step
        downscale (float): Factor to reduce image size in pyramid
        threshold (float): Confidence threshold for detections
        nms_threshold (float): IoU threshold for non-max suppression
        hog_params (dict): Parameters for HOG feature extraction
    
    Example:
        >>> detector = SimpleDetector('model.npy', 'scaler.npy')
        >>> image = cv2.imread('image.jpg')
        >>> image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        >>> detections = detector.detect(image)
    """
    
    def __init__(self, 
                 model_path: str, 
                 scaler_path: str,
                 window_size: Tuple[int, int] = (64, 64),
                 step_size: int = 10,
                 downscale: float = 1.5,
                 threshold: float = 2.0,
                 nms_threshold: float = 0.008) -> None:
        """
        Initialize the detector with model and parameters.
        
        Args:
            model_path: Path to saved classifier model (.npy file)
            scaler_path: Path to saved feature scaler (.npy file)
            window_size: Size of detection window (width, height)
            step_size: Pixels to move window in each step
            downscale: Factor to reduce image size in pyramid
            threshold: Confidence threshold for detections
            nms_threshold: IoU threshold for non-max suppression
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.window_size = window_size
        self.step_size = step_size
        self.downscale = downscale
        self.threshold = threshold
        self.nms_threshold = nms_threshold
        
        self.hog_params = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'block_norm': 'L2'
        }
    
    def sliding_window(self, 
                      image: np.ndarray) -> Generator[Tuple[int, int, np.ndarray], None, None]:
        """
        Generate sliding windows over the image.
        
        Args:
            image: Input image array (height, width, channels)
        
        Yields:
            Tuple containing:
                - x: X coordinate of window
                - y: Y coordinate of window
                - window: Image patch at (x,y)
        
        Example:
            >>> for (x, y, window) in detector.sliding_window(image):
            ...     # Process each window
            ...     process_window(window)
        """
        for y in range(0, image.shape[0], self.step_size):
            for x in range(0, image.shape[1], self.step_size):
                window = image[y:y + self.window_size[1], x:x + self.window_size[0]]
                if window.shape[:2] == self.window_size:
                    yield (x, y, window)

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect objects in the image.
        
        Args:
            image: Input image array (height, width, channels)
        
        Returns:
            Array of detection boxes, each box is [x1, y1, x2, y2]
        
        Example:
            >>> image = cv2.imread('image.jpg')
            >>> image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            >>> boxes = detector.detect(image)
            >>> print(f"Found {len(boxes)} objects")
        """
        detections = []
        scale = 0
        
        # Create image pyramid
        for resized in pyramid_gaussian(image, downscale=self.downscale, channel_axis=-1):
            # Process each window
            for (x, y, window) in self.sliding_window(resized):
                window = color.rgb2gray(window)
                
                # Extract HOG features
                features = hog(window, **self.hog_params)
                features = self.scaler.transform(features.reshape(1, -1))
                
                # Predict
                if self.model.predict(features) == 1:
                    if self.model.decision_function(features) > self.threshold:
                        detection_scale = self.downscale ** scale
                        detections.append([
                            int(x * detection_scale),
                            int(y * detection_scale),
                            int((x + self.window_size[0]) * detection_scale),
                            int((y + self.window_size[1]) * detection_scale)
                        ])
            
            scale += 1
            
            # Stop if image becomes too small
            if resized.shape[0] < self.window_size[1] or resized.shape[1] < self.window_size[0]:
                break
                
        # Apply non-maximum suppression
        if detections:
            detections = np.array(detections)
            scores = np.ones(len(detections))  # Simplified scoring
            pick = non_max_suppression(detections, scores, self.nms_threshold)
            return pick
        return np.array([])





###################
# Model Management
###################
class ModelManager:
    def __init__(self):
        self.models = {}
        self.load_models()

    def load_models(self):
        """Load all models"""
        try:
            # Load YOLOv8 model
            self.models['YOLOv8'] = YOLO('yolo.pt')

            # Load Faster R-CNN model
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=8)
            model.eval()
            self.models['Faster R-CNN'] = model

            # Optionally load model weights if you have a trained model
            model.load_state_dict(torch.load('frcnn.pth', map_location=torch.device('cpu')))
        
        except Exception as e:
            st.error(f"Error loading YOLOv8 or Faster R-CNN models: {str(e)}")

        try:
            # Load HOG + SVM model
            self.models['HOG_SVM'] = SimpleDetector(
                model_path='model.npy',
                scaler_path='scalar.npy'
            )
        except Exception as e:
            st.error(f"Error loading HOG_SVM model: {str(e)}")

    def predict(self, image: np.ndarray, model_type: str, conf: float, iou: float) -> np.ndarray:
        """
        Make predictions using selected model
        """
        try:
            if model_type == "YOLOv8":
                predictions = self.models[model_type].predict(
                    image, conf=conf, iou=iou
                )[0].boxes.xyxy.cpu().numpy()
                return predictions

            elif model_type == "Faster R-CNN":
                # Convert image to PyTorch tensor and normalize
                img_tensor = torch.tensor(image).permute(2, 0, 1).float()  # [C, H, W]
                img_tensor = img_tensor / 255.0  # Normalize to [0, 1]
                
                # Make sure it's a list of tensors
                img_tensor = [img_tensor]

                # Get predictions
                with torch.no_grad():
                    predictions = self.models[model_type](img_tensor)[0]

                # Filter predictions by confidence threshold
                scores = predictions['scores'].numpy()
                boxes = predictions['boxes'].numpy()
                mask = scores >= conf
                boxes = boxes[mask]

                return boxes.astype(int)

            elif model_type == "HOG_SVM":
                # Convert image to RGB if needed
                if image.shape[2] == 4:  # RGBA
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                elif image.shape[2] == 1:  # Grayscale
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                # Get predictions
                boxes = self.models[model_type].detect(image)
                return boxes
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")
            return None

###################
# Visualization
###################
class Visualizer:
    """Handles all visualization operations"""
    
    @staticmethod
    def plot_predictions(
        image: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        predictions: Optional[np.ndarray] = None
    ) -> plt.Figure:
        """
        Plot image with ground truth and predicted boxes
        
        Args:
            image: numpy array of RGB image
            ground_truth: numpy array of ground truth boxes [x1, y1, x2, y2]
            predictions: numpy array of predicted boxes [x1, y1, x2, y2]
        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(image)
        
        # Plot ground truth boxes in green
        if ground_truth is not None:
            for box in ground_truth:
                x1, y1, x2, y2 = box
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, color='green', linewidth=1,
                                   label='Ground Truth')
                ax.add_patch(rect)
        
        # Plot predicted boxes in red
        if predictions is not None:
            for box in predictions:
                x1, y1, x2, y2 = box
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, color='red', linewidth=1,
                                   label='Prediction')
                ax.add_patch(rect)
        
        # Add legend (only one entry per class)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        ax.axis('off')
        plt.tight_layout()
        return fig

###################
# UI Components
###################
class UIComponents:
    """Handles all UI component creation and layout"""

    @staticmethod
    def create_sidebar() -> Dict:
        """Create and return sidebar elements"""
        st.sidebar.header("Model Settings")

        def on_change():
            # Toggle a session state variable to implicitly trigger a rerun
            st.session_state["settings_changed"] = not st.session_state.get("settings_changed", False)

        settings = {
            'model_type': st.sidebar.selectbox(
                "Select Model",
                ["YOLOv8", "Faster R-CNN", "HOG_SVM"],
                on_change=on_change
            ),
            'conf_threshold': st.sidebar.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                on_change=on_change
            ),
            'iou_threshold': st.sidebar.slider(
                "IOU Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                on_change=on_change
            )
        }
        return settings


    @staticmethod
    def create_upload_section() -> Tuple:
        """Create and return file upload section"""
        st.header("Upload Data")
        col1, col2 = st.columns(2)
        with col1:
            image_file = st.file_uploader(
                "Upload Image",
                type=list(DataLoader.SUPPORTED_FORMATS)
            )
        with col2:
            # Change CSV to TXT file upload
            csv_file = st.file_uploader("Upload Ground Truth (YOLO format)", type=['txt'])
        return image_file, csv_file

    @staticmethod
    def show_instructions():
        """Display usage instructions"""
        st.info("Please upload an image to begin.")
        st.markdown("""
        ### Instructions:
        1. Select a model from the sidebar
        2. Adjust confidence and IOU thresholds if needed
        3. Upload an image for detection
        4. Optionally upload ground truth boxes in CSV format
        5. Click 'Make Predictions' to run the model
        
        The CSV file should have columns: x1, y1, x2, y2
        """)

###################
# Main Application
###################
class InsectDetectionApp:
    """Main application class"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.model_manager = ModelManager()
        self.visualizer = Visualizer()
        self.ui = UIComponents()

    def run(self):
        """Run the Streamlit application"""
        st.title("Insect Detection Model Interface")
        
        # Create UI components with callbacks
        settings = self.ui.create_sidebar()
        
        # Initialize session state for tracking changes
        if 'last_settings' not in st.session_state:
            st.session_state.last_settings = {}
        if 'current_image' not in st.session_state:
            st.session_state.current_image = None
            
        # Create upload section
        image_file, csv_file = self.ui.create_upload_section()
        
        # Process image if uploaded
        if image_file is not None:
            # Load and display image
            image = self.data_loader.load_image(image_file)
            if image is not None:
                # Store image in session state
                st.session_state.current_image = image
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Load ground truth if provided
                ground_truth = self.data_loader.load_csv(csv_file)
                
                # Check if settings have changed
                settings_changed = (
                    st.session_state.last_settings.get('model_type') != settings['model_type'] or
                    st.session_state.last_settings.get('conf_threshold') != settings['conf_threshold'] or
                    st.session_state.last_settings.get('iou_threshold') != settings['iou_threshold']
                )
                
                # Make predictions automatically if image is loaded or settings changed
                with st.spinner("Processing..."):
                    # Get predictions
                    predictions = self.model_manager.predict(
                        image,
                        settings['model_type'],
                        settings['conf_threshold'],
                        settings['iou_threshold']
                    )
                    
                    if predictions is not None:
                        # Plot results
                        fig = self.visualizer.plot_predictions(
                            image, ground_truth, predictions
                        )
                        st.pyplot(fig)
                
                # Update last settings
                st.session_state.last_settings = settings.copy()
        else:
            self.ui.show_instructions()

###################
# Run Application
###################
if __name__ == "__main__":
    app = InsectDetectionApp()
    app.run()
