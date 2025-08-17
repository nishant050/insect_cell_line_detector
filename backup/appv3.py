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
from torchvision.ops import nms
import joblib
from skimage import color
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
import os
from pkg_resources import parse_version

# Version check for ultralytics
try:
    import ultralytics
    if parse_version(ultralytics.__version__) < parse_version("8.0.0"):
        st.warning(
            f"Your `ultralytics` version is {ultralytics.__version__}, but this app is optimized for version 8.0.0 or higher. "
            "Please upgrade for optimal performance: `pip install --upgrade ultralytics`"
        )
except ImportError:
    st.error("`ultralytics` is not installed. Please install it using `pip install ultralytics`")


###################
# Data Handling
###################
class DataLoader:
    """Handles all data loading operations (images and ground truth)"""
    
    SUPPORTED_FORMATS = {
        'jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp', 'gif', 'webp'
    }
    
    @staticmethod
    def get_file_extension(file) -> str:
        return file.name.split('.')[-1].lower()

    @classmethod
    def load_image(cls, image_file) -> Optional[np.ndarray]:
        if image_file is not None:
            try:
                file_ext = cls.get_file_extension(image_file)
                
                if file_ext not in cls.SUPPORTED_FORMATS:
                    st.error(f"Unsupported file format: {file_ext}")
                    st.info(f"Supported formats: {', '.join(cls.SUPPORTED_FORMATS)}")
                    return None
                
                image_bytes = image_file.getvalue()
                
                if file_ext in ['tif', 'tiff']:
                    try:
                        img = Image.open(io.BytesIO(image_bytes))
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        image = np.array(img)
                    except Exception as e:
                        st.error(f"Error loading TIFF file: {str(e)}")
                        return None
                else:
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError("Failed to decode image")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                if not isinstance(image, np.ndarray):
                    raise ValueError("Failed to convert image to numpy array")
                
                if len(image.shape) != 3:
                    st.warning("Image is not in RGB format. Converting...")
                    if len(image.shape) == 2:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    else:
                        raise ValueError("Invalid image dimensions")
                
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
    def load_csv(txt_file, image) -> Optional[np.ndarray]:
        if txt_file is not None:
            try:
                content = txt_file.getvalue().decode('utf-8')
                
                if image is None:
                    st.error("Please upload an image first to load ground truth.")
                    return None

                image_height, image_width = image.shape[:2]
                
                boxes = []
                for line in content.splitlines():
                    if line.strip():
                        class_id, x, y, w, h = map(float, line.strip().split())
                        
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
    def __init__(self, 
                 model_path: str, 
                 scaler_path: str,
                 window_size: Tuple[int, int] = (64, 64),
                 step_size: int = 10,
                 downscale: float = 1.5,
                 threshold: float = 2.0,
                 nms_threshold: float = 0.008) -> None:
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
        for y in range(0, image.shape[0], self.step_size):
            for x in range(0, image.shape[1], self.step_size):
                window = image[y:y + self.window_size[1], x:x + self.window_size[0]]
                if window.shape[:2] == self.window_size:
                    yield (x, y, window)

    def detect(self, image: np.ndarray) -> np.ndarray:
        detections = []
        scale = 0
        
        for resized in pyramid_gaussian(image, downscale=self.downscale, channel_axis=-1):
            for (x, y, window) in self.sliding_window(resized):
                window = color.rgb2gray(window)
                
                features = hog(window, **self.hog_params)
                features = self.scaler.transform(features.reshape(1, -1))
                
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
            
            if resized.shape[0] < self.window_size[1] or resized.shape[1] < self.window_size[0]:
                break
                
        if detections:
            detections = np.array(detections)
            scores = np.ones(len(detections))
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
        try:
            self.models['YOLOv11n'] = YOLO('yolov11n.pt')

            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=8)
            model.eval()
            self.models['Faster R-CNN'] = model
            model.load_state_dict(torch.load('frcnn.pth', map_location=torch.device('cpu')))
        
        except Exception as e:
            st.error(f"Error loading YOLOv11n or Faster R-CNN models: {str(e)}")

        try:
            self.models['HOG_SVM'] = SimpleDetector(
                model_path='model.npy',
                scaler_path='scalar.npy'
            )
        except Exception as e:
            st.error(f"Error loading HOG_SVM model: {str(e)}")

    def predict(self, image: np.ndarray, model_type: str, conf: float, iou: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            if model_type == "YOLOv11n":
                results = self.models[model_type].predict(image, conf=conf, iou=iou)[0]
                boxes = results.boxes.xyxy.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy()
                return boxes, classes

            elif model_type == "Faster R-CNN":
                img_tensor = torch.tensor(image).permute(2, 0, 1).float()
                img_tensor = img_tensor / 255.0
                img_tensor = [img_tensor]

                with torch.no_grad():
                    predictions = self.models[model_type](img_tensor)[0]

                scores = predictions['scores']
                boxes = predictions['boxes']

                conf_mask = scores >= conf
                boxes = boxes[conf_mask]
                scores = scores[conf_mask]

                keep_indices = nms(boxes, scores, iou)
                final_boxes = boxes[keep_indices]

                return final_boxes.cpu().numpy().astype(int), None

            elif model_type == "HOG_SVM":
                if image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                elif image.shape[2] == 1:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                boxes = self.models[model_type].detect(image)
                return boxes, None
        except Exception as e:
            st.error(f"Error making predictions with {model_type}: {str(e)}")
            return None, None

###################
# Visualization
###################
class Visualizer:
    @staticmethod
    def plot_predictions(
        image: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        predictions: Optional[np.ndarray] = None,
        classes: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        highlighted_index: Optional[int] = None
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(image)
        
        if title:
            ax.set_title(title, fontsize=16)

        if ground_truth is not None:
            for box in ground_truth:
                x1, y1, x2, y2 = box
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, color='green', linewidth=2,
                                   label='Ground Truth')
                ax.add_patch(rect)
        
        if predictions is not None:
            class_colors = {0: 'red', 1: 'blue'}
            class_labels = {0: 'Class 0', 1: 'Class 1'}

            for i, box in enumerate(predictions):
                x1, y1, x2, y2 = box
                
                # Check if this box should be highlighted
                if i == highlighted_index:
                    color = 'blue'
                    linewidth = 3
                    label = 'Selected'
                else:
                    linewidth = 1
                    color = 'red'
                    label = 'Prediction'
                    if classes is not None:
                        class_id = int(classes[i])
                        color = class_colors.get(class_id, 'magenta')
                        label = class_labels.get(class_id, f'Class {class_id}')

                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, color=color, linewidth=linewidth,
                                   label=label)
                ax.add_patch(rect)
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        ax.axis('off')
        plt.tight_layout()
        return fig

###################
# UI Components
###################
class UIComponents:
    @staticmethod
    def create_upload_section(key_suffix="") -> Tuple:
        st.header("Upload Data")
        col1, col2 = st.columns(2)
        with col1:
            image_file = st.file_uploader(
                "Upload Image",
                type=list(DataLoader.SUPPORTED_FORMATS),
                key=f"image_uploader_{key_suffix}"
            )
        with col2:
            txt_file = st.file_uploader(
                "Upload Ground Truth (YOLO format)", 
                type=['txt'],
                key=f"txt_uploader_{key_suffix}"
            )
        return image_file, txt_file

    @staticmethod
    def show_instructions():
        st.info("Please upload an image to begin.")
        st.markdown("""
        ### Instructions:
        1. Select a tab: 'Single Model Prediction' or 'Compare Models'.
        2. Use the sidebar to select models and adjust thresholds.
        3. Upload an image for detection.
        4. Optionally, upload a ground truth file for comparison.
        5. Predictions will be generated automatically.
        6. Click a row in the results table to highlight the detection on the image.
        """)
    
    @staticmethod
    def display_results_table(predictions, classes, model_type, key):
        if len(predictions) > 0:
            df_data = {'x1': predictions[:, 0], 'y1': predictions[:, 1], 'x2': predictions[:, 2], 'y2': predictions[:, 3]}
            if classes is not None:
                df_data['class'] = classes
            df = pd.DataFrame(df_data)
            df['Area'] = (df['x2'] - df['x1']) * (df['y2'] - df['y1'])
        else:
            columns = ['x1', 'y1', 'x2', 'y2', 'Area']
            if model_type == 'YOLOv11n':
                columns.insert(4, 'class')
            df = pd.DataFrame(columns=columns)

        st.write(f"Detected objects: {len(df)}")
        st.dataframe(df, key=key, on_select="rerun", selection_mode="single-row")
        return df

###################
# Main Application
###################
class InsectDetectionApp:
    def __init__(self):
        self.data_loader = DataLoader()
        self.model_manager = ModelManager()
        self.visualizer = Visualizer()
        self.ui = UIComponents()

    def run_single_model_tab(self, settings):
        image_file, txt_file = self.ui.create_upload_section("single")
        
        if image_file:
            image = self.data_loader.load_image(image_file)
            if image is not None:
                ground_truth = self.data_loader.load_csv(txt_file, image)
                
                with st.spinner("Processing..."):
                    predictions, classes = self.model_manager.predict(
                        image,
                        settings['model_type'],
                        settings['conf_threshold'],
                        settings['iou_threshold']
                    )
                    
                    if predictions is not None:
                        st.subheader("Detection Results")
                        
                        # Get selected row from session state
                        df_key = "df_single"
                        selection = st.session_state.get(df_key, {}).get("selection", {}).get("rows", [])
                        highlighted_index = selection[0] if selection else None

                        # Display the plot in a placeholder
                        plot_placeholder = st.empty()
                        fig = self.visualizer.plot_predictions(
                            image, ground_truth, predictions, classes, 
                            title=settings['model_type'], 
                            highlighted_index=highlighted_index
                        )
                        plot_placeholder.pyplot(fig)

                        # Display the interactive dataframe
                        df = self.ui.display_results_table(predictions, classes, settings['model_type'], key=df_key)

                        # Add download button
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df.to_excel(writer, index=False, sheet_name='Detections')
                        excel_data = output.getvalue()

                        st.download_button(
                            label="📥 Download Excel File",
                            data=excel_data,
                            file_name=f"{settings['model_type']}_detections.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
        else:
            self.ui.show_instructions()

    def run_compare_models_tab(self, settings):
        image_file, txt_file = self.ui.create_upload_section("compare")
        
        if not settings['models_to_compare']:
            st.warning("Please select at least one model from the sidebar to compare.")
            return

        if image_file:
            image = self.data_loader.load_image(image_file)
            if image is not None:
                ground_truth = self.data_loader.load_csv(txt_file, image)
                
                st.subheader("Model Comparison")
                
                num_models = len(settings['models_to_compare'])
                cols = st.columns(num_models)

                with st.spinner("Running all selected models..."):
                    for i, model_name in enumerate(settings['models_to_compare']):
                        with cols[i]:
                            st.markdown(f"#### {model_name}")
                            predictions, classes = self.model_manager.predict(
                                image,
                                model_name,
                                settings['conf_threshold'],
                                settings['iou_threshold']
                            )

                            if predictions is not None:
                                # Get selected row for this specific dataframe
                                df_key = f"df_compare_{model_name}"
                                selection = st.session_state.get(df_key, {}).get("selection", {}).get("rows", [])
                                highlighted_index = selection[0] if selection else None

                                # Display plot in a placeholder
                                plot_placeholder = st.empty()
                                fig = self.visualizer.plot_predictions(
                                    image, ground_truth, predictions, classes, 
                                    title=f"{model_name} Predictions",
                                    highlighted_index=highlighted_index
                                )
                                plot_placeholder.pyplot(fig)
                                
                                # Display the interactive dataframe
                                self.ui.display_results_table(predictions, classes, model_name, key=df_key)
                            else:
                                st.error(f"Failed to get predictions from {model_name}.")
        else:
            self.ui.show_instructions()


    def run(self):
        st.title("Insect Detection Model Interface v2")
        
        # --- Create a single, unified sidebar for the whole app ---
        st.sidebar.header("Model Settings")
        model_options = ["YOLOv11n", "Faster R-CNN", "HOG_SVM"]

        # Settings for Single Model Tab
        with st.sidebar.expander("Single Model Settings", expanded=True):
            single_model_type = st.selectbox(
                "Select Model", model_options, key="single_model_select"
            )

        # Settings for Compare Models Tab
        with st.sidebar.expander("Compare Models Settings", expanded=True):
            models_to_compare = st.multiselect(
                "Select Models to Compare", model_options, default=model_options[:2], key="compare_model_select"
            )

        # Common settings for both tabs
        conf_threshold = st.sidebar.slider(
            "Confidence Threshold", 0.0, 1.0, 0.5, 0.05, key="conf_thresh"
        )
        iou_threshold = st.sidebar.slider(
            "IOU Threshold", 0.0, 1.0, 0.5, 0.05, key="iou_thresh"
        )

        # Package settings into dictionaries to pass to the tabs
        settings_single = {
            'model_type': single_model_type,
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold
        }
        settings_compare = {
            'models_to_compare': models_to_compare,
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold
        }
        # --- End of sidebar creation ---

        tab1, tab2 = st.tabs(["Single Model Prediction", "Compare Models"])

        with tab1:
            self.run_single_model_tab(settings_single)

        with tab2:
            self.run_compare_models_tab(settings_compare)


if __name__ == "__main__":
    app = InsectDetectionApp()
    app.run()
