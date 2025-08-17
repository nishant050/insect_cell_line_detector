import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
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
import base64
import streamlit.components.v1 as components
import tempfile

# Version check for ultralytics
try:
    from ultralytics import YOLO
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
                
                return image
                
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
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
            # Check for custom YOLO weights
            yolo_path = st.session_state.get('custom_yolo_path', 'yolov11n.pt')
            if not os.path.exists(yolo_path):
                st.warning(f"Weights file not found at {yolo_path}. Falling back to default 'yolov11n.pt'.")
                yolo_path = 'yolov11n.pt'
            
            self.models['YOLOv11n'] = YOLO(yolo_path)

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

    def predict(self, image: np.ndarray, model_type: str, conf: float, iou: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            if model_type == "YOLOv11n":
                results = self.models[model_type].predict(image, conf=conf, iou=iou)[0]
                boxes = results.boxes.xyxy.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy()
                confidences = results.boxes.conf.cpu().numpy()
                return boxes, classes, confidences

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
                final_scores = scores[keep_indices]

                return final_boxes.cpu().numpy().astype(int), None, final_scores.cpu().numpy()

            elif model_type == "HOG_SVM":
                if image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                elif image.shape[2] == 1:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                boxes = self.models[model_type].detect(image)
                confidences = np.ones(len(boxes)) # HOG doesn't provide confidence scores
                return boxes, None, confidences
        except Exception as e:
            st.error(f"Error making predictions with {model_type}: {str(e)}")
            return None, None, None

###################
# Interactive Canvas Component
###################
def create_interactive_canvas(image_data, predictions, classes, confidences, key):
    # Convert image to base64
    _, buffer = cv2.imencode('.png', cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR))
    img_str = base64.b64encode(buffer).decode()

    # Convert predictions to list of dicts
    boxes_data = []
    if predictions is not None:
        for i, box in enumerate(predictions):
            area = (box[2] - box[0]) * (box[3] - box[1])
            box_info = {
                "x1": int(box[0]), "y1": int(box[1]),
                "x2": int(box[2]), "y2": int(box[3]),
                "class": int(classes[i]) if classes is not None else 0,
                "confidence": float(confidences[i]) if confidences is not None else 1.0,
                "area": int(area)
            }
            boxes_data.append(box_info)

    # ---- FIX: compute iframe height dynamically to match image aspect ratio ----
    max_display_width = 800   # adjust to your Streamlit layout (try 800–1000)
    iframe_height = int(image_data.shape[0] * (max_display_width / image_data.shape[1]))

    components.html(
        f"""
        <div style="position: relative; width: 100%; margin: auto;">
            <canvas id="canvas-{key}" style="width: 100%; height: auto; border: 1px solid black; cursor: pointer; display: block;"></canvas>
            <div id="tooltip-{key}" style="position: fixed; background: rgba(0,0,0,0.7); color: white; padding: 5px; border-radius: 3px; pointer-events: none; display: none; z-index: 1000;"></div>
        </div>
        <script>
            const canvas = document.getElementById('canvas-{key}');
            const tooltip = document.getElementById('tooltip-{key}');
            const ctx = canvas.getContext('2d');
            const img = new Image();
            img.src = "data:image/png;base64,{img_str}";

            const boxes = {boxes_data};
            let selectedBoxIndex = -1;
            let hoveredBoxIndex = -1;
            let scaleX = 1, scaleY = 1;

            img.onload = () => {{
                canvas.width = img.naturalWidth;
                canvas.height = img.naturalHeight;
                resizeCanvas();
                window.addEventListener('resize', resizeCanvas);
            }};

            function resizeCanvas() {{
                const rect = canvas.getBoundingClientRect();
                scaleX = canvas.width / rect.width;
                scaleY = canvas.height / rect.height;
                draw();
            }}

            function draw() {{
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

                boxes.forEach((box, index) => {{
                    ctx.beginPath();
                    ctx.rect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
                    if (index === selectedBoxIndex) {{
                        ctx.strokeStyle = 'blue';
                        ctx.lineWidth = 4;
                    }} else if (index === hoveredBoxIndex) {{
                        ctx.strokeStyle = 'yellow';
                        ctx.lineWidth = 3;
                    }} else {{
                        ctx.strokeStyle = box.class === 1 ? 'cyan' : 'red';
                        ctx.lineWidth = 2;
                    }}
                    ctx.stroke();
                }});
            }}

            function getMousePos(canvas, evt) {{
                const rect = canvas.getBoundingClientRect();
                return {{
                    x: (evt.clientX - rect.left) * scaleX,
                    y: (evt.clientY - rect.top) * scaleY
                }};
            }}

            canvas.addEventListener('mousemove', (e) => {{
                const pos = getMousePos(canvas, e);
                let foundBox = false;
                hoveredBoxIndex = -1;
                for (let i = boxes.length - 1; i >= 0; i--) {{
                    const box = boxes[i];
                    if (pos.x >= box.x1 && pos.x <= box.x2 && pos.y >= box.y1 && pos.y <= box.y2) {{
                        hoveredBoxIndex = i;
                        tooltip.style.display = 'block';
                        tooltip.style.left = `${{e.clientX + 15}}px`;
                        tooltip.style.top = `${{e.clientY}}px`;
                        tooltip.innerHTML = `Row: ${{i}}, Area: ${{box.area}}, Conf: ${{box.confidence.toFixed(2)}}`;
                        foundBox = true;
                        break;
                    }}
                }}
                if (!foundBox) {{
                    tooltip.style.display = 'none';
                }}
                draw();
            }});

            canvas.addEventListener('click', (e) => {{
                const pos = getMousePos(canvas, e);
                let newSelectedBoxIndex = -1;
                for (let i = boxes.length - 1; i >= 0; i--) {{
                    const box = boxes[i];
                    if (pos.x >= box.x1 && pos.x <= box.x2 && pos.y >= box.y1 && pos.y <= box.y2) {{
                        newSelectedBoxIndex = i;
                        break;
                    }}
                }}
                selectedBoxIndex = (newSelectedBoxIndex === selectedBoxIndex) ? -1 : newSelectedBoxIndex;
                draw();
            }});
        </script>
        """,
        height=iframe_height,
        scrolling=False
    )


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
        2. Use the sidebar to select models, adjust thresholds, or upload custom weights.
        3. Upload an image for detection.
        4. **Interact with the image**:
           - **Hover** over a box to see its row number, area, and confidence.
           - **Click** a box to highlight it.
        """)
    
    @staticmethod
    def display_results_table(predictions, classes, confidences, model_type):
        if predictions is not None and len(predictions) > 0:
            df_data = {
                'x1': predictions[:, 0], 'y1': predictions[:, 1], 
                'x2': predictions[:, 2], 'y2': predictions[:, 3],
                'confidence': confidences
            }
            if classes is not None:
                df_data['class'] = classes
            df = pd.DataFrame(df_data)
            df['Area'] = (df['x2'] - df['x1']) * (df['y2'] - df['y1'])
        else:
            columns = ['x1', 'y1', 'x2', 'y2', 'Area', 'confidence']
            if model_type == 'YOLOv11n':
                columns.insert(4, 'class')
            df = pd.DataFrame(columns=columns)
        return df

###################
# Main Application
###################
class InsectDetectionApp:
    def __init__(self):
        self.data_loader = DataLoader()
        self.model_manager = ModelManager()
        self.ui = UIComponents()

    def run_single_model_tab(self, settings):
        image_file, txt_file = self.ui.create_upload_section("single")
        
        if image_file:
            image = self.data_loader.load_image(image_file)
            if image is not None:
                with st.spinner("Processing..."):
                    predictions, classes, confidences = self.model_manager.predict(
                        image,
                        settings['model_type'],
                        settings['conf_threshold'],
                        settings['iou_threshold']
                    )
                    
                    if predictions is not None:
                        st.subheader("Interactive Detection Results")
                        
                        create_interactive_canvas(image, predictions, classes, confidences, key="single_canvas")
                        
                        st.write("") # Add space
                        
                        df = self.ui.display_results_table(predictions, classes, confidences, settings['model_type'])
                        
                        st.write("#### All Detections:")
                        st.dataframe(df)
                        
                        # Prepare Excel file without confidence column
                        df_for_download = df.drop(columns=['confidence'], errors='ignore')
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df_for_download.to_excel(writer, index=False, sheet_name='Detections')
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
                st.subheader("Model Comparison")
                
                num_models = len(settings['models_to_compare'])
                cols = st.columns(num_models)

                with st.spinner("Running all selected models..."):
                    for i, model_name in enumerate(settings['models_to_compare']):
                        with cols[i]:
                            st.markdown(f"#### {model_name}")
                            predictions, classes, confidences = self.model_manager.predict(
                                image,
                                model_name,
                                settings['conf_threshold'],
                                settings['iou_threshold']
                            )

                            if predictions is not None:
                                create_interactive_canvas(image, predictions, classes, confidences, key=f"compare_{model_name}")
                                
                                st.write("") # Add space

                                st.write("Detections:")
                                df = self.ui.display_results_table(predictions, classes, confidences, model_name)
                                st.dataframe(df)
                                
                            else:
                                st.error(f"Failed to get predictions from {model_name}.")
        else:
            self.ui.show_instructions()


    def run(self):
        st.title("Insect Detection Model Interface v3")
        
        st.sidebar.header("Model Settings")
        
        with st.sidebar.expander("Upload Custom Weights", expanded=False):
            uploaded_weights = st.file_uploader("Upload YOLO Weights (.pt)", type=['pt'])
            if uploaded_weights is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
                    tmp.write(uploaded_weights.getvalue())
                    st.session_state['custom_yolo_path'] = tmp.name
                st.success(f"Using custom weights: {uploaded_weights.name}")
                # Add a button to trigger rerun and model reload
                if st.button("Apply Custom Weights"):
                    st.rerun()
            else:
                # If no file is uploaded, ensure we use the default
                if 'custom_yolo_path' in st.session_state:
                    del st.session_state['custom_yolo_path']

        model_options = ["YOLOv11n", "Faster R-CNN", "HOG_SVM"]

        with st.sidebar.expander("Single Model Settings", expanded=True):
            single_model_type = st.selectbox(
                "Select Model", model_options, key="single_model_select"
            )

        with st.sidebar.expander("Compare Models Settings", expanded=True):
            models_to_compare = st.multiselect(
                "Select Models to Compare", model_options, default=model_options[:2], key="compare_model_select"
            )

        conf_threshold = st.sidebar.slider(
            "Confidence Threshold", 0.0, 1.0, 0.5, 0.05, key="conf_thresh"
        )
        iou_threshold = st.sidebar.slider(
            "IOU Threshold", 0.0, 1.0, 0.5, 0.05, key="iou_thresh"
        )

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

        tab1, tab2 = st.tabs(["Single Model Prediction", "Compare Models"])

        with tab1:
            self.run_single_model_tab(settings_single)

        with tab2:
            self.run_compare_models_tab(settings_compare)


if __name__ == "__main__":
    app = InsectDetectionApp()
    app.run()
