import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import io
from typing import Optional, Tuple, Generator
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
import joblib
from skimage import color
from skimage.feature import hog
from skimage.transform import pyramid_gaussian, resize
from imutils.object_detection import non_max_suppression
import os
from pkg_resources import parse_version
import base64
import json
import streamlit.components.v1 as components
import tempfile

# Make page wider so canvases get more space
st.set_page_config(layout="wide")

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
                 nms_threshold: float = 0.15) -> None:
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.window_size = window_size
        self.step_size = step_size
        self.downscale = downscale
        self.threshold = threshold
        self.nms_threshold = nms_threshold
        
        # Use common HOG params that produce 1764 features for 64x64 windows
        self.hog_params = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'block_norm': 'L2-Hys',   # length unaffected by this choice
            'feature_vector': True
        }

        # Infer expected feature length from scaler (if available)
        self.expected_features = None
        try:
            if hasattr(self.scaler, 'n_features_in_'):
                self.expected_features = int(self.scaler.n_features_in_)
            elif hasattr(self.scaler, 'mean_'):
                self.expected_features = int(self.scaler.mean_.shape[0])
        except Exception:
            pass

    def sliding_window(self, image: np.ndarray) -> Generator[Tuple[int, int, np.ndarray], None, None]:
        """Slide a window across the image and yield (x, y, window) with the configured window size."""
        W, H = self.window_size
        for y in range(0, max(1, image.shape[0] - H + 1), self.step_size):
            for x in range(0, max(1, image.shape[1] - W + 1), self.step_size):
                yield (x, y, image[y:y + H, x:x + W])
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        detections = []
        scale = 0
        
        # Ensure RGB float in [0,1] for skimage
        if image.dtype != np.float32 and image.dtype != np.float64:
            img_f = image.astype(np.float32) / 255.0
        else:
            img_f = image

        for resized in pyramid_gaussian(img_f, downscale=self.downscale, channel_axis=-1):
            for (x, y, window) in self.sliding_window(resized):
                # Force window to canonical HxW (e.g., 64x64) to match training
                W, H = self.window_size
                if window.shape[0] != H or window.shape[1] != W:
                    # Skip if window not exact size (shouldn't happen, but safe)
                    continue

                # Convert to grayscale and ensure exact 64x64 via resize (guards against subtle rounding)
                gray = color.rgb2gray(window)
                if gray.shape != (H, W):
                    gray = resize(gray, (H, W), mode='reflect', anti_aliasing=True)

                features = hog(gray, **self.hog_params)

                # Validate feature length against scaler expectations
                if self.expected_features is not None and features.shape[0] != self.expected_features:
                    # Try to force-resize once more if mismatch (defensive)
                    gray_fixed = resize(gray, (64, 64), mode='reflect', anti_aliasing=True)
                    features = hog(gray_fixed, **self.hog_params)
                    if features.shape[0] != self.expected_features:
                        # If still mismatched, skip this window
                        continue

                features_scaled = self.scaler.transform(features.reshape(1, -1))
                
                pred = int(self.model.predict(features_scaled)[0])
                if pred == 1:
                    score = float(self.model.decision_function(features_scaled)[0])
                    if score > self.threshold:
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
            # Use dummy scores (all 1s) for NMS since we don't have per-window prob
            scores = np.ones(len(detections))
            # imutils non_max_suppression expects float; returns boxes of same dtype
            pick = non_max_suppression(detections.astype(float), probs=scores, overlapThresh=self.nms_threshold)
            return pick.astype(int)
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

            # Load Faster R-CNN with modern API, then replace predictor for your classes
            try:
                model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            except Exception:
                # Backward-compatible fallback
                model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=8)
            # Load your trained weights if present
            if os.path.exists('frcnn.pth'):
                model.load_state_dict(torch.load('frcnn.pth', map_location=torch.device('cpu')))
            else:
                st.warning("Faster R-CNN weights file 'frcnn.pth' was not found; using base weights.")
            model.eval()
            self.models['Faster R-CNN'] = model
        
        except Exception as e:
            st.error(f"Error loading YOLOv11n or Faster R-CNN models: {str(e)}")

        try:
            self.models['HOG_SVM'] = SimpleDetector(
                model_path='model.npy',
                scaler_path='scalar.npy'
            )
        except Exception as e:
            st.error(f"Error loading HOG_SVM model: {str(e)}")

    def predict(self, image: np.ndarray, model_type: str, conf: float, iou: float):
        try:
            if model_type == "YOLOv11n":
                results = self.models[model_type].predict(image, conf=conf, iou=iou, verbose=False)[0]
                boxes = results.boxes.xyxy.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy().astype(int)
                confidences = results.boxes.conf.cpu().numpy()
                return boxes, classes, confidences

            elif model_type == "Faster R-CNN":
                img_tensor = torch.as_tensor(image).permute(2, 0, 1).float() / 255.0
                img_tensor = [img_tensor]

                with torch.no_grad():
                    predictions = self.models[model_type](img_tensor)[0]

                scores = predictions['scores']
                boxes = predictions['boxes']

                if boxes.numel() == 0:
                    return np.empty((0, 4)), None, np.array([])

                conf_mask = scores >= conf
                boxes = boxes[conf_mask]
                scores = scores[conf_mask]

                if boxes.numel() == 0:
                    return np.empty((0, 4)), None, np.array([])

                keep_indices = nms(boxes, scores, iou)
                final_boxes = boxes[keep_indices]
                final_scores = scores[keep_indices]

                return final_boxes.cpu().numpy(), None, final_scores.cpu().numpy()

            elif model_type == "HOG_SVM":
                # Normalize channels
                if image.ndim == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

                boxes = self.models[model_type].detect(image)
                confidences = np.ones(len(boxes))  # HOG doesn't provide confidence scores
                return boxes, None, confidences

        except Exception as e:
            st.error(f"Error making predictions with {model_type}: {str(e)}")
            return None, None, None

###################
# Interactive Canvas Component
###################
def create_interactive_canvas(image_data, predictions, classes, confidences, key, target_display_width=900):
    # Convert image to base64
    _, buffer = cv2.imencode('.png', cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR))
    img_str = base64.b64encode(buffer).decode()

    # Convert predictions to list of dicts for JS (use JSON to avoid Python None in JS)
    boxes_data = []
    if predictions is not None and len(predictions) > 0:
        for i, box in enumerate(predictions):
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            area = w * h
            cls_val = int(classes[i]) if classes is not None else None
            conf_val = float(confidences[i]) if confidences is not None and len(confidences) > i else 1.0
            boxes_data.append({
                "row": int(i),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "w": w, "h": h, "area": area,
                "cls": cls_val,
                "confidence": conf_val
            })

    boxes_json = json.dumps(boxes_data)

    # Compute a reasonable iframe height to match expected on-page width
    # We cannot know column width from Python, so we approximate via target_display_width
    h = image_data.shape[0]
    w = image_data.shape[1]
    aspect_h = max(1, int(h * (target_display_width / max(1, w))))
    extra_panel_px = 70  # compact info panel height
    iframe_height = aspect_h + extra_panel_px

    components.html(
        f"""
        <div style="position: relative; width: 100%; margin: 0;">
            <canvas id="canvas-{key}" style="width: 100%; height: auto; border: 1px solid #ddd; cursor: pointer; display: block;"></canvas>
            <div id="tooltip-{key}" style="position: fixed; background: rgba(0,0,0,0.75); color: white; padding: 5px 8px; border-radius: 4px; pointer-events: none; display: none; z-index: 1000; font-size: 12px;"></div>
            <div id="selected-info-{key}" style="margin-top: 6px; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; font-size: 13px; background: #fafafa; border: 1px solid #eee; border-radius: 6px; padding: 8px;">
                <em>No box selected</em>
            </div>
        </div>
        <script>
            const canvas = document.getElementById('canvas-{key}');
            const tooltip = document.getElementById('tooltip-{key}');
            const info = document.getElementById('selected-info-{key}');
            const ctx = canvas.getContext('2d');
            const img = new Image();
            img.src = "data:image/png;base64,{img_str}";
            const boxes = {boxes_json};

            let selectedBoxIndex = -1;
            let hoveredBoxIndex = -1;
            let scaleX = 1, scaleY = 1;

            img.onload = () => {{
                canvas.width = img.naturalWidth;
                canvas.height = img.naturalHeight;
                resizeCanvas();
                window.addEventListener('resize', resizeCanvas);
                renderSelectedInfo();
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
                        ctx.strokeStyle = '#2d6cdf';
                        ctx.lineWidth = 3.5;
                    }} else if (index === hoveredBoxIndex) {{
                        ctx.strokeStyle = '#f0ad4e';
                        ctx.lineWidth = 3;
                    }} else {{
                        // simple color coding by class presence
                        ctx.strokeStyle = (box.cls === 1) ? '#00ffff' : '#ff4d4f';
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

            function renderSelectedInfo() {{
                if (selectedBoxIndex < 0 || selectedBoxIndex >= boxes.length) {{
                    info.innerHTML = '<em>No box selected</em>';
                    return;
                }}
                const b = boxes[selectedBoxIndex];
                const clsStr = (b.cls === null || b.cls === undefined) ? '' : `, class=${{b.cls}}`;
                info.innerHTML = `
                    <b>Selected detection (row ${{b.row}})</b><br/>
                    x1=${{b.x1.toFixed(0)}}, y1=${{b.y1.toFixed(0)}}, x2=${{b.x2.toFixed(0)}}, y2=${{b.y2.toFixed(0)}}<br/>
                    w=${{b.w.toFixed(0)}}, h=${{b.h.toFixed(0)}}, area=${{b.area.toFixed(0)}}<br/>
                    conf=${{(b.confidence ?? 1).toFixed(2)}}${{clsStr}}
                `;
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
                        tooltip.innerHTML = `Row: ${{i}}, Area: ${{Math.round(box.area)}}, Conf: ${{(box.confidence ?? 1).toFixed(2)}}`;
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
                renderSelectedInfo();
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
        - Use the sidebar to select models, thresholds, or upload custom YOLO weights.
        - Hover a box to see row/area/confidence.
        - Click a box to highlight it and see its details under the image.
        """)

    @staticmethod
    def display_results_table(predictions, classes, confidences, model_type):
        if predictions is not None and len(predictions) > 0:
            preds = np.asarray(predictions)
            x1, y1, x2, y2 = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
            w = np.maximum(0, x2 - x1)
            h = np.maximum(0, y2 - y1)
            area = w * h
            df_data = {
                'x1': x1, 'y1': y1, 
                'x2': x2, 'y2': y2,
                'Area': area,
            }
            if confidences is not None:
                df_data['confidence'] = confidences
            if classes is not None:
                df_data['class'] = classes
            df = pd.DataFrame(df_data)
        else:
            columns = ['x1', 'y1', 'x2', 'y2', 'Area', 'confidence']
            if model_type == 'YOLOv11n':
                columns.insert(5, 'class')
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
                    
                    st.subheader("Interactive Detection Results")
                    create_interactive_canvas(image, predictions, classes, confidences, key="single_canvas", target_display_width=1000)
                    total = 0 if predictions is None else len(predictions)
                    st.caption(f"Total detections: {total}")
                    
                    df = self.ui.display_results_table(predictions, classes, confidences, settings['model_type'])
                    st.markdown("#### All Detections:")
                    st.dataframe(df, use_container_width=True)
                    
                    # Prepare Excel file without confidence column if needed
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

                # Use 2 columns for better visibility if >1 model
                n_models = len(settings['models_to_compare'])
                n_cols = 1 if n_models == 1 else 2
                cols = st.columns(n_cols)

                with st.spinner("Running all selected models..."):
                    for i, model_name in enumerate(settings['models_to_compare']):
                        col = cols[i % n_cols]
                        with col:
                            st.markdown(f"#### {model_name}")
                            predictions, classes, confidences = self.model_manager.predict(
                                image,
                                model_name,
                                settings['conf_threshold'],
                                settings['iou_threshold']
                            )

                            # Larger width in 2-col layout
                            target_w = 1000 if n_cols == 1 else 580
                            create_interactive_canvas(image, predictions, classes, confidences, key=f"compare_{model_name}", target_display_width=target_w)
                            total = 0 if predictions is None else len(predictions)
                            st.caption(f"Total detections: {total}")

                            st.markdown("Detections:")
                            df = self.ui.display_results_table(predictions, classes, confidences, model_name)
                            st.dataframe(df, use_container_width=True)
        else:
            self.ui.show_instructions()


    def run(self):
        st.title("Insect Detection Model Interface v3")
        
        st.sidebar.header("Model Settings")
        
        with st.sidebar.expander("Upload Custom YOLO Weights", expanded=False):
            uploaded_weights = st.file_uploader("Upload YOLO Weights (.pt)", type=['pt'])
            if uploaded_weights is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
                    tmp.write(uploaded_weights.getvalue())
                    st.session_state['custom_yolo_path'] = tmp.name
                st.success(f"Using custom weights: {uploaded_weights.name}")
                if st.button("Apply Custom Weights"):
                    st.rerun()
            else:
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