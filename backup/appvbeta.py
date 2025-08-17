import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import io
from typing import Optional, Tuple, List, Dict
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
import json

# =========================
# Global Page Config & CSS
# =========================
st.set_page_config(page_title="Insect Detection – Pro", layout="wide")
st.markdown(
    """
    <style>
      /* Clean, modern look */
      .block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1400px;}
      .stTabs [data-baseweb="tab-list"] {gap: 0.5rem;}
      .stDataFrame, .stTable {border-radius: 16px; overflow: hidden;}
      .metric-card {border-radius: 16px; padding: 10px 14px; background: rgba(0,0,0,0.03); border: 1px solid rgba(0,0,0,0.06);}
      .small-muted {color: #6b7280; font-size: 0.85rem;}
      .tip {background: #eef6ff; border: 1px solid #cfe8ff; padding: 10px 12px; border-radius: 12px;}
      /* Ensure side-by-side layout has breathing room */
      .viewer-col {padding-right: 0.5rem;}
      .panel-col {padding-left: 0.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Version check for ultralytics
# =========================
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    import ultralytics
    YOLO_AVAILABLE = True
    if parse_version(ultralytics.__version__) < parse_version("8.0.0"):
        st.warning(
            f"Your `ultralytics` version is {ultralytics.__version__}, but this app is optimized for version 8.0.0 or higher. "
            "Please upgrade for optimal performance: `pip install --upgrade ultralytics`"
        )
except ImportError:
    st.info("`ultralytics` not installed. YOLOv11n will be unavailable. Install with `pip install ultralytics`.")

# =========================
# Data Handling
# =========================
class DataLoader:
    SUPPORTED_FORMATS = {'jpg','jpeg','png','tif','tiff','bmp','gif','webp'}

    @staticmethod
    def get_file_extension(file) -> str:
        return file.name.split('.')[-1].lower()

    @classmethod
    def load_image(cls, image_file) -> Optional[np.ndarray]:
        if image_file is None: return None
        try:
            file_ext = cls.get_file_extension(image_file)
            if file_ext not in cls.SUPPORTED_FORMATS:
                st.error(f"Unsupported file format: {file_ext}")
                st.info(f"Supported formats: {', '.join(sorted(cls.SUPPORTED_FORMATS))}")
                return None

            image_bytes = image_file.getvalue()
            if file_ext in ['tif','tiff']:
                img = Image.open(io.BytesIO(image_bytes))
                if img.mode != 'RGB': img = img.convert('RGB')
                image = np.array(img)
            else:
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Failed to decode image")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return None

    @staticmethod
    def load_csv(txt_file, image) -> Optional[np.ndarray]:
        if txt_file is None: return None
        try:
            content = txt_file.getvalue().decode('utf-8')
            if image is None:
                st.error("Upload an image before ground truth.")
                return None

            h, w = image.shape[:2]
            boxes = []
            for line in content.splitlines():
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) != 5: continue
                    class_id, x, y, ww, hh = map(float, parts)
                    x1 = int((x - ww/2) * w)
                    y1 = int((y - hh/2) * h)
                    x2 = int((x + ww/2) * w)
                    y2 = int((y + hh/2) * h)
                    boxes.append([x1, y1, x2, y2])
            return np.array(boxes) if boxes else None
        except Exception as e:
            st.error(f"Error loading ground truth file: {e}")
            return None

# =========================
# HOG + SVM Detector
# =========================
class SimpleDetector:
    def __init__(self, model_path: str, scaler_path: str,
                 window_size: Tuple[int,int]=(64,64), step_size:int=10,
                 downscale: float=1.5, threshold: float=2.0, nms_threshold: float=0.008) -> None:
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.window_size = window_size
        self.step_size = step_size
        self.downscale = downscale
        self.threshold = threshold
        self.nms_threshold = nms_threshold
        self.hog_params = dict(orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm='L2')

    def sliding_window(self, image):
        for y in range(0, image.shape[0] - self.window_size[1], self.step_size):
            for x in range(0, image.shape[1] - self.window_size[0], self.step_size):
                yield (x, y, image[y:y+self.window_size[1], x:x+self.window_size[0]])

    def detect(self, image: np.ndarray) -> np.ndarray:
        detections = []
        scale = 0
        for resized in pyramid_gaussian(image, downscale=self.downscale, channel_axis=-1):
            for (x, y, window) in self.sliding_window(resized):
                gray = color.rgb2gray(window)
                features = hog(gray, **self.hog_params)
                features = self.scaler.transform(features.reshape(1, -1))
                if self.model.predict(features) == 1:
                    if self.model.decision_function(features) > self.threshold:
                        s = self.downscale ** scale
                        detections.append([
                            int(x * s), int(y * s),
                            int((x + self.window_size[0]) * s),
                            int((y + self.window_size[1]) * s)
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

# =========================
# Model Management
# =========================
class ModelManager:
    def __init__(self):
        self.models: Dict[str, object] = {}
        self.available: Dict[str, bool] = {"YOLOv11n": False, "Faster R-CNN": False, "HOG_SVM": False}
        self.load_models()

    def load_models(self):
        # YOLO
        try:
            if YOLO_AVAILABLE:
                yolo_path = st.session_state.get('custom_yolo_path', 'yolov11n.pt')
                if not os.path.exists(yolo_path):
                    st.info(f"YOLO weights not found at {yolo_path}. Will try to use pretrained 'yolov11n.pt' if available locally.")
                    yolo_path = 'yolov11n.pt'
                self.models['YOLOv11n'] = YOLO(yolo_path)
                self.available['YOLOv11n'] = True
        except Exception as e:
            st.warning(f"YOLOv11n unavailable: {e}")

        # Faster R-CNN
        try:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=8)
            model.eval()
            if os.path.exists('frcnn.pth'):
                model.load_state_dict(torch.load('frcnn.pth', map_location=torch.device('cpu')))
                self.models['Faster R-CNN'] = model
                self.available['Faster R-CNN'] = True
            else:
                st.info("Missing 'frcnn.pth' – Faster R-CNN disabled.")
        except Exception as e:
            st.warning(f"Faster R-CNN unavailable: {e}")

        # HOG + SVM
        try:
            if os.path.exists('model.npy') and os.path.exists('scalar.npy'):
                self.models['HOG_SVM'] = SimpleDetector('model.npy', 'scalar.npy')
                self.available['HOG_SVM'] = True
            else:
                st.info("Missing 'model.npy'/'scalar.npy' – HOG_SVM disabled.")
        except Exception as e:
            st.warning(f"HOG_SVM unavailable: {e}")

    def predict(self, image: np.ndarray, model_type: str, conf: float, iou: float):
        try:
            if model_type == "YOLOv11n":
                if not self.available['YOLOv11n']: return None, None, None
                results = self.models[model_type].predict(image, conf=conf, iou=iou)[0]
                boxes = results.boxes.xyxy.cpu().numpy().astype(int) if results.boxes.xyxy is not None else np.empty((0,4),dtype=int)
                classes = results.boxes.cls.cpu().numpy().astype(int) if results.boxes.cls is not None else None
                confidences = results.boxes.conf.cpu().numpy() if results.boxes.conf is not None else None
                return boxes, classes, confidences

            elif model_type == "Faster R-CNN":
                if not self.available['Faster R-CNN']: return None, None, None
                img_tensor = torch.tensor(image).permute(2, 0, 1).float() / 255.0
                with torch.no_grad():
                    pred = self.models[model_type]([img_tensor])[0]
                scores = pred.get('scores', torch.tensor([]))
                boxes = pred.get('boxes', torch.empty((0,4)))
                conf_mask = scores >= conf
                boxes = boxes[conf_mask]
                scores = scores[conf_mask]
                if len(boxes) > 0:
                    keep = nms(boxes, scores, iou)
                    boxes = boxes[keep].cpu().numpy().astype(int)
                    scores = scores[keep].cpu().numpy()
                else:
                    boxes = np.empty((0,4), dtype=int)
                    scores = np.array([])
                return boxes, None, scores

            elif model_type == "HOG_SVM":
                if not self.available['HOG_SVM']: return None, None, None
                if image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                elif image.shape[2] == 1:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                boxes = self.models[model_type].detect(image)
                confidences = np.ones(len(boxes)) if len(boxes) else np.array([])
                return boxes.astype(int), None, confidences
        except Exception as e:
            st.error(f"Prediction error ({model_type}): {e}")
            return None, None, None

# =========================
# Utils
# =========================
def draw_boxes_on_image(image: np.ndarray, boxes: np.ndarray, classes, confidences, selected_idx: Optional[int]=None) -> np.ndarray:
    img = image.copy()
    for i, b in enumerate(boxes if boxes is not None and len(boxes) > 0 else []):
        x1,y1,x2,y2 = map(int, b)
        color = (0,255,255) if (classes is not None and i < len(classes) and int(classes[i])==1) else (255,0,0)
        if selected_idx is not None and i == selected_idx:
            color = (0,128,255)
            thickness = 2
        else:
            thickness = 2
        cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)
    return img

def df_from_predictions(predictions, classes, confidences, model_type: str) -> pd.DataFrame:
    if predictions is None or len(predictions) == 0:
        cols = ['x1','y1','x2','y2','area','confidence']
        if model_type == 'YOLOv11n': cols.insert(5, 'class')
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame({
        'x1': predictions[:,0], 'y1': predictions[:,1],
        'x2': predictions[:,2], 'y2': predictions[:,3],
    })
    df['area'] = (df['x2']-df['x1'])*(df['y2']-df['y1'])
    if confidences is not None:
        df['confidence'] = confidences
    else:
        df['confidence'] = np.nan
    if classes is not None:
        df['class'] = classes
    return df

def to_base64_png(img_rgb: np.ndarray) -> str:
    _, buffer = cv2.imencode('.png', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode()

# =========================
# Interactive Canvas Component (NO CUTOFF + Zoom/Pan)
# =========================
def interactive_canvas(image: np.ndarray, boxes, classes, confidences, key: str, selected_index: int = -1) -> int:
    """
    Returns the currently selected box index (or -1).
    This canvas ALWAYS shows the full image by computing canvas height from the container width
    and fixing iframe height to be comfortably larger than the canvas (no clipping).
    Zoom & pan supported (wheel + drag).
    """
    img_b64 = to_base64_png(image)
    bxs = []
    if boxes is not None:
        for i, b in enumerate(boxes):
            area = int((b[2]-b[0])*(b[3]-b[1]))
            bxs.append({
                "x1": int(b[0]), "y1": int(b[1]), "x2": int(b[2]), "y2": int(b[3]),
                "class": int(classes[i]) if classes is not None and i < len(classes) else -1,
                "confidence": float(confidences[i]) if confidences is not None and i < len(confidences) else 1.0,
                "area": area
            })

    # We use a predictable max content width so we can set an iframe height that won't crop.
    max_content_width = 1000  # px – matches container style below
    h, w = image.shape[0], image.shape[1]
    expected_canvas_height = int(h * (min(max_content_width, w) / w))
    iframe_height = max( expected_canvas_height + 80, 420 )  # padding for toolbars/tooltip

    html = f"""
    <div id="root-{key}" style="max-width:{max_content_width}px; width:100%; margin:0 auto;">
      <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
        <div class="small-muted">Scroll to zoom • drag to pan • click a box to select</div>
        <div>
          <button id="fit-{key}" style="padding:6px 10px;border-radius:10px;border:1px solid #ddd;cursor:pointer;">Fit</button>
          <button id="reset-{key}" style="padding:6px 10px;border-radius:10px;border:1px solid #ddd;cursor:pointer;margin-left:6px;">Reset</button>
        </div>
      </div>
      <div style="position:relative; width:100%; border:1px solid #ddd; border-radius:12px; overflow:hidden; background:#fff;">
        <canvas id="canvas-{key}" style="display:block; width:100%; height:auto; cursor:grab;"></canvas>
        <div id="tooltip-{key}" style="position:absolute; background: rgba(0,0,0,0.75); color:#fff; padding:6px 8px; border-radius:8px; pointer-events:none; display:none; font-size:12px;"></div>
      </div>
    </div>
    <script>
      (function(){{
        const key = "{key}";
        const boxes = {json.dumps(bxs)};
        let selectedIndex = {int(selected_index)};
        const img = new Image();
        img.src = "data:image/png;base64,{img_b64}";
        const canvas = document.getElementById("canvas-{key}");
        const ctx = canvas.getContext("2d");
        const tooltip = document.getElementById("tooltip-{key}");
        const fitBtn = document.getElementById("fit-{key}");
        const resetBtn = document.getElementById("reset-{key}");
        let scale = 1, minScale = 1, maxScale = 6;
        let offsetX = 0, offsetY = 0;
        let isPanning = false, startX=0, startY=0;

        function setCanvasSize() {{
          // Fit full image in container width
          const rect = canvas.getBoundingClientRect();
          const cw = rect.width;
          const ch = cw * (img.naturalHeight / img.naturalWidth);
          canvas.width = cw;
          canvas.height = ch;
          minScale = 1; // Always fit
          if (scale < minScale) scale = minScale;
          draw();
        }}

        function draw() {{
          ctx.clearRect(0,0,canvas.width, canvas.height);
          // Draw image with current pan/zoom
          ctx.save();
          ctx.translate(offsetX, offsetY);
          ctx.scale(scale, scale);
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

          // Draw boxes (coordinates are in image coord scaled to canvas)
          const sx = canvas.width / img.naturalWidth;
          const sy = canvas.height / img.naturalHeight;

          boxes.forEach((b, i) => {{
            const x1 = b.x1 * sx;
            const y1 = b.y1 * sy;
            const x2 = b.x2 * sx;
            const y2 = b.y2 * sy;
            ctx.beginPath();
            ctx.rect(x1, y1, x2 - x1, y2 - y1);
            if (i === selectedIndex) {{
              ctx.lineWidth = 2 / scale;
              ctx.strokeStyle = "#0a84ff";
            }} else {{
              ctx.lineWidth = 2 / scale;
              ctx.strokeStyle = (b.class === 1) ? "#00ffff" : "#ff3b30";
            }}
            ctx.stroke();
          }});
          ctx.restore();
        }}

        function pointToBoxIndex(px, py) {{
          // Convert screen point to image space considering pan/zoom
          const invScale = 1/scale;
          const x = (px - offsetX) * invScale;
          const y = (py - offsetY) * invScale;
          const sx = canvas.width / img.naturalWidth;
          const sy = canvas.height / img.naturalHeight;
          for (let i = boxes.length-1; i >= 0; i--) {{
            const b = boxes[i];
            const x1 = b.x1 * sx, y1 = b.y1 * sy, x2 = b.x2 * sx, y2 = b.y2 * sy;
            if (x >= x1 && x <= x2 && y >= y1 && y <= y2) return i;
          }}
          return -1;
        }}

        function showTooltip(e, i) {{
          if (i < 0) {{ tooltip.style.display = "none"; return; }}
          const b = boxes[i];
          tooltip.style.display = "block";
          tooltip.style.left = (e.offsetX + 14) + "px";
          tooltip.style.top  = (e.offsetY + 10) + "px";
          tooltip.innerHTML = "Row: " + i + "<br/>Area: " + b.area + "<br/>Conf: " + (b.confidence?.toFixed ? b.confidence.toFixed(2) : b.confidence);
        }}

        function fit() {{
          scale = 1; offsetX = 0; offsetY = 0; draw();
        }}
        function reset() {{
          selectedIndex = -1; scale = 1; offsetX = 0; offsetY = 0; draw();
          // Send selection back (cleared)
          const data = {{type: "selection", index: -1}};
          window.parent.postMessage({{"isStreamlitMessage":true,"type":"streamlit:setComponentValue","value":data}}, "*");
        }}

        img.onload = () => {{
          setCanvasSize();
          window.addEventListener("resize", setCanvasSize);
        }};

        // Zoom (wheel)
        canvas.addEventListener("wheel", (e) => {{
          e.preventDefault();
          const delta = e.deltaY < 0 ? 1.1 : 0.9;
          const mx = e.offsetX, my = e.offsetY;
          const prevScale = scale;
          scale = Math.max(minScale, Math.min(maxScale, scale * delta));
          // Keep mouse position stable
          offsetX = mx - (mx - offsetX) * (scale/prevScale);
          offsetY = my - (my - offsetY) * (scale/prevScale);
          draw();
        }}, {{passive:false}});

        // Pan
        canvas.addEventListener("mousedown", (e) => {{ isPanning = true; startX = e.offsetX - offsetX; startY = e.offsetY - offsetY; canvas.style.cursor='grabbing'; }});
        canvas.addEventListener("mouseup",   () => {{ isPanning = false; canvas.style.cursor='grab'; }});
        canvas.addEventListener("mouseleave",() => {{ isPanning = false; canvas.style.cursor='grab'; }});
        canvas.addEventListener("mousemove", (e) => {{
          if (isPanning) {{ offsetX = e.offsetX - startX; offsetY = e.offsetY - startY; draw(); }}
          const idx = pointToBoxIndex(e.offsetX, e.offsetY);
          showTooltip(e, idx);
        }});

        // Click select
        canvas.addEventListener("click", (e) => {{
          const idx = pointToBoxIndex(e.offsetX, e.offsetY);
          selectedIndex = (idx === selectedIndex) ? -1 : idx;
          draw();
          // Send selection back to Streamlit
          const data = {{type: "selection", index: selectedIndex}};
          window.parent.postMessage({{"isStreamlitMessage":true,"type":"streamlit:setComponentValue","value":data}}, "*");
        }});

        fitBtn.addEventListener("click", fit);
        resetBtn.addEventListener("click", reset);
      }})();
    </script>
    """

    # Use components.html with a sufficiently large height to avoid cropping.
    # Since the canvas height is tied to the container width, the iframe is set to a safe height.
    event = components.html(html, height=iframe_height, scrolling=False)
    # components.html doesn't directly return messages; we capture selection via Streamlit's component protocol.
    # Streamlit will automatically set a value for this component; read it from session_state:
    selected = st.session_state.get(f"cmp_{key}")
    if isinstance(selected, dict) and selected.get("type") == "selection":
        return int(selected.get("index", -1))
    return selected_index

# =========================
# UI Components
# =========================
class UI:
    @staticmethod
    def upload_section():
        left, right = st.columns([2,1])
        with left:
            image_files = st.file_uploader(
                "Upload Image(s)",
                type=list(DataLoader.SUPPORTED_FORMATS),
                accept_multiple_files=True
            )
        with right:
            txt_file = st.file_uploader("Upload Ground Truth (YOLO txt, optional)", type=['txt'])
        return image_files, txt_file

    @staticmethod
    def settings_sidebar(model_options, available_map):
        st.sidebar.header("⚙️ Settings")
        with st.sidebar.expander("Upload Custom YOLO Weights", expanded=False):
            uploaded_weights = st.file_uploader("YOLO Weights (.pt)", type=['pt'], key="weights_upload")
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

        active_models = [m for m in model_options if available_map.get(m, False)]
        if not active_models:
            st.sidebar.warning("No models available. Check weights.")
        model = st.sidebar.selectbox("Model", active_models or model_options)
        conf = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
        iou  = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.5, 0.01)

        st.sidebar.markdown("---")
        st.sidebar.subheader("Filter Detections")
        min_conf = st.sidebar.slider("Min Confidence (table/filter)", 0.0, 1.0, 0.0, 0.01)
        min_area = st.sidebar.number_input("Min Area (px²)", 0, value=0, step=10)
        max_area = st.sidebar.number_input("Max Area (px²)", 0, value=0, step=10, help="0 = no max limit")

        sort_by = st.sidebar.selectbox("Sort Table By", ["index","confidence","area","x1","y1"])
        sort_asc = st.sidebar.checkbox("Sort Ascending", value=False)

        return {
            "model": model, "conf": conf, "iou": iou,
            "min_conf": min_conf, "min_area": min_area, "max_area": max_area,
            "sort_by": sort_by, "sort_asc": sort_asc
        }

# =========================
# Main App
# =========================
class InsectDetectionApp:
    def __init__(self):
        self.loader = DataLoader()
        self.manager = ModelManager()
        self.ui = UI()

    def _filter_and_sort_df(self, df: pd.DataFrame, settings):
        if df.empty: return df
        # Filters
        if "confidence" in df.columns and settings["min_conf"] > 0:
            df = df[(df["confidence"].fillna(0) >= settings["min_conf"])]
        if settings["min_area"] > 0:
            df = df[df["area"] >= settings["min_area"]]
        if settings["max_area"] > 0:
            df = df[df["area"] <= settings["max_area"]]
        # Sort
        col = settings["sort_by"]
        if col in df.columns:
            df = df.sort_values(by=col, ascending=settings["sort_asc"])
        return df.reset_index(drop=True)

    def _export_buttons(self, image, boxes, classes, confidences, df, model_label):
        col1, col2, col3 = st.columns(3)
        # Annotated PNG
        with col1:
            if boxes is not None and len(boxes) > 0:
                annotated = draw_boxes_on_image(image, boxes, classes, confidences, None)
                _, buf = cv2.imencode('.png', cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                st.download_button("📸 Download Annotated PNG", data=buf.tobytes(),
                                   file_name=f"{model_label}_annotated.png",
                                   mime="image/png")
            else:
                st.button("📸 Download Annotated PNG", disabled=True)
        # CSV
        with col2:
            if df is not None and not df.empty:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📄 Download CSV", data=csv,
                                   file_name=f"{model_label}_detections.csv",
                                   mime="text/csv")
            else:
                st.button("📄 Download CSV", disabled=True)
        # JSON
        with col3:
            if df is not None and not df.empty:
                st.download_button("🧾 Download JSON",
                                   data=df.to_json(orient="records").encode('utf-8'),
                                   file_name=f"{model_label}_detections.json",
                                   mime="application/json")
            else:
                st.button("🧾 Download JSON", disabled=True)

    def run(self):
        st.title("🪲 Insect Detection – Pro Interface")
        st.caption("Pretty • Powerful • Reliable")

        image_files, txt_file = self.ui.upload_section()

        # Sidebar Settings
        model_options = ["YOLOv11n","Faster R-CNN","HOG_SVM"]
        settings = self.ui.settings_sidebar(model_options, self.manager.available)

        if not image_files:
            st.markdown(
                """
                <div class="tip">
                <b>Tip:</b> Upload one or more images to begin. Use the sidebar to select a model and thresholds.
                Hover over boxes for details, click to select, scroll to zoom, drag to pan.
                </div>
                """,
                unsafe_allow_html=True
            )
            return

        # Only process the first image for now; batch can be iterated if needed
        image_file = image_files[0]
        image = self.loader.load_image(image_file)
        if image is None:
            st.stop()

        # Optional ground-truth parsing (unused in UI below, but loaded if needed later)
        _ = self.loader.load_csv(txt_file, image) if txt_file else None

        # Predict
        with st.spinner("Running model..."):
            boxes, classes, confidences = self.manager.predict(
                image, settings["model"], settings["conf"], settings["iou"]
            )

        # Layout: viewer left, panel right
        viewer_col, panel_col = st.columns([7,5], gap="large")
        with viewer_col:
            st.subheader("Viewer", anchor=False)
            # Interactive canvas (returns selected index)
            st.markdown('<div class="small-muted">Full image is always fitted to width. Use zoom/pan for details.</div>', unsafe_allow_html=True)
            if boxes is None:
                st.warning("No output – model unavailable or prediction failed.")
                return
            if len(boxes) == 0:
                st.info("No detections.")
            selected_idx = st.session_state.get("selected_idx", -1)
            selected_idx = interactive_canvas(image, boxes, classes, confidences, key="main", selected_index=selected_idx)
            st.session_state["selected_idx"] = selected_idx

        with panel_col:
            st.subheader("Detections & Tools", anchor=False)

            # Table
            df_all = df_from_predictions(boxes, classes, confidences, settings["model"])
            df_view = self._filter_and_sort_df(df_all.copy(), settings)

            # Sync selection: if user selected a row in table below (via click action)
            st.markdown("**Detections**")
            if not df_view.empty:
                # Add a row index for reference
                df_view_display = df_view.copy()
                df_view_display.insert(0, "row", df_view_display.index)
                st.dataframe(df_view_display, use_container_width=True, hide_index=True)
                st.caption("Use filters in the sidebar. Click a box in the viewer to highlight it here.")

                # Selected Box quick view
                if 0 <= selected_idx < len(df_all):
                    st.markdown("**Selected Box**")
                    sel_row = df_all.iloc[selected_idx]
                    st.json({
                        "row": int(selected_idx),
                        "x1": int(sel_row["x1"]), "y1": int(sel_row["y1"]),
                        "x2": int(sel_row["x2"]), "y2": int(sel_row["y2"]),
                        "area": int(sel_row["area"]),
                        "confidence": None if np.isnan(sel_row["confidence"]) else float(sel_row["confidence"]),
                        "class": None if "class" not in sel_row else (None if pd.isna(sel_row.get("class", None)) else int(sel_row["class"]))
                    })
            else:
                st.info("No detections after filtering.")

            st.markdown("---")
            self._export_buttons(image, boxes, classes, confidences, df_all, settings["model"])

        # Compare Models Tab (optional lightweight view)
        with st.expander("🔬 Quick Compare (runs on current image)", expanded=False):
            compare_models = [m for m in model_options if m != settings["model"] and self.manager.available.get(m, False)]
            if not compare_models:
                st.write("No other available models to compare.")
            else:
                cols = st.columns(len(compare_models))
                for i, m in enumerate(compare_models):
                    with cols[i]:
                        st.markdown(f"**{m}**")
                        b, c, confs = self.manager.predict(image, m, settings["conf"], settings["iou"])
                        if b is None:
                            st.caption("Unavailable")
                            continue
                        thumb = draw_boxes_on_image(image, b, c, confs)
                        st.image(thumb, use_container_width=True)
                        st.caption(f"{len(b)} detections")

# =========================
# Run
# =========================
if __name__ == "__main__":
    app = InsectDetectionApp()
    app.run()
