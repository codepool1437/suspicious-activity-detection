import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Retail Security AI", layout="wide")
st.title("Intelligent Retail Security AI")
st.markdown("Select an AI Module from the sidebar to analyze footage. Keeping them separate ensures visual clarity!")

# Resolve paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOODIE_MODEL_PATH = os.path.join(BASE_DIR, "Model", "Hoodie-detection", "weights", "best.pt")
BOTSORT_YAML_PATH = os.path.join(BASE_DIR, "custom_botsort.yaml")

# Sidebar Configuration
st.sidebar.title("Control Panel")
app_mode = st.sidebar.radio("Select AI Module:", 
    ["1. Baggage & Carrier Detection", "2. Customer Tracking (BoT-SORT)", "3. Hoodie Detection"]
)
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4)

# ================================
# MODEL LOADING (Cached for speed)
# ================================
@st.cache_resource
def load_hoodie_model():
    if os.path.exists(HOODIE_MODEL_PATH):
        model = YOLO(HOODIE_MODEL_PATH)
        return model
    return None

@st.cache_resource
def load_general_model():
    return YOLO("yolo11s.pt")

# ================================
# PROCESSING LOGIC
# ================================

if app_mode == "3. Hoodie Detection":
    st.subheader("Custom Hoodie Detection")
    model = load_hoodie_model()
    if not model:
        st.error(f"Hoodie Model not found! Expected at: {HOODIE_MODEL_PATH}")
        st.stop()
    option = "Video"

elif app_mode == "1. Baggage & Carrier Detection":
    st.subheader("Baggage & Carrier Detection")
    model = load_general_model()
    option = st.radio("Choose Input:", ("Image", "Video"))

elif app_mode == "2. Customer Tracking (BoT-SORT)":
    st.subheader("Customer Trajectory Tracking")
    model = load_general_model()
    option = "Video"

# ================================
# MEDIA UPLOADER & EXECUTION
# ================================

if option == "Image":
    uploaded_file = st.file_uploader("Upload an Image...", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_container_width=True)
        
        if st.button("Detect Now"):
            with st.spinner("Analyzing image..."):
                cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                results = model.predict(cv_img, conf=conf_threshold, verbose=False)
                
                if app_mode == "3. Hoodie Detection":
                    annotated_frame = cv_img.copy()
                    for box in results[0].boxes:
                        class_id = int(box.cls[0])
                        display_text = "Hoodie" if class_id == 1 else "Normal"
                        color = (0, 0, 255) if class_id == 1 else (255, 0, 0)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.rectangle(annotated_frame, (x1, y1 - 22), (x1 + max(80, len(display_text)*12), y1), color, -1)
                        cv2.putText(annotated_frame, f"{display_text} {box.conf[0]:.2f}", (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                elif app_mode == "1. Baggage & Carrier Detection":
                    annotated_frame = cv_img.copy()
                    for box in results[0].boxes:
                        cls_name = model.names[int(box.cls[0])]
                        if cls_name in ["backpack", "handbag", "suitcase"]:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.putText(annotated_frame, f"{cls_name} {box.conf[0]:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                else:
                    annotated_rgb = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
                    
                st.image(annotated_rgb, caption="Output Detection", use_container_width=True)

elif option == "Video":
    uploaded_file = st.file_uploader("Upload a Video...", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        
        if st.button("Start Processing"):
            stframe = st.empty()
            cap = cv2.VideoCapture(tfile.name)
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if fps == 0: fps = 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            
            progress_bar = st.progress(0)
            frame_count = 0
            colors = np.random.randint(0, 255, size=(1000, 3), dtype="int")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                annotated_frame = frame.copy()
                
                if app_mode == "3. Hoodie Detection":
                    results = model.predict(frame, conf=conf_threshold, verbose=False)
                    for box in results[0].boxes:
                        class_id = int(box.cls[0])
                        display_text = "Hoodie" if class_id == 1 else "Normal"
                        color = (0, 0, 255) if class_id == 1 else (255, 0, 0)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.rectangle(annotated_frame, (x1, y1 - 22), (x1 + max(80, len(display_text)*12), y1), color, -1)
                        cv2.putText(annotated_frame, f"{display_text} {box.conf[0]:.2f}", (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                elif app_mode == "1. Baggage & Carrier Detection":
                    results = model.predict(frame, conf=conf_threshold, verbose=False)
                    for box in results[0].boxes:
                        cls_name = model.names[int(box.cls[0])]
                        if cls_name in ["backpack", "handbag", "suitcase"]:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.putText(annotated_frame, f"{cls_name} {box.conf[0]:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                elif app_mode == "2. Customer Tracking (BoT-SORT)":
                    track_args = {"source": frame, "persist": True, "classes": [0], "conf": conf_threshold, "verbose": False}
                    
                    # Ensure ReID is explicitly ignored at runtime if the yaml is injected
                    if os.path.exists(BOTSORT_YAML_PATH): 
                        track_args["tracker"] = BOTSORT_YAML_PATH
                        
                    results = model.track(**track_args)
                    if results[0].boxes.id is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                        ids = results[0].boxes.id.cpu().numpy().astype(int)
                        for box, track_id in zip(boxes, ids):
                            x1, y1, x2, y2 = box
                            color = [int(c) for c in colors[track_id % len(colors)]]
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.rectangle(annotated_frame, (x1, y1 - 22), (x1 + 60, y1), color, -1)
                            cv2.putText(annotated_frame, f"ID:{track_id}", (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                out.write(annotated_frame)
                stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                frame_count += 1
                if total_frames > 0: progress_bar.progress(min(frame_count / total_frames, 1.0))
            
            cap.release()
            out.release()
            st.success("Analysis Complete!")
            
            with open(out_path, 'rb') as f:
                st.download_button("Download Result Video", f, file_name=f"{app_mode.split('.')[0]}_output.mp4", mime="video/mp4")
