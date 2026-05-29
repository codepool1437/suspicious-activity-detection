import streamlit as st
import cv2
import tempfile
import os
import math
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

# Heuristic parameters
POCKET_DISTANCE_THRESHOLD = 50.0  
PHONE_DISTANCE_THRESHOLD = 80.0   
FRAMES_THRESHOLD = 10            

# Helper Functions
def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def is_point_near_box(px, py, box, margin=20):
    x1, y1, x2, y2 = box
    return (x1 - margin <= px <= x2 + margin) and (y1 - margin <= py <= y2 + margin)

# Sidebar Configuration
st.sidebar.title("Control Panel")
app_mode = st.sidebar.radio("Select AI Module:", 
    [
        "1. Customer Tracking (BoT-SORT)", 
        "2. Bag Detection", 
        "3. Action: Pocketing (Cell Phone)",
        "4. Hoodie Detection"
    ]
)
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4)

# ================================
# MODEL LOADING (Cached for speed)
# ================================
@st.cache_resource
def load_hoodie_model():
    if os.path.exists(HOODIE_MODEL_PATH):
        return YOLO(HOODIE_MODEL_PATH)
    return None

@st.cache_resource
def load_general_model():
    return YOLO("yolo12n.pt") # For Tracking / Bag

@st.cache_resource
def load_small_model():
    return YOLO("yolo11s.pt") # For Phones

@st.cache_resource
def load_pose_model():
    return YOLO("yolo11n-pose.pt") # For Actions

# ================================
# PROCESSING LOGIC SETUP
# ================================

if app_mode == "1. Customer Tracking (BoT-SORT)":
    st.subheader("Customer Trajectory Tracking")
    model = load_general_model()

elif app_mode == "2. Bag Detection":
    st.subheader("Bag Detection")
    bag_model = load_general_model()

elif app_mode == "3. Action: Pocketing (Cell Phone)":
    st.subheader("Action Recognition: Item Pocketing")
    obj_model = load_small_model()
    pose_model = load_pose_model()
    
elif app_mode == "4. Hoodie Detection":
    st.subheader("Custom Hoodie Detection")
    model = load_hoodie_model()
    if not model:
        st.error(f"Hoodie Model not found! Expected at: {HOODIE_MODEL_PATH}")
        st.stop()

# ================================
# MEDIA UPLOADER & EXECUTION
# ================================

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
        
        # State tracking variables for Action Modules
        hand_in_bag_state = {}
        pocketing_state = {}
        has_phone_history = {}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            annotated_frame = frame.copy()
            
            # MODULE 1: CUSTOMER TRACKING
            if app_mode == "1. Customer Tracking (BoT-SORT)":
                track_args = {"source": frame, "persist": True, "classes": [0], "conf": conf_threshold, "verbose": False}
                if os.path.exists(BOTSORT_YAML_PATH): 
                    track_args["tracker"] = BOTSORT_YAML_PATH
                    
                results = model.track(**track_args)
                if results[0].boxes and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    ids = results[0].boxes.id.cpu().numpy().astype(int)
                    for box, track_id in zip(boxes, ids):
                        x1, y1, x2, y2 = box
                        color = [int(c) for c in colors[track_id % len(colors)]]
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.rectangle(annotated_frame, (x1, y1 - 22), (x1 + 60, y1), color, -1)
                        cv2.putText(annotated_frame, f"ID:{track_id}", (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
            # MODULE 2: BAG DETECTION
            elif app_mode == "2. Bag Detection":
                bag_results = bag_model(frame, classes=[24, 26, 28], conf=conf_threshold, verbose=False)
                if bag_results[0].boxes:
                    for box in bag_results[0].boxes:
                        cls_name = bag_model.names[int(box.cls[0])]
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                        cv2.putText(annotated_frame, f"{cls_name.capitalize()} {box.conf[0].item():.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

            # MODULE 3: POCKETING
            elif app_mode == "3. Action: Pocketing (Cell Phone)":
                obj_results = obj_model(frame, classes=[67], conf=0.15, verbose=False)
                phone_centers = []
                if obj_results[0].boxes:
                    for box in obj_results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        center = get_box_center((x1, y1, x2, y2))
                        phone_centers.append(center)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        cv2.putText(annotated_frame, f"Phone {box.conf[0].item():.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        
                pose_results = pose_model.track(frame, persist=True, verbose=False)
                if pose_results[0].boxes is not None and pose_results[0].boxes.id is not None and pose_results[0].keypoints is not None:
                    boxes = pose_results[0].boxes.xyxy.cpu().numpy()
                    ids = pose_results[0].boxes.id.cpu().numpy().astype(int)
                    keypoints = pose_results[0].keypoints.xy.cpu().numpy()
                    confs = pose_results[0].keypoints.conf.cpu().numpy()
                    
                    for box, track_id, kpts, conf in zip(boxes, ids, keypoints, confs):
                        x1, y1, x2, y2 = map(int, box)
                        left_wrist, right_wrist = kpts[9], kpts[10]
                        left_hip, right_hip = kpts[11], kpts[12]
                        lw_conf, rw_conf = conf[9], conf[10]
                        lh_conf, rh_conf = conf[11], conf[12]
                        
                        is_holding_phone = False
                        for p_center in phone_centers:
                            if lw_conf > 0.5 and calculate_distance(left_wrist, p_center) < PHONE_DISTANCE_THRESHOLD: is_holding_phone = True
                            if rw_conf > 0.5 and calculate_distance(right_wrist, p_center) < PHONE_DISTANCE_THRESHOLD: is_holding_phone = True
                                
                        if is_holding_phone:
                            has_phone_history[track_id] = 60 
                        elif has_phone_history.get(track_id, 0) > 0:
                            has_phone_history[track_id] -= 1
                            
                        min_dist = float('inf')
                        if lw_conf > 0.5 and lh_conf > 0.5: min_dist = min(min_dist, calculate_distance(left_wrist, left_hip))
                        if rw_conf > 0.5 and rh_conf > 0.5: min_dist = min(min_dist, calculate_distance(right_wrist, right_hip))
                            
                        if min_dist < POCKET_DISTANCE_THRESHOLD and has_phone_history.get(track_id, 0) > 0:
                            pocketing_state[track_id] = pocketing_state.get(track_id, 0) + 1
                        elif min_dist >= POCKET_DISTANCE_THRESHOLD:
                            pocketing_state[track_id] = 0
                            
                        is_pocketing = pocketing_state.get(track_id, 0) >= FRAMES_THRESHOLD
                        color = (0, 0, 255) if is_pocketing else (0, 255, 0)
                        
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        if is_pocketing:
                            cv2.putText(annotated_frame, f"ID:{track_id} PHONE POCKETED!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 3)
                        elif has_phone_history.get(track_id, 0) > 0:
                            cv2.putText(annotated_frame, f"ID:{track_id} (Holding Phone)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        else:
                            cv2.putText(annotated_frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # MODULE 4: HOODIE DETECTION
            elif app_mode == "4. Hoodie Detection":
                results = model.predict(frame, conf=conf_threshold, verbose=False)
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    display_text = "Hoodie" if class_id == 1 else "Normal"
                    color = (0, 0, 255) if class_id == 1 else (255, 0, 0)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - 22), (x1 + max(80, len(display_text)*12), y1), color, -1)
                    cv2.putText(annotated_frame, f"{display_text} {box.conf[0]:.2f}", (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            out.write(annotated_frame)
            stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            frame_count += 1
            if total_frames > 0: progress_bar.progress(min(frame_count / total_frames, 1.0))
        
        cap.release()
        out.release()
        st.success("Analysis Complete!")
        
        with open(out_path, 'rb') as f:
            st.download_button("Download Result Video", f, file_name=f"{app_mode.split('.')[0]}_output.mp4", mime="video/mp4")
