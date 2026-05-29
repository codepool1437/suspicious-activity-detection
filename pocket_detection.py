import cv2
import os
import math
import numpy as np
from ultralytics import YOLO

# Configuration
INPUT_FOLDER = os.path.join("shoplifting", "new")
OUTPUT_FOLDER = "output_videos"
POSE_MODEL_PATH = "yolo11n-pose.pt" 
OBJ_MODEL_PATH = "yolo11s.pt"  # Upgraded to 'Small' model for better accuracy

# Heuristic parameters
POCKET_DISTANCE_THRESHOLD = 50.0  # Max distance in pixels between wrist and hip
PHONE_DISTANCE_THRESHOLD = 80.0   # Max distance in pixels between wrist and item
FRAMES_THRESHOLD = 10            # Consecutive frames hand must be near pocket

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def main():
    print(f"[*] Loading YOLO Pose model ({POSE_MODEL_PATH})...")
    pose_model = YOLO(POSE_MODEL_PATH)
    
    print(f"[*] Loading YOLO Object model ({OBJ_MODEL_PATH})...")
    obj_model = YOLO(OBJ_MODEL_PATH)
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Input folder '{INPUT_FOLDER}' does not exist!")
        return

    # Process all video files in the INPUT_FOLDER
    video_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        print(f"No video files found in {INPUT_FOLDER}")
        return
        
    print(f"[*] Found {len(video_files)} videos to process in {INPUT_FOLDER}")
    
    for video_name in video_files:
        input_path = os.path.join(INPUT_FOLDER, video_name)
        output_path = os.path.join(OUTPUT_FOLDER, f"pocket_{video_name}")
        
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0: fps = 30
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"\n[*] Processing: {video_name}...")
        
        pocketing_state = {} 
        has_phone_history = {} # Track if ID recently held a phone
        
        frame_count = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            annotated_frame = frame.copy()
                
            # 1. Detect Cell Phones
            # We lowered confidence to 0.15 to catch blurry/occluded phones
            obj_results = obj_model(frame, classes=[67], conf=0.15, verbose=False)
            phone_centers = []
            if obj_results[0].boxes:
                for box in obj_results[0].boxes:
                    # Draw object bounding box in Cyan
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    center = get_box_center((x1, y1, x2, y2))
                    phone_centers.append(center)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Phone {box.conf[0].item():.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    
            # 2. Run Pose Tracking for Humans
            pose_results = pose_model.track(frame, persist=True, verbose=False)
            
            if pose_results[0].boxes is not None and pose_results[0].boxes.id is not None and pose_results[0].keypoints is not None:
                boxes = pose_results[0].boxes.xyxy.cpu().numpy()
                ids = pose_results[0].boxes.id.cpu().numpy().astype(int)
                keypoints = pose_results[0].keypoints.xy.cpu().numpy()
                confs = pose_results[0].keypoints.conf.cpu().numpy()
                
                for box, track_id, kpts, conf in zip(boxes, ids, keypoints, confs):
                    x1, y1, x2, y2 = map(int, box)
                    
                    left_wrist = kpts[9]
                    right_wrist = kpts[10]
                    left_hip = kpts[11]
                    right_hip = kpts[12]
                    
                    lw_conf, rw_conf = conf[9], conf[10]
                    lh_conf, rh_conf = conf[11], conf[12]
                    
                    # Check if this person is holding a phone
                    is_holding_phone = False
                    for p_center in phone_centers:
                        if lw_conf > 0.5 and calculate_distance(left_wrist, p_center) < PHONE_DISTANCE_THRESHOLD:
                            is_holding_phone = True
                        if rw_conf > 0.5 and calculate_distance(right_wrist, p_center) < PHONE_DISTANCE_THRESHOLD:
                            is_holding_phone = True
                            
                    # Memory system: Increased to 60 frames (~2 seconds)
                    if is_holding_phone:
                        has_phone_history[track_id] = 60 
                    elif has_phone_history.get(track_id, 0) > 0:
                        has_phone_history[track_id] -= 1
                        
                    # Check distance to pocket
                    min_dist = float('inf')
                    if lw_conf > 0.5 and lh_conf > 0.5:
                        min_dist = min(min_dist, calculate_distance(left_wrist, left_hip))
                    if rw_conf > 0.5 and rh_conf > 0.5:
                        min_dist = min(min_dist, calculate_distance(right_wrist, right_hip))
                        
                    # CORE LOGIC: Hand is near pocket AND they recently held a phone!
                    if min_dist < POCKET_DISTANCE_THRESHOLD and has_phone_history.get(track_id, 0) > 0:
                        pocketing_state[track_id] = pocketing_state.get(track_id, 0) + 1
                    elif min_dist >= POCKET_DISTANCE_THRESHOLD:
                        pocketing_state[track_id] = 0 # Reset
                        
                    is_pocketing = pocketing_state.get(track_id, 0) >= FRAMES_THRESHOLD
                    
                    # Visual Output
                    color = (0, 0, 255) if is_pocketing else (0, 255, 0)
                    label = f"ID:{track_id} {'PHONE POCKETED!' if is_pocketing else ''}"
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Only draw text if doing something suspicious to keep screen clean
                    if is_pocketing:
                        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 3)
                    elif has_phone_history.get(track_id, 0) > 0:
                        cv2.putText(annotated_frame, f"ID:{track_id} (Holding Phone)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    else:
                        cv2.putText(annotated_frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Draw skeleton joints
                    if lw_conf > 0.5: cv2.circle(annotated_frame, (int(left_wrist[0]), int(left_wrist[1])), 5, (255, 0, 0), -1)
                    if lh_conf > 0.5: cv2.circle(annotated_frame, (int(left_hip[0]), int(left_hip[1])), 5, (255, 0, 0), -1)
                    if rw_conf > 0.5: cv2.circle(annotated_frame, (int(right_wrist[0]), int(right_wrist[1])), 5, (0, 255, 255), -1)
                    if rh_conf > 0.5: cv2.circle(annotated_frame, (int(right_hip[0]), int(right_hip[1])), 5, (0, 255, 255), -1)
    
            out.write(annotated_frame)
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"   -> Processed {frame_count} frames...")
                
        cap.release()
        out.release()
        print(f"✅ Saved advanced processed video to: {output_path}")

if __name__ == "__main__":
    main()
