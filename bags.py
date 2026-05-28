import cv2
import os
import math
import numpy as np
import torch
from ultralytics import YOLO

# =========================
# CHECK DEVICE (GPU/CPU)
# =========================
if torch.cuda.is_available():
    print("CUDA is available")
    print("GPU:", torch.cuda.get_device_name(0))
    DEVICE = 0
else:
    print("CUDA GPU not detected! Falling back to CPU...")
    DEVICE = "cpu"

# =========================
# CONFIG
# =========================
POSE_MODEL_PATH = "yolo11n-pose.pt"
BAG_MODEL_PATH = "yolo12n.pt"  # Or yolo11s.pt

INPUT_FOLDER = "shoplifting"
OUTPUT_FOLDER = "output_videos"

# COCO Classes: 24 (backpack), 26 (handbag), 28 (suitcase)
BAG_CLASSES = [24, 26, 28]

FRAMES_THRESHOLD = 15  # Consecutive frames hand must be in bag to trigger alert

# =========================
# HELPER FUNCTIONS
# =========================
def is_point_near_box(px, py, box, margin=20):
    """Check if a point (x,y) is inside or very close to a bounding box."""
    x1, y1, x2, y2 = box
    return (x1 - margin <= px <= x2 + margin) and (y1 - margin <= py <= y2 + margin)

def main():
    print(f"[*] Loading YOLO Pose model ({POSE_MODEL_PATH})...")
    pose_model = YOLO(POSE_MODEL_PATH)
    if DEVICE != "cpu": pose_model.to(DEVICE)
    
    print(f"[*] Loading YOLO Bag model ({BAG_MODEL_PATH})...")
    bag_model = YOLO(BAG_MODEL_PATH)
    if DEVICE != "cpu": bag_model.to(DEVICE)
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    video_files = ["WIN_20260518_13_05_06_Pro.mp4"]  # Change to your target video
    
    for video_name in video_files:
        input_path = os.path.join(INPUT_FOLDER, video_name)
        if not os.path.exists(input_path):
            print(f"❌ Video {input_path} not found.")
            continue
            
        output_path = os.path.join(OUTPUT_FOLDER, f"detected_{video_name}")
        
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0: fps = 30
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"\n🎥 Processing: {video_name}")
        
        hand_in_bag_state = {} # Track frames where hand is in bag per person ID
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            annotated_frame = frame.copy()
            
            # 1. Detect Bags
            bag_results = bag_model(frame, classes=BAG_CLASSES, conf=0.35, verbose=False, device=DEVICE)
            bag_boxes = []
            if bag_results[0].boxes:
                for box in bag_results[0].boxes:
                    cls_id = int(box.cls[0])
                    cls_name = bag_model.names[cls_id]
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    bag_boxes.append((x1, y1, x2, y2))
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                    cv2.putText(annotated_frame, f"{cls_name.capitalize()} {box.conf[0].item():.2f}", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                                
            # 2. Run Pose Tracking for Humans
            pose_results = pose_model.track(frame, persist=True, verbose=False, device=DEVICE)
            
            if pose_results[0].boxes is not None and pose_results[0].boxes.id is not None and pose_results[0].keypoints is not None:
                boxes = pose_results[0].boxes.xyxy.cpu().numpy()
                ids = pose_results[0].boxes.id.cpu().numpy().astype(int)
                keypoints = pose_results[0].keypoints.xy.cpu().numpy()
                confs = pose_results[0].keypoints.conf.cpu().numpy()
                
                for box, track_id, kpts, conf in zip(boxes, ids, keypoints, confs):
                    x1, y1, x2, y2 = map(int, box)
                    
                    left_wrist = kpts[9]
                    right_wrist = kpts[10]
                    lw_conf, rw_conf = conf[9], conf[10]
                    
                    hand_is_in_bag = False
                    
                    # Check if either wrist is inside any bag bounding box
                    for b_box in bag_boxes:
                        if lw_conf > 0.5 and is_point_near_box(left_wrist[0], left_wrist[1], b_box):
                            hand_is_in_bag = True
                        if rw_conf > 0.5 and is_point_near_box(right_wrist[0], right_wrist[1], b_box):
                            hand_is_in_bag = True
                            
                    if hand_is_in_bag:
                        hand_in_bag_state[track_id] = hand_in_bag_state.get(track_id, 0) + 1
                    else:
                        hand_in_bag_state[track_id] = 0 # Reset if hand leaves bag
                        
                    is_suspicious = hand_in_bag_state.get(track_id, 0) >= FRAMES_THRESHOLD
                    
                    # Visual Output
                    color = (0, 0, 255) if is_suspicious else (0, 255, 0)
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    if is_suspicious:
                        cv2.putText(annotated_frame, f"ID:{track_id} HAND IN BAG!", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 3)
                    else:
                        cv2.putText(annotated_frame, f"ID:{track_id}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                                    
                    # Draw wrist dots for debugging
                    if lw_conf > 0.5: cv2.circle(annotated_frame, (int(left_wrist[0]), int(left_wrist[1])), 5, (255, 0, 0), -1)
                    if rw_conf > 0.5: cv2.circle(annotated_frame, (int(right_wrist[0]), int(right_wrist[1])), 5, (0, 255, 255), -1)

            out.write(annotated_frame)
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"   -> Processed {frame_count} frames...")
                
        cap.release()
        out.release()
        print(f"✅ Saved output to: {output_path}")

if __name__ == "__main__":
    main()