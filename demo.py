import cv2
import os
import numpy as np
from ultralytics import YOLO

# 1. GENERATE A CUSTOM BoT-SORT CONFIGURATION
# We increase the 'track_buffer' significantly. This tells the algorithm to remember 
# a person for up to 120 frames (about 4-5 seconds) if they get blocked by a shelf or move out of frame.
custom_yaml_path = "custom_botsort.yaml"
with open(custom_yaml_path, "w") as f:
    f.write("""
tracker_type: botsort
track_high_thresh: 0.3
track_low_thresh: 0.1
new_track_thresh: 0.6
track_buffer: 120       # MUCH larger buffer to prevent ID switching on occlusion
match_thresh: 0.95
gmc_method: sparseOptFlow
proximity_thresh: 0.5
appearance_thresh: 0.25
with_reid: true
fuse_score: true
model: auto
""")

# Folders configuration
INPUT_DIR = "shoplifting_Videos"
OUTPUT_DIR = "output_botsort"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. LOAD YOLO11s (More accurate, larger model for continuity)
print("[*] Loading YOLO11s model...")
model = YOLO("yolo12s.pt") 

# Process a single specific video
video_files = ["Shoplifting001.mp4"]
print(f"[*] Target video set to: {video_files[0]}")

# Generate a static palette of 1000 vibrant colors to assign to different IDs
colors = np.random.randint(0, 255, size=(1000, 3), dtype="int")

for video_name in video_files:
    video_path = os.path.join(INPUT_DIR, video_name)
    output_path = os.path.join(OUTPUT_DIR, video_name)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video {video_name}")
        continue
        
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30
        
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    print(f"\n🎥 Processing: {video_name}")
    print(f"   Settings: YOLO11s | Native Res | custom track_buffer=120 | ReID Active")
    
    frame_count = 0
    
    # Process frame by frame manually so we can draw custom thick colors
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        results = model.track(
            frame, 
            persist=True, 
            tracker=custom_yaml_path, 
            classes=[0], 
            conf=0.4, 
            verbose=False
        )
        
        annotated_frame = frame.copy()
        
        # 4. CUSTOM ID DRAWING
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = box
                
                # Assign a specific, constant color to each unique ID
                color = [int(c) for c in colors[track_id % len(colors)]]
                
                # Draw thin bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw ID Tag background
                cv2.rectangle(annotated_frame, (x1, y1 - 22), (x1 + 60, y1), color, -1)
                
                # Write ID Number
                cv2.putText(annotated_frame, f"ID: {track_id}", (x1 + 5, y1 - 6), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
        out.write(annotated_frame)
        frame_count += 1
        
        if frame_count % 150 == 0:
            print(f"   -> Processed {frame_count} frames...")
            
    cap.release()
    out.release()
    print(f"✅ Saved advanced processed video to: {output_path}")

print("\n🚀 All videos have been tracked using ADVANCED BoT-SORT!")
