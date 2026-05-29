import cv2
import os
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
BAG_MODEL_PATH = "yolo12n.pt" 
INPUT_FOLDER = "shoplifting"
OUTPUT_FOLDER = "output_videos"

# COCO Classes: 24 (backpack), 26 (handbag), 28 (suitcase)
BAG_CLASSES = [24, 26, 28]

def main():
    print(f"[*] Loading YOLO Bag model ({BAG_MODEL_PATH})...")
    bag_model = YOLO(BAG_MODEL_PATH)
    if DEVICE != "cpu": bag_model.to(DEVICE)
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Input folder '{INPUT_FOLDER}' does not exist!")
        return

    video_files = ["WhatsApp Video 2026-05-29 at 7.10.32 AM.mp4"]  # Change to your target video
    
    for video_name in video_files:
        input_path = os.path.join(INPUT_FOLDER, video_name)
        if not os.path.exists(input_path):
            print(f"Video {input_path} not found.")
            continue
            
        output_path = os.path.join(OUTPUT_FOLDER, f"detected_{video_name}")
        
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0: fps = 30
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"\n Processing: {video_name}")
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            annotated_frame = frame.copy()
            
            # 1. Detect Bags
            bag_results = bag_model(frame, classes=BAG_CLASSES, conf=0.35, verbose=False, device=DEVICE)
            if bag_results[0].boxes:
                for box in bag_results[0].boxes:
                    cls_id = int(box.cls[0])
                    cls_name = bag_model.names[cls_id]
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                    cv2.putText(annotated_frame, f"{cls_name.capitalize()} {box.conf[0].item():.2f}", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                                
            out.write(annotated_frame)
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"   -> Processed {frame_count} frames...")
                
        cap.release()
        out.release()
        print(f"Saved output to: {output_path}")

if __name__ == "__main__":
    main()