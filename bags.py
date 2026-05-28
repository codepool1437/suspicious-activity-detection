# pip install ultralytics opencv-python torch torchvision torchaudio

from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import torch

# =========================
# CHECK DEVICE (GPU/CPU)
# =========================

if torch.cuda.is_available():
    print("CUDA is available")
    print("GPU:", torch.cuda.get_device_name(0))
    DEVICE = 0  # GPU index 0
else:
    print("CUDA GPU not detected! Falling back to CPU...")
    DEVICE = "cpu"

# =========================
# CONFIG
# =========================

# YOLOv12 model
# yolov12n.pt -> Nano
# yolov12s.pt -> Small
MODEL_PATH = "yolo12n.pt"

INPUT_FOLDER = "shoplifting"
OUTPUT_FOLDER = "output_videos"

CONFIDENCE = 0.35

# Bag-related classes from COCO
BAG_CLASSES = [
    "backpack",
    "handbag",
    "suitcase"
]

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]

# =========================
# LOAD MODEL
# =========================

model = YOLO(MODEL_PATH)

# Use detected device
if DEVICE != "cpu":
    model.to("cuda")

print(f"Model loaded and ready for processing on: {DEVICE}")

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =========================
# GET VIDEO FILES
# =========================

video_files = ["shoplifting-1.mp4"]

print(f"Found {len(video_files)} videos")

# =========================
# PROCESS VIDEOS
# =========================

for video_name in video_files:

    input_path = os.path.join(INPUT_FOLDER, video_name)

    output_path = os.path.join(
        OUTPUT_FOLDER,
        f"detected_{video_name}"
    )

    print(f"\nProcessing: {video_name}")

    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # MP4 output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # =========================
        # YOLO INFERENCE
        # =========================

        results = model.predict(
            source=frame,
            device=DEVICE,     # Uses GPU 0 if available, else "cpu"
            conf=CONFIDENCE,
            verbose=False
        )

        result = results[0]

        # =========================
        # DRAW DETECTIONS
        # =========================

        for box in result.boxes:

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            class_name = model.names[cls_id]

            if class_name in BAG_CLASSES:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # Green rectangle
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                label = f"{class_name} {conf:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        out.write(frame)

        frame_count += 1

        if frame_count % 30 == 0:
            print(f"{video_name}: {frame_count} frames processed")

    cap.release()
    out.release()

    print(f"Saved -> {output_path}")

print("\nAll videos processed successfully.")