import os
import cv2
import torch
from ultralytics import YOLO

# Define paths
MODEL_PATH = os.path.join("models", "best.pt")
INPUT_FOLDER = "test_videos"
OUTPUT_FOLDER = "output"

# Create necessary folders if they don't exist
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def main():
    # 1. Load the model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at '{MODEL_PATH}'. Please put your best.pt file there.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    device = 0 if torch.cuda.is_available() else "cpu"
    use_half = bool(torch.cuda.is_available())
    print(f"Using device: {'CUDA (GPU)' if device == 0 else 'CPU'} | FP16: {use_half}")

    # 2. Target specific video
    videos = ["Explosion027_x264.mp4"]

    if not videos:
        print(f"No videos found in '{INPUT_FOLDER}'. Please add some video files and try again.")
        return

    # 3. Process each video like a live CCTV feed
    for video in videos:
        input_path = os.path.join(INPUT_FOLDER, video)
        output_path = os.path.join(OUTPUT_FOLDER, video)
        print(f"\nStarting live feed simulation for {video}...")
        
        # Open the video file (This is exactly how you open a real CCTV RTSP stream or webcam!)
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties to save the output properly
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Setup the video writer for saving the output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("End of video stream.")
                break

            # Run YOLO inference on the single current frame
            # This is how real-time edge devices work
            results = model(
                frame,
                conf=0.4,
                verbose=False,
                device=device,
                half=use_half,
                imgsz=640,
            )  # verbose=False stops console spam
            
            # Get the frame with the bounding boxes drawn on it
            annotated_frame = results[0].plot()

            # Write to our saved output file
            out.write(annotated_frame)
            
            # SHOW the frame in a live window, just like a security guard's monitor
            cv2.imshow("CCTV Live Feed (Press 'Q' to stop)", annotated_frame)

            # Wait 1ms and check if the user pressed the 'q' key to quit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Live feed interrupted by user.")
                break
                
        # Clean up this video's resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"\nAll done! Processed videos are saved in the '{OUTPUT_FOLDER}' folder.")

if __name__ == "__main__":
    main()
