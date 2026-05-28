# 🛡️ Suspicious Activity Detection in Retail Environments

> **TCS Industry Project** — AI-powered surveillance system for detecting shoplifting indicators in retail CCTV footage using YOLO object detection and BoT-SORT multi-object tracking.

---

## 📌 Problem Statement

Retail stores face significant losses due to shoplifting. Traditional CCTV monitoring relies entirely on human operators who cannot watch every camera feed simultaneously. This project aims to build an **intelligent video analytics system** that automatically detects suspicious behavioral indicators such as:

- Persons wearing **hoodies** (face concealment)
- Persons carrying **bags and carriers** (backpacks, handbags, suitcases)
- **Customer trajectory tracking** for movement analysis

---

## 🏗️ System Architecture

The system is built around **three independent detection modules**, each targeting a specific shoplifting indicator:

```
┌─────────────────────────────────────────────────────┐
│              Streamlit Web Interface                 │
│         (Upload Video → View Detections)            │
├────────────────┬───────────────┬─────────────────────┤
│   Module 1     │   Module 2    │     Module 3        │
│   Hoodie       │   Bag/Carrier │     Customer        │
│   Detection    │   Detection   │     Tracking        │
├────────────────┼───────────────┼─────────────────────┤
│ Custom YOLO12n │ Pretrained    │ YOLO12s +           │
│ (best.pt)      │ YOLO (COCO)   │ BoT-SORT (ReID)     │
└────────────────┴───────────────┴─────────────────────┘
```

---

## 🔍 Module Details

### 1. Hoodie Detection (Custom Trained)

Detects persons wearing hoodies — a common face-concealment tactic during shoplifting.

| Detail | Value |
|---|---|
| **Base Model** | YOLOv12n (Nano) |
| **Classes** | `Normal` (0), `Hoodie` (1) |
| **Dataset** | 3,959 images (mined from UFC Crime Dataset) |
| **Annotation** | Hand-annotated on [Roboflow](https://universe.roboflow.com/jaydips-workspace/person-with-hoodie-detection/dataset/1) |
| **Augmentation** | Horizontal flip, brightness adjustment, Gaussian blur |
| **Training** | 300 epochs, batch 16, imgsz 640, patience 50 |
| **Platform** | Kaggle GPU |
| **Weights** | `Model/Hoodie-detection/weights/best.pt` |

### 2. Bag & Carrier Detection

Detects bags and carriers that could be used to conceal stolen merchandise.

| Detail | Value |
|---|---|
| **Model** | Pretrained YOLO (COCO dataset) |
| **Target Classes** | `backpack`, `handbag`, `suitcase` |
| **Approach** | Filters relevant classes from COCO's 80-class detection |
| **Confidence** | 0.35 threshold |

### 3. Customer Tracking (BoT-SORT)

Tracks individual customers across frames with persistent IDs for movement analysis.

| Detail | Value |
|---|---|
| **Model** | YOLOv12s (Small) |
| **Tracker** | BoT-SORT with ReID |
| **Track Buffer** | 120 frames (~4-5 sec memory on occlusion) |
| **Target Class** | `person` (COCO class 0) |
| **Features** | Sparse optical flow GMC, appearance matching, unique color per ID |

---

## 📁 Project Structure

```
industry_project/
│
├── Model/
│   └── Hoodie-detection/
│       ├── weights/
│       │   └── best.pt              # Custom trained hoodie detection weights
│       ├── training_results/        # PR curves, confusion matrix, training logs
│       └── data.yaml                # Dataset configuration
│
├── frontend/
│   └── app.py                      # Streamlit web interface
│
├── hoodie_detection.py             # Standalone hoodie detection script
├── bags.py                         # Standalone bag detection script
├── advanced_botsort.py             # Standalone customer tracking script
├── custom_botsort.yaml             # BoT-SORT tracker configuration
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended, CPU fallback supported)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd industry_project

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Option 1: Streamlit Web Interface (Recommended)

The web interface lets you upload videos and run any of the three detection modules interactively.

```bash
cd frontend
streamlit run app.py
```

This opens a browser UI where you can:
1. Select a detection module from the sidebar
2. Adjust the confidence threshold
3. Upload a video file
4. View real-time detection results
5. Download the processed output video

### Option 2: Standalone Scripts

Run individual detection modules directly from the command line:

```bash
# Hoodie Detection
python hoodie_detection.py

# Bag & Carrier Detection
python bags.py

# Customer Tracking (BoT-SORT)
python advanced_botsort.py
```

> **Note:** Edit the `video_files` list inside each script to specify which videos to process.

---

## 📊 Training Results

Training artifacts for the custom hoodie detection model are available in `Model/Hoodie-detection/training_results/`:

- `results.png` — Training loss and metrics over epochs
- `confusion_matrix.png` — Classification performance matrix
- `BoxPR_curve.png` — Precision-Recall curve
- `BoxF1_curve.png` — F1 score curve
- Validation batch predictions vs ground truth

---

## 🔧 Configuration

### BoT-SORT Tracker (`custom_botsort.yaml`)

| Parameter | Value | Purpose |
|---|---|---|
| `track_buffer` | 120 | Frames to remember occluded targets |
| `track_high_thresh` | 0.3 | High confidence detection threshold |
| `new_track_thresh` | 0.6 | Minimum confidence to create new track |
| `match_thresh` | 0.95 | IoU matching threshold |
| `with_reid` | true | Enable Re-Identification features |
| `gmc_method` | sparseOptFlow | Global Motion Compensation method |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Object Detection | [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (v12) |
| Multi-Object Tracking | BoT-SORT with ReID |
| Video Processing | OpenCV |
| Web Interface | Streamlit |
| Dataset Annotation | [Roboflow](https://roboflow.com/) |
| Model Training | Kaggle (GPU P100) |
| Language | Python 3.x |

---

## 👥 Team

**TCS Industry Project** — Suspicious Activity Detection  
Third Year, College Industry Project

---

## 📄 License

This project is developed as part of an academic industry collaboration with TCS.  
The hoodie detection dataset is licensed under **CC BY 4.0**.
