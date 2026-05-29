# 🛡️ Suspicious Activity Detection in Retail Environments

> surveillance system for detecting shoplifting indicators in retail CCTV footage using YOLO object detection, Pose Estimation, and BoT-SORT multi-object tracking.

---

## 📌 Problem Statement

Retail stores face significant losses due to shoplifting. Traditional CCTV monitoring relies entirely on human operators who cannot watch every camera feed simultaneously. This project aims to build an **intelligent video analytics system** that automatically detects suspicious behavioral indicators such as:

- Persons wearing **hoodies** (face concealment)
- Detecting high-risk **bags and carriers** (backpacks/handbags) used for shoplifting
- **Customer trajectory tracking** for movement analysis
- **Action Recognition: Pocketing Items** (Proof-of-Concept for tracking items being hidden in pockets)

---

## 🏗️ System Architecture

The system is built around **four independent modules**, each targeting a specific shoplifting indicator:

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                          Streamlit Web Interface                         │
│                     (Upload Video → View Detections)                     │
├───────────────┬───────────────┬─────────────────────┬────────────────────┤
│   Module 1    │   Module 2    │     Module 3        │      Module 4      │
│   Hoodie      │ Bag Detection │     Tracking        │      Pocketing     │
├───────────────┼───────────────┼─────────────────────┼────────────────────┤
│ Custom YOLO12n│ YOLO12n       │ YOLO12n +           │ YOLO11s (Object) + │
│ (best.pt)     │ (Object)      │ BoT-SORT (ReID)     │ YOLO11n-Pose       │
└───────────────┴───────────────┴─────────────────────┴────────────────────┘
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
| **Training** | 300 epochs, Kaggle GPU |
| **Weights** | `Model/Hoodie-detection/weights/best.pt` |

### 2. Receptacle (Bag) Detection

Detects common shoplifting receptacles to alert security to individuals carrying items capable of concealing large amounts of stolen goods.

| Detail | Value |
|---|---|
| **Model**      | Pretrained YOLO12n |
| **Target Classes**| COCO classes: `backpack` (24), `handbag` (26), `suitcase` (28) |
| **Logic**      | Draws real-time bounding boxes and confidence scores around detected receptacles. |

### 3. Customer Tracking (BoT-SORT)

Tracks individual customers across frames with persistent IDs for movement analysis.

| Detail | Value |
|---|---|
| **Tracker** | BoT-SORT with ReID |
| **Track Buffer** | 120 frames (~4-5 sec memory on occlusion) |
| **Target Class** | `person` (COCO class 0) |

### 4. Action Recognition: Item Pocketing (Proof-of-Concept)

Fuses Object Detection and Pose Estimation to detect the action of hiding an item in a pocket. 
*Note: We utilize Cell Phones as our Proof-of-Concept (PoC) target because training a custom model for thousands of unique retail items (e.g., cosmetics, snacks) requires massive proprietary datasets. This module proves the tracking logic is scalable to any store's custom item weights.*

| Detail | Value |
|---|---|
| **Pose Model** | YOLO11n-Pose (Tracks Wrist & Hip Keypoints) |
| **Object Model**| YOLO11s (Tracks Cell Phones with high sensitivity `conf=0.15`) |
| **Logic**      | Tracks if wrist overlaps with object, retains 60-frame memory, triggers if wrist enters pocket area. |

---

## 📁 Project Structure

```text
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
├── bags.py                         # Bag/Receptacle detection script
├── advanced_botsort.py             # Standalone customer tracking script
├── pocket_detection.py             # Dual-model item pocketing detection
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

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Option 1: Streamlit Web Interface

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

# Bag/Receptacle Detection
python bags.py

# Customer Tracking (BoT-SORT)
python advanced_botsort.py

# Item Pocketing Action Recognition
python pocket_detection.py
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Object Detection | Ultralytics YOLO (v11/v12) |
| Pose Estimation | Ultralytics YOLO-Pose |
| Multi-Object Tracking | BoT-SORT with ReID |
| Video Processing | OpenCV |
| Web Interface | Streamlit |
| Language | Python 3.x |

---

## 👥 Team

**TCS Industry Project** — Suspicious Activity Detection  
Third Year, College Industry Project
