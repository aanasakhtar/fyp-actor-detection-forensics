# FYP Computer Vision Pipeline: Face Detection, Actor Verification, and Manipulation Forensics

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Environment: Kaggle](https://img.shields.io/badge/Environment-Kaggle%20GPU-orange.svg)](https://www.kaggle.com)

An experimental computer vision pipeline designed for the **Talent Marketplace and AI Likeness Monitoring System** Final Year Project (FYP). The pipeline implements and evaluates a three-stage workflow:
1. **Face Detection & Localization:** SCRFD (SCRFD 10G KPS)
2. **Actor Identification & Verification:** ArcFace (ResNet-50 on WebFace600K)
3. **Facial Manipulation Detection:** GenD (Generalized Deepfake Detection across CLIP, Perception Encoder, and DINOv3 backbones)

The codebase is engineered to run in **Kaggle Notebooks** with GPU acceleration (NVIDIA T4 / P100), keeping local development lightweight and reproducible.

---

## 1. System Architecture

The pipeline processes video content through sequential, decoupled stages with dual-path preprocessing:

```
[ Input Video (.mp4) ]
         │
         ▼
[ VideoReader: Uniform Temporal Frame Sampling ]
         │
         ▼
[ Stage 1: SCRFD 10G KPS Detection & IoU Multi-Face Tracking ]
         │
         ├─────────────────────────────────────────┐
         │ (5-Point Landmark Similarity)           │ (1.3x Context Bounding Box)
         ▼                                         ▼
[ Stage 2: ArcFace 112x112 Crop ]         [ Stage 3: GenD 224x224 Crop ]
[ Extract 512-d L2 Embedding ]            [ Encoders: CLIP / PE / DINOv3 ]
[ Actor Gallery Cosine Matching ]         [ Frame Alteration Probabilities ]
         │                                         │
         └────────────────────┬────────────────────┘
                              ▼
            [ Score Aggregator & Evidence Builder ]
                              │
                              ▼
           [ Output Record: JSON, CSV, Evidence Cards ]
```

---

## 2. Model Breakdown

| Stage | Model | Variant / Checkpoint | Input Specification | Output |
| :--- | :--- | :--- | :--- | :--- |
| **1. Face Detection** | SCRFD | `SCRFD 10G KPS` (`det_10g.onnx`) | $640 \times 640$ BGR image | BBoxes $[x_1, y_1, x_2, y_2, \text{conf}]$ + 5 landmarks |
| **2. Actor Verification** | ArcFace | `w600k_r50.onnx` (ResNet-50) | $112 \times 112$ 5-point aligned crop | 512-d normalized embedding ($\|v\|_2 = 1$) |
| **3. Deepfake Detection** | GenD | `GenD_CLIP_L_14`<br>`GenD_PE_L`<br>`GenD_DINOv3_L` | $224 \times 224$ $1.3\times$ context crop | Frame & video $p(\text{Fake}) \in [0, 1]$ |

*For complete licensing and checkpoint availability details, see [docs/MODEL_AVAILABILITY.md](docs/MODEL_AVAILABILITY.md).*

---

## 3. Repository Structure

```
fyp-actor-detection-forensics/
├── README.md                          # Project documentation
├── LICENSE                            # MIT License
├── pyproject.toml                     # Package specification
├── requirements.txt                   # Environment dependencies
├── configs/                           # Modular YAML configurations
│   ├── default.yaml
│   ├── dataset.yaml
│   ├── scrfd.yaml
│   ├── arcface.yaml
│   └── gend.yaml
├── docs/                              # Technical research & setup guides
│   ├── MODEL_AVAILABILITY.md
│   ├── PIPELINE_ARCHITECTURE.md
│   ├── DATASET_GUIDE.md
│   ├── EXPERIMENTAL_METHODOLOGY.md
│   └── KAGGLE_SETUP.md
├── notebooks/                         # Kaggle experiment notebooks
│   ├── 01_environment_setup.ipynb
│   ├── 02_dataset_exploration.ipynb
│   ├── 03_scrfd_evaluation.ipynb
│   ├── 04_arcface_evaluation.ipynb
│   ├── 05_gend_evaluation.ipynb
│   └── 06_end_to_end_pipeline.ipynb
├── scripts/                           # Standalone execution scripts
│   ├── inspect_dataset.py
│   ├── run_scrfd.py
│   ├── run_arcface.py
│   ├── run_gend.py
│   └── evaluate_pipeline.py
├── src/                               # Modular core library
│   └── pipeline/
│       ├── data/                      # Readers and metadata mappers
│       ├── preprocessing/             # Alignment and cropping
│       ├── detection/                 # SCRFD wrappers and IoU tools
│       ├── recognition/               # ArcFace encoder and gallery
│       ├── forensics/                 # GenD model backbones
│       ├── tracking/                  # IoU multi-face tracker
│       ├── evaluation/                # Detection, verification, forensic metrics
│       ├── visualization/             # Plotting and evidence cards
│       └── utils/                     # Device, logging, reproducibility
├── tests/                             # Unit tests
└── results/                           # Evaluation outputs and figures (.gitkeep)
```

---

## 4. Kaggle Quickstart Guide

### 1. Open a Kaggle Notebook
Set the notebook accelerator: **Settings -> Accelerator -> GPU T4 x 2** or **P100**.

### 2. Clone and Install
```python
!git clone https://github.com/aanasakhtar/fyp-actor-detection-forensics.git /kaggle/working/fyp-actor-detection-forensics
%cd /kaggle/working/fyp-actor-detection-forensics
!pip install -q onnxruntime-gpu insightface timm transformers
!pip install -q -e .
```

### 3. Attach Dataset
Attach your FaceForensics++ pilot dataset under `/kaggle/input/faceforensics-pilot/`.

### 4. Execute the Notebooks
Execute notebooks in sequence:
* `01_environment_setup.ipynb`: Diagnostic checks and checkpoint caching.
* `02_dataset_exploration.ipynb`: Dataset pair parsing and split generation.
* `03_scrfd_evaluation.ipynb`: Face detection coverage and tracking evaluation.
* `04_arcface_evaluation.ipynb`: Actor reference gallery matching and ROC calibration.
* `05_gend_evaluation.ipynb`: GenD backbone comparison (CLIP vs PE vs DINOv3).
* `06_end_to_end_pipeline.ipynb`: Full end-to-end video analysis and evidence export.

---

## 5. Local Development and Testing

Run the test suite locally:
```bash
pytest tests/
```
All 12 unit tests validate data split independence, IoU computation, face tracking, ArcFace prototype gallery logic, cosine matching, score aggregation, and forensic metrics.

---

## 6. Research References & Citations

1. **FaceForensics++:** Andreas Rössler et al., *FaceForensics++: Learning to Detect Manipulated Facial Images*, ICCV 2019.
2. **SCRFD:** Jia Guo et al., *Sample and Computation Redistribution for Efficient Face Detection*, 2021.
3. **ArcFace:** Jiankang Deng et al., *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*, CVPR 2019.
4. **GenD:** Andrii Yermakov, Jan Cech, Jiri Matas, Mario Fritz, *Deepfake Detection that Generalizes Across Benchmarks*, WACV 2026.
