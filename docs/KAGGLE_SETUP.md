# Kaggle Setup & Workflow Guide

## 1. Kaggle Environment Overview

* **Accelerator:** GPU T4 x 2 or GPU P100 (16 GB VRAM).
* **Storage Paths:**
  * Input directory (read-only): `/kaggle/input/`
  * Working directory (read/write): `/kaggle/working/`
* **Session Persistence:** Files in `/kaggle/working/` are ephemeral. Outputs must be saved to Kaggle notebook output artifacts or bundled for download.

---

## 2. Setting Up the Repository in Kaggle

In the first cell of your Kaggle Notebook:

```python
import os
import sys

# 1. Clone repository from GitHub (private repo via personal access token or public release)
!git clone https://github.com/aanasakhtar/fyp-actor-detection-forensics.git /kaggle/working/fyp-actor-detection-forensics

# 2. Add repository source to Python path
sys.path.append("/kaggle/working/fyp-actor-detection-forensics/src")
%cd /kaggle/working/fyp-actor-detection-forensics

# 3. Install required packages
!pip install -q insightface onnxruntime-gpu timm transformers
!pip install -q -e .
```

---

## 3. Dataset Configuration

### Option A: Direct Download in Kaggle (EU2 Server)
```python
# Download pilot subset using automated script
!python scripts/inspect_dataset.py --download-pilot --output-dir /kaggle/working/data
```

### Option B: Attached Kaggle Dataset (Recommended)
If attached under `/kaggle/input/faceforensics-pilot/`:
```python
DATA_ROOT = "/kaggle/input/faceforensics-pilot"
```

---

## 4. Running the Pipeline & Resuming Sessions

To process videos and save intermediate predictions:

```python
from pipeline.utils.device import get_device_info
from pipeline.data.dataset_loader import FFppDatasetLoader

print(get_device_info())
loader = FFppDatasetLoader(data_root=DATA_ROOT)
print(f"Discovered {len(loader)} videos across categories.")
```

All predictions are written to `/kaggle/working/predictions/`. If your session is interrupted, re-running the cell will skip already processed videos.

---

## 5. Exporting Results

At the conclusion of the notebook run:

```bash
# Bundle evaluation results, plots, and metrics into a single zip
!zip -r /kaggle/working/evaluation_results.zip /kaggle/working/results /kaggle/working/predictions
```
Download `evaluation_results.zip` directly from the Kaggle right-hand Output pane to your local machine for reporting.
