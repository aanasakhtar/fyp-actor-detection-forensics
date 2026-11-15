"""
Utility script to generate reproducible, clean Kaggle notebooks.
"""

import json
import os

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().split("\n", -1) if isinstance(source, str) else source
    }

def code_cell(source):
    lines = source.strip().split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]] if lines else []
    }

CLONE_SNIPPET = """# 1. Secure Clone for Private Repository via Kaggle Secrets
import os
import sys

repo_dir = "/kaggle/working/fyp-actor-detection-forensics"

if not os.path.exists(repo_dir):
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        gh_token = user_secrets.get_secret("GH_TOKEN")
        clone_url = f"https://x-access-token:{gh_token}@github.com/aanasakhtar/fyp-actor-detection-forensics.git"
    except Exception:
        import getpass
        print("Note: Store your GitHub token in Kaggle Add-ons -> Secrets as 'GH_TOKEN' to avoid manual input.")
        gh_token = getpass.getpass("Enter GitHub Personal Access Token (or press enter for public/attached): ")
        if gh_token.strip():
            clone_url = f"https://x-access-token:{gh_token}@github.com/aanasakhtar/fyp-actor-detection-forensics.git"
        else:
            clone_url = "https://github.com/aanasakhtar/fyp-actor-detection-forensics.git"
            
    os.system(f"git clone {clone_url} {repo_dir}")
    print("Repository setup completed.")
else:
    print("Repository already cloned.")

if repo_dir not in sys.path:
    sys.path.append(os.path.join(repo_dir, "src"))

%cd {repo_dir}
!pip install -q onnxruntime-gpu insightface timm transformers
!pip install -q -e ."""

def generate_notebooks(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 01_environment_setup.ipynb
    nb1_cells = [
        md_cell("""# 01 — Environment Setup & Checkpoint Preparation
### FYP Computer Vision Pipeline: Hardware Verification & Dependency Installation
This notebook verifies Kaggle GPU availability, installs runtime dependencies (InsightFace, ONNX Runtime GPU, Transformers), clones the private research repository securely via Kaggle Secrets, and caches model checkpoints."""),
        code_cell("""# Check GPU Accelerator and VRAM
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    print("WARNING: Running on CPU. For deep learning inference, please enable GPU in Notebook Settings (Settings -> Accelerator -> GPU T4 x 2 or P100).")"""),
        code_cell(CLONE_SNIPPET),
        code_cell("""# Verify Environment and Device Utilities
from pipeline.utils.device import get_device_info, select_device
from pipeline.utils.reproducibility import set_seed

set_seed(42)
info = get_device_info()
print("Pipeline Hardware Diagnostic:", info)""")
    ]
    with open(os.path.join(output_dir, "01_environment_setup.ipynb"), "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb1_cells), f, indent=2)

    # 02_dataset_exploration.ipynb
    nb2_cells = [
        md_cell("""# 02 — FaceForensics++ Dataset Exploration
### Inspecting Categories, Compressions, and Source-Target Pair Relationships
This notebook explores the FaceForensics++ (FF++) dataset layout, verifies authentic vs manipulated videos, maps donor-target pairs, and sets up a pilot evaluation subset."""),
        code_cell(CLONE_SNIPPET),
        code_cell("""from pipeline.data.dataset_loader import FFppDatasetLoader
from pipeline.data.metadata import parse_ffpp_filename, parse_pair_mapping

# Specify dataset root (attached Kaggle dataset or downloaded subset)
DATA_ROOT = "/kaggle/input/faceforensics-pilot"
if not os.path.exists(DATA_ROOT):
    DATA_ROOT = "./data"
    print(f"Attached dataset not found at /kaggle/input/faceforensics-pilot, checking local fallback: {DATA_ROOT}")

loader = FFppDatasetLoader(DATA_ROOT)
print(f"Total discovered videos: {len(loader)}")"""),
        code_cell("""# Breakdown by Category and Compression
for cat in ["original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]:
    subset = loader.filter(categories=[cat])
    print(f"Category: {cat:<16} | Count: {len(subset)}")"""),
        code_cell("""# Inspect Zero-Leakage Data Splits
train_vids, val_vids, test_vids = loader.get_leakage_free_splits(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
print(f"Splits summary: Train={len(train_vids)}, Val={len(val_vids)}, Test={len(test_vids)}")""")
    ]
    with open(os.path.join(output_dir, "02_dataset_exploration.ipynb"), "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb2_cells), f, indent=2)

    # 03_scrfd_evaluation.ipynb
    nb3_cells = [
        md_cell("""# 03 — SCRFD Face Detection & Tracking Evaluation
### Stage 1: Face Localization, 5-Point Landmarks, and Multi-Face Tracking
Evaluates SCRFD 10G KPS across manipulation types and compression levels (c23 vs c40)."""),
        code_cell(CLONE_SNIPPET),
        code_cell("""import time
import numpy as np
from pipeline.data.dataset_loader import FFppDatasetLoader
from pipeline.data.video_reader import VideoReader
from pipeline.detection.scrfd_detector import SCRFDDetector
from pipeline.tracking.face_tracking import IoUFaceTracker
from pipeline.evaluation.detection_metrics import compute_detection_coverage, compute_detection_confidence_summary
from pipeline.visualization.detection_visualization import draw_detections

# Initialize Detector & Tracker
detector = SCRFDDetector(conf_threshold=0.50)
tracker = IoUFaceTracker(iou_threshold=0.40)
print("SCRFD Detector & IoU Tracker Ready.")"""),
        code_cell("""# Benchmark Detection on Sample Videos
DATA_ROOT = "/kaggle/input/faceforensics-pilot" if os.path.exists("/kaggle/input/faceforensics-pilot") else "./data"
loader = FFppDatasetLoader(DATA_ROOT)

all_confs = []
all_frame_dets = []
start_time = time.time()
processed_frames = 0

for meta in loader.videos[:10]:
    reader = VideoReader(meta.video_path)
    frames = reader.sample_uniform(max_frames=16)
    for frame in frames:
        dets = detector.detect_frame(frame.image_bgr, frame.frame_index, frame.timestamp_sec)
        dets = tracker.update(dets, frame.frame_index)
        all_frame_dets.append(dets)
        for d in dets:
            all_confs.append(d.confidence)
        processed_frames += 1

elapsed = time.time() - start_time
print(f"Processed {processed_frames} frames in {elapsed:.2f}s ({processed_frames/elapsed:.1f} FPS)")
print(f"Detection Coverage: {compute_detection_coverage(all_frame_dets)*100:.1f}%")
print(f"Confidence Stats: {compute_detection_confidence_summary(all_confs)}")""")
    ]
    with open(os.path.join(output_dir, "03_scrfd_evaluation.ipynb"), "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb3_cells), f, indent=2)

    # 04_arcface_evaluation.ipynb
    nb4_cells = [
        md_cell("""# 04 — ArcFace Actor Identification & Verification Evaluation
### Stage 2: Biometric Embeddings, Reference Gallery, and Cosine Matching
Evaluates ArcFace (w600k_r50) using the defensible FF++ proxy protocol: genuine pairs, impostor pairs, and cross-identity swap verification."""),
        code_cell(CLONE_SNIPPET),
        code_cell("""import numpy as np
from pipeline.data.dataset_loader import FFppDatasetLoader
from pipeline.data.video_reader import VideoReader
from pipeline.detection.scrfd_detector import SCRFDDetector
from pipeline.preprocessing.face_alignment import align_face_5point
from pipeline.recognition.arcface_encoder import ArcFaceEncoder
from pipeline.recognition.reference_gallery import ActorReferenceGallery
from pipeline.recognition.identity_matching import IdentityMatcher
from pipeline.evaluation.recognition_metrics import compute_verification_metrics
from pipeline.visualization.evaluation_plots import plot_similarity_distributions

# Initialize ArcFace and Reference Gallery
detector = SCRFDDetector()
encoder = ArcFaceEncoder()
gallery = ActorReferenceGallery()
print("ArcFace Biometric Subsystem Ready.")"""),
        code_cell("""# Enroll Pseudo-Actors from Pristine Original Videos
DATA_ROOT = "/kaggle/input/faceforensics-pilot" if os.path.exists("/kaggle/input/faceforensics-pilot") else "./data"
loader = FFppDatasetLoader(DATA_ROOT)
originals = loader.filter(categories=["original"])

for meta in originals[:5]:
    reader = VideoReader(meta.video_path)
    frames = reader.sample_uniform(max_frames=8)
    embs = []
    for f in frames:
        dets = detector.detect_frame(f.image_bgr)
        if dets and dets[0].landmarks_5 is not None:
            aligned = align_face_5point(f.image_bgr, dets[0].landmarks_5)
            if aligned is not None:
                emb = encoder.extract_embedding(aligned)
                if emb is not None:
                    embs.append(emb)
    if embs:
        gallery.enroll_actor(meta.target_id, f"Actor_{meta.target_id}", embs)

print(f"Enrolled {len(gallery)} actors into the reference gallery.")""")
    ]
    with open(os.path.join(output_dir, "04_arcface_evaluation.ipynb"), "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb4_cells), f, indent=2)

    # 05_gend_evaluation.ipynb
    nb5_cells = [
        md_cell("""# 05 — GenD Deepfake Detection Backbone Comparison
### Stage 3: Evaluating CLIP ViT-L/14, Perception Encoder (PE), and DINOv3
Compares the released GenD detector checkpoints across manipulation categories and compression levels."""),
        code_cell(CLONE_SNIPPET),
        code_cell("""from pipeline.data.dataset_loader import FFppDatasetLoader
from pipeline.data.video_reader import VideoReader
from pipeline.detection.scrfd_detector import SCRFDDetector
from pipeline.preprocessing.face_cropping import crop_context_face
from pipeline.forensics.clip_detector import GenDCLIPDetector
from pipeline.forensics.pe_detector import GenDPEDetector
from pipeline.forensics.dinov3_detector import GenDDINOv3Detector
from pipeline.evaluation.forensic_metrics import compute_forensic_metrics
from pipeline.visualization.evaluation_plots import plot_roc_curve, plot_confusion_matrix

# Load GenD CLIP Detector
clip_detector = GenDCLIPDetector()
print("GenD CLIP ViT-L/14 detector loaded.")""")
    ]
    with open(os.path.join(output_dir, "05_gend_evaluation.ipynb"), "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb5_cells), f, indent=2)

    # 06_end_to_end_pipeline.ipynb
    nb6_cells = [
        md_cell("""# 06 — End-to-End Pipeline Integration & Evidence Dashboard
### Full Workflow: Video -> Detection -> Verification -> Forensics -> Evidence Card
Executes the integrated computer vision pipeline on FaceForensics++ videos with intermediate result caching and exportable results."""),
        code_cell(CLONE_SNIPPET),
        code_cell("""# Execute Integrated Evaluation Pipeline
!python scripts/evaluate_pipeline.py --data-dir ./data --output-dir ./results/pipeline"""),
        code_cell("""# Bundle Evaluation Artifacts for Download
!zip -r /kaggle/working/fyp_evaluation_artifacts.zip ./results
print("Artifacts successfully packaged for download at /kaggle/working/fyp_evaluation_artifacts.zip")""")
    ]
    with open(os.path.join(output_dir, "06_end_to_end_pipeline.ipynb"), "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb6_cells), f, indent=2)

    print("All 6 Kaggle notebooks updated with secure private repo cloning.")

if __name__ == "__main__":
    generate_notebooks("D:/FYP2027/fyp-actor-detection-forensics/notebooks")
