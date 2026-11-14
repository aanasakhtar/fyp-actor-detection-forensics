# Pipeline Architecture & Data Flow Specification

## 1. Pipeline Overview

The evaluation pipeline processes FaceForensics++ (FF++) video files to perform face detection, registered actor verification, and manipulation forensics. It aggregates frame-level predictions into robust video-level decisions.

```mermaid
flowchart TD
    Video["Input Video (.mp4)"] --> VR["VideoReader (OpenCV)"]
    VR -->|Uniform Sampled Frames| SCRFD["SCRFD 10G KPS Detector"]
    SCRFD -->|BBoxes & 5 Landmarks| Tracker["IoU Face Tracker"]
    
    Tracker -->|Tracked Face Instances| DualCrop{"Dual Crop Generator"}
    
    DualCrop -->|5-Point Affine 112x112| ArcFace["ArcFace Recognition Model"]
    DualCrop -->|1.3x Context Box 224x224| GenD["GenD Forensics Model"]
    
    ArcFace -->|512-d L2 Embedding| Matcher["Actor Gallery Matcher (Cosine Sim)"]
    GenD -->|p(Fake) Logit| Aggregator["Temporal Score Aggregator"]
    
    Matcher --> ResultAggregator["Case Evidence & Report Builder"]
    Aggregator --> ResultAggregator
    
    ResultAggregator --> JSON["Results Record (.json)"]
    ResultAggregator --> Plot["Evidence Visualizations"]
```

---

## 2. Component Specifications

### 2.1 Video Reader & Frame Extraction
* Decodes video streams via OpenCV `cv2.VideoCapture`.
* Applies uniform temporal sampling (default: up to 32 frames per video or 2 FPS) to ensure efficient Kaggle execution while capturing temporal dynamics.
* Preserves frame indices and presentation timestamps (PTS).

### 2.2 Face Detection & Multi-Face Tracking (SCRFD)
* Runs SCRFD 10G KPS on downscaled or original resolution frames (default: $640 \times 640$).
* Produces bounding boxes $[x_1, y_1, x_2, y_2]$, confidence scores, and 5 facial keypoints.
* Integrates a lightweight IoU tracker across consecutive sampled frames to maintain face instance identity (tracks) across multi-face video sequences.

### 2.3 Dual-Crop Strategy
* **ArcFace Path:** Uses standard similarity transformation mapping the 5 detected keypoints to a canonical $112 \times 112$ template.
* **GenD Path:** Calculates face center and dimensions, applies a $1.3\times$ scale expansion factor, pads when boundary overflows occur, and resizes to $224 \times 224$.

### 2.4 Actor Verification (ArcFace)
* Extracts 512-dimensional feature embedding vectors normalized to unit length.
* Compares incoming embeddings against registered actor prototypes in the enrolled gallery.
* Output classification:
  * $\text{Cosine Similarity} \ge \tau_{\text{match}} \implies \textbf{Possible Match}$
  * $\tau_{\text{uncertain}} \le \text{Cosine Similarity} < \tau_{\text{match}} \implies \textbf{Uncertain}$
  * $\text{Cosine Similarity} < \tau_{\text{uncertain}} \implies \textbf{Unknown}$

### 2.5 Manipulation Detection (GenD)
* Runs inference across candidate backbones (`CLIP ViT-L/14`, `Perception Encoder`, `DINOv3 ViT-L/16`).
* Generates frame-level manipulation probability $p(\text{Fake}) \in [0, 1]$.
* Aggregates frame scores into a video-level manipulation prediction using trimmed mean or 80th-percentile pooling.

---

## 3. Resumption and Caching Architecture

To withstand Kaggle session timeouts and connection resets:
* Every processed video writes an individual JSON summary: `predictions/<category>/<compression>/<video_id>.json`.
* Before decoding a video, the loader checks if the corresponding JSON cache exists.
* Completed stages are immediately skipped upon session restart.
