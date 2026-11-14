# Model Availability, Checkpoints, and Licensing Report

## 1. Overview

This document presents the detailed investigation into the official repositories, model checkpoints, input/output specifications, inference requirements, and licensing constraints for the shortlisted models in the FYP computer vision pipeline.

---

## 2. Model 1: SCRFD (Sample and Computation Redistribution Face Detection)

* **Architecture:** SCRFD 10G KPS (RetinaFace-10GF compute budget with Keypoints Prediction).
* **Official Repository:** `https://github.com/deepinsight/insightface/tree/master/detection/scrfd`
* **Checkpoint Sources:**
  * InsightFace Model Zoo (`buffalo_l` pack contains `det_10g.onnx`).
  * Direct ONNX files: `scrfd_10g_bnkps.onnx`.
* **Input Specifications:**
  * Image format: BGR format, standard dimensions $640 \times 640$ (dynamic shapes supported).
  * Normalization: $(x - 127.5) / 128.0$.
* **Output Specifications:**
  * Bounding boxes: $[x_1, y_1, x_2, y_2, \text{confidence}]$.
  * 5 Facial Keypoints: Left eye, right eye, nose tip, left mouth corner, right mouth corner.
* **Dependencies:** `insightface`, `onnxruntime` or `onnxruntime-gpu`, `opencv-python`.
* **Hardware & Runtime:** Light footprint (~1.2 GB VRAM on GPU, <15 ms per frame inference on NVIDIA T4).
* **License & Commercial Constraints:** InsightFace model weights are released strictly for **Academic Research and Non-Commercial Use**. Commercial deployment requires custom retraining on open-licensed datasets.

---

## 3. Model 2: ArcFace (Additive Angular Margin Loss)

* **Architecture:** ArcFace ResNet-50 (`w600k_r50.onnx`) or ResNet-100 (`glintr100.onnx`).
* **Official Repository:** `https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch`
* **Checkpoint Sources:**
  * Bundled in InsightFace `buffalo_l` distribution (`w600k_r50.onnx`, trained on WebFace600K).
* **Input Specifications:**
  * Aligned face crop: strictly $112 \times 112$ pixels.
  * Alignment algorithm: 5-point similarity transformation (`norm_crop` against standard template).
  * Normalization: $(x - 127.5) / 127.5$.
* **Output Specifications:**
  * 512-dimensional feature embedding vector, normalized to unit $L_2$ sphere ($\|v\|_2 = 1.0$).
* **Metric:** Cosine similarity via dot product $\mathbf{u} \cdot \mathbf{v} \in [-1.0, 1.0]$.
* **Dependencies:** `insightface`, `onnxruntime-gpu`, `scipy`.
* **License & Constraints:** InsightFace weights carry non-commercial research licensing.

---

## 4. Model 3: GenD (Generalized Deepfake Detection)

* **Reference:** Yermakov et al., "Deepfake Detection that Generalizes Across Benchmarks", WACV 2026.
* **Official Repositories:**
  * Code: `https://github.com/yermandy/GenD` and `https://github.com/yermandy/deepfake-detection`
  * Model Checkpoints: Hugging Face Collection `yermandy/gend`
* **Evaluated Backbones & Checkpoints:**
  1. `yermandy/GenD_CLIP_L_14`: OpenAI CLIP ViT-L/14 visual encoder backbone.
  2. `yermandy/GenD_PE_L`: Perception Encoder Large backbone.
  3. `yermandy/GenD_DINOv3_L`: Meta DINOv3 ViT-L/16 backbone.
* **Weight Nature & Training Stage:**
  * **Critical Distinction:** These released checkpoints are **complete, fully trained deepfake detectors**.
  * Parameter-efficient fine-tuning (PEFT) was applied to the foundational vision backbones (specifically LayerNorm weights, comprising ~0.03% of parameters) alongside a hyperspherical classification head trained on deepfake benchmarks.
  * Untouched foundation vision backbones (e.g. raw CLIP or raw DINOv3 without detector heads) produce image embeddings, not real/fake probability scores.
* **Input Specifications:**
  * Contextual face crop: $1.3\times$ enlarged bounding box around the face.
  * Input resolution: $224 \times 224$ pixels.
  * Image normalization: Standard ImageNet or CLIP mean/std.
* **Output Specifications:**
  * Binary logit / softmax probability: $p(\text{Real}) = \text{index } 0$, $p(\text{Fake}) = \text{index } 1$.
* **Hardware & Runtime:**
  * VRAM: ~3.5 to 6.0 GB per model (comfortably fits on Kaggle 16 GB T4 or P100).
* **Licensing:**
  * GenD code: Open-source academic research.
  * CLIP: MIT License.
  * DINOv3: Meta Research License (non-commercial restrictions may apply).

---

## 5. Comparative Compatibility Summary

| Stage | Model | Input Dim | Alignment / Crop Strategy | Execution Provider |
| :--- | :--- | :--- | :--- | :--- |
| Detection | SCRFD 10G KPS | $640 \times 640$ | Full frame | ONNX Runtime GPU (CUDA) |
| Recognition | ArcFace r50 | $112 \times 112$ | 5-point Similarity Transform | ONNX Runtime GPU (CUDA) |
| Forensics | GenD (CLIP/PE/DINO) | $224 \times 224$ | $1.3\times$ Expanded Bounding Box | PyTorch / Transformers (CUDA) |
