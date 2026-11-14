# Experimental Methodology & Evaluation Protocol

## 1. Principles of Experimental Validity

To ensure rigorous and defensible scientific outcomes:
1. **Pretrained Baseline Evaluation:** All initial evaluations utilize released pretrained weights. No fine-tuning or weight adaptation is performed at this stage.
2. **Zero Data Leakage:** Video splits are partitioned at the **source video ID** level. All frames, recompressed versions (`c23`, `c40`), and manipulated counterparts derived from original video `003` reside strictly in the same split.
3. **Transparent Metric Attribution:** Published benchmark metrics from original papers are clearly distinguished from empirical results generated on our Kaggle test partitions.

---

## 2. Evaluation Protocol by Component

### Stage 1: SCRFD Face Detection
* **Detection Coverage Rate:**
  $$\text{Coverage} = \frac{\text{Frames with at least 1 detected face}}{\text{Total extracted frames}}$$
* **Confidence Distribution:** Mean and standard deviation of detection confidences across manipulation categories and compression levels.
* **Inference Speed:** Average latency per frame in milliseconds on NVIDIA GPU.

### Stage 2: ArcFace Actor Recognition & Verification Proxy Protocol
Because FaceForensics++ does not provide official registered-actor labels, we implement a defensible proxy protocol:
1. **Gallery Enrollment:** 5 pristine frames of known original video IDs (e.g. `000`, `001`, `002`) are enrolled as pseudo-registered actors.
2. **Genuine Testing:**
   * Unseen frames from the same pristine video.
   * Reenacted frames (`Face2Face`, `NeuralTextures`) where target face identity is preserved.
3. **Impostor Testing:** Frames featuring different individuals.
4. **Identity Transfer Analysis:** For face-swap videos ($A\_B$), evaluate whether ArcFace recognizes donor $B$ (source) or retains target $A$.
5. **Metrics:**
   * Cosine Similarity distributions (Genuine vs. Impostor).
   * True Accept Rate (TAR) at fixed False Accept Rates (FAR = 0.01, 0.05).
   * Equal Error Rate (EER) and ROC-AUC.

### Stage 3: GenD Manipulation Detection
* **Frame-Level Evaluation:** Frame-level probability of manipulation $p(\text{Fake})$.
* **Video-Level Aggregation:**
  $$\hat{y}_{\text{video}} = \text{TrimmedMean}_{20\%}\left(\{p_i\}_{i=1}^N\right)$$
* **Metrics:**
  * Area Under ROC Curve (ROC-AUC).
  * Area Under Precision-Recall Curve (PR-AUC).
  * Accuracy, Precision, Recall, and F1-score at threshold $\tau = 0.50$.
  * False Positive Rate on authentic videos.
* **Comparative Backbones:**
  * GenD with CLIP ViT-L/14
  * GenD with Perception Encoder (PE)
  * GenD with DINOv3 ViT-L/16

### Stage 4: End-to-End Pipeline
* Total time per video across extraction, detection, verification, and manipulation scoring.
* Cascade error rate: Percentage of videos where face detection failure leads to unassessed manipulation.
