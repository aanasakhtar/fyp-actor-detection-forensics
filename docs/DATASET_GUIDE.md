# FaceForensics++ (FF++) Dataset Guide & Ingestion Protocol

## 1. Dataset Organization

FaceForensics++ (FF++) is organized into authentic source videos and manipulated sequences generated using distinct synthesis approaches:

```
FaceForensics++/
├── original_sequences/
│   ├── youtube/
│   │   ├── c23/videos/ (1000 original videos)
│   │   └── c40/videos/
│   └── actors/ (DeepFakeDetection pristine actors)
└── manipulated_sequences/
    ├── Deepfakes/c23/videos/ (1000 face swapped videos)
    ├── Face2Face/c23/videos/ (1000 reenacted videos)
    ├── FaceSwap/c23/videos/ (1000 face swapped videos)
    ├── NeuralTextures/c23/videos/ (1000 facial reenacted videos)
    └── FaceShifter/c23/videos/ (1000 face swapped videos)
```

---

## 2. Source-Target Pair Nomenclature

In FaceForensics++, manipulated video filenames follow the pattern:
$$\text{<target\_id>}\_\text{<source\_id>}.mp4 \quad (\text{e.g., } 003\_045.mp4)$$

* **Target video (`target_id`):** The background video and body performing the action.
* **Source video (`source_id`):** The face donor.
* **Manipulation type mechanics:**
  * **FaceSwap / Deepfakes / FaceShifter:** The donor face appearance (`source_id`) replaces the target face (`target_id`).
  * **Face2Face / NeuralTextures:** The target face (`target_id`) is driven by the expressions of `source_id`. The facial identity remains `target_id`.

---

## 3. Recommended Pilot Subset for Kaggle

Downloading the complete multi-terabyte FF++ dataset inside a 9-hour Kaggle session is impractical. The recommended pilot subset provides complete coverage while keeping disk usage under 2 GB:

| Category | Compression | Video Count | Estimated Size |
| :--- | :--- | :--- | :--- |
| `original` | `c23` | 10 videos | ~150 MB |
| `Deepfakes` | `c23` | 10 videos | ~150 MB |
| `Face2Face` | `c23` | 10 videos | ~150 MB |
| `FaceSwap` | `c23` | 10 videos | ~150 MB |
| `NeuralTextures` | `c23` | 10 videos | ~150 MB |
| `FaceShifter` | `c23` | 10 videos | ~150 MB |
| `original` | `c40` | 5 videos (compression test) | ~40 MB |
| `Deepfakes` | `c40` | 5 videos (compression test) | ~40 MB |
| **Total Pilot** | - | **70 videos** | **~1.0 GB** |

---

## 4. Download Automation & Non-Interactive Execution

The official script `faceforensics_download_v4.py` requires accepting the Terms of Service. In Kaggle, automated non-interactive runs must pipe an empty newline into the script:

```bash
# Automated download of Face2Face c23 subset from EU2 server
python -c "import sys; sys.stdout.write('\n')" | python scripts/download_ffpp.py \
    /kaggle/working/data \
    -d Face2Face \
    -c c23 \
    -n 10 \
    --server EU2
```

Alternatively, the pilot subset can be created once as a private Kaggle Dataset (`faceforensics-pilot-c23`) and attached to the notebook under `/kaggle/input/faceforensics-pilot/` for instant zero-download startup.
