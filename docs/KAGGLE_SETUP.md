# Kaggle Setup & Workflow Guide

## 1. Kaggle Environment Overview

* **Accelerator:** GPU T4 x 2 or GPU P100 (16 GB VRAM).
* **Storage Paths:**
  * Input directory (read-only): `/kaggle/input/`
  * Working directory (read/write): `/kaggle/working/`
* **Session Persistence:** Files in `/kaggle/working/` are ephemeral. Outputs must be saved to Kaggle notebook output artifacts or bundled for download.

---

## 2. Setting Up the Private Repository in Kaggle

Because `fyp-actor-detection-forensics` is a **private** repository, a standard unauthenticated `!git clone` will fail with an authentication error. Use one of the two secure methods below:

### Method A: Kaggle Secrets (Recommended & Secure)

Kaggle provides an encrypted secrets manager so your GitHub credentials are never hardcoded in code cells or committed.

1. **Generate a GitHub Personal Access Token (PAT):**
   * On GitHub: Go to **Settings** $\to$ **Developer Settings** $\to$ **Personal access tokens** (fine-grained or classic).
   * Grant **Contents: Read-only** access to `aanasakhtar/fyp-actor-detection-forensics` (or classic `repo` scope).
2. **Add to Kaggle Secrets:**
   * In your Kaggle Notebook, click **Add-ons** in the top menu $\to$ **Secrets**.
   * Click **Add a new secret**:
     * **Label:** `GH_TOKEN`
     * **Value:** *(paste your GitHub token)*
3. **Execute in the First Notebook Cell:**

```python
import os
import sys

# 1. Fetch GitHub Token securely from Kaggle Secrets
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    gh_token = user_secrets.get_secret("GH_TOKEN")
    clone_url = f"https://x-access-token:{gh_token}@github.com/aanasakhtar/fyp-actor-detection-forensics.git"
except Exception:
    import getpass
    gh_token = getpass.getpass("Enter GitHub Personal Access Token: ")
    clone_url = f"https://x-access-token:{gh_token}@github.com/aanasakhtar/fyp-actor-detection-forensics.git"

# 2. Clone repository into Kaggle working directory
repo_dir = "/kaggle/working/fyp-actor-detection-forensics"
if not os.path.exists(repo_dir):
    os.system(f"git clone {clone_url} {repo_dir}")
    print("Repository cloned successfully.")
else:
    print("Repository directory already present.")

# 3. Add to Python path and install in editable mode
sys.path.append(os.path.join(repo_dir, "src"))
%cd {repo_dir}
!pip install -q onnxruntime-gpu insightface timm transformers
!pip install -q -e .
```

---

### Method B: Private Kaggle Dataset (Zero-Token / Instant Startup)

If you prefer not to manage GitHub tokens inside Kaggle:
1. Package the repository locally:
   ```bash
   git archive --format=zip --output=fyp-pipeline-code.zip HEAD
   ```
2. In Kaggle, create a new private dataset titled `fyp-pipeline-code` and upload `fyp-pipeline-code.zip`.
3. In your notebook, click **+ Add Input** $\to$ select your dataset.
4. Mount directly:
   ```python
   import sys
   sys.path.append("/kaggle/input/fyp-pipeline-code/src")
   !pip install -q onnxruntime-gpu insightface timm transformers
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
