"""Download Qwen3-1.7B-Base from HuggingFace."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen3-1.7B-Base"
LOCAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "qwen3-1.7b-base")

if __name__ == "__main__":
    print(f"Downloading {MODEL_ID} to {LOCAL_DIR}...")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_DIR,
    )
    print(f"Done. Model saved to {LOCAL_DIR}")
