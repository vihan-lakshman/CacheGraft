"""Shared constants + helpers for the C2C-RAG experiments.

Models / fuser:
  Receiver : Qwen/Qwen3-0.6B  (small model that decodes the answer)
  Sharer   : Qwen/Qwen3-4B-Base  (large model whose KV-cache is fused in, frozen)
  Fuser    : nics-efc/C2C_Fuser :: qwen3_0.6b+qwen3_4b_base_Fuser
             (the released projector reproduced at OpenBookQA C2C = 53.4)
"""
import os
os.environ.setdefault("USE_TF", "0")  # avoid the system-TensorFlow import path in transformers
from huggingface_hub import snapshot_download

RECEIVER = "Qwen/Qwen3-0.6B"
SHARER_BASE = "Qwen/Qwen3-4B-Base"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
FUSER_REPO = "nics-efc/C2C_Fuser"
FUSER_SUBFOLDER = "qwen3_0.6b+qwen3_4b_base_Fuser"

def get_fuser_dir(subfolder: str = FUSER_SUBFOLDER) -> str:
    """Download the released C2C fuser projectors and return the '<subfolder>/final' dir."""
    d = snapshot_download(repo_id=FUSER_REPO, allow_patterns=[f"{subfolder}/*"])
    return os.path.join(d, subfolder, "final")
