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

# Model pair + fuser are env-overridable so the same harness can run a locally-trained fuser for
# a different pair (e.g. Llama-1B receiver + Llama-8B sharer) without touching the Qwen defaults:
#   C2C_RECEIVER   - base/receiver model id          (default Qwen3-0.6B)
#   C2C_SHARER     - teacher/sharer model id         (default Qwen3-4B-Base)
#   C2C_FUSER_DIR  - path to a trained projector dir (the '.../final' checkpoint dir). If set,
#                    get_fuser_dir() returns it directly instead of pulling the released Qwen fuser.
RECEIVER = os.environ.get("C2C_RECEIVER", "Qwen/Qwen3-0.6B")
SHARER_BASE = os.environ.get("C2C_SHARER", "Qwen/Qwen3-4B-Base")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
FUSER_REPO = "nics-efc/C2C_Fuser"
FUSER_SUBFOLDER = "qwen3_0.6b+qwen3_4b_base_Fuser"

def get_fuser_dir(subfolder: str = FUSER_SUBFOLDER) -> str:
    """Return the fuser projector dir. A locally-trained checkpoint via C2C_FUSER_DIR takes
    precedence; otherwise download the released C2C fuser and return its '<subfolder>/final' dir."""
    local = os.environ.get("C2C_FUSER_DIR")
    if local:
        return local
    d = snapshot_download(repo_id=FUSER_REPO, allow_patterns=[f"{subfolder}/*"])
    return os.path.join(d, subfolder, "final")

_HERE = os.path.dirname(os.path.abspath(__file__))

# Model-pair tag. The eval sets (<data>_data.json / _corpus_emb.npy) are built by bge-small and
# are model-independent, so every pair SHARES them. Everything a model pair produces -- results,
# KV caches, distilled summaries -- is suffixed with this tag, so a second pair (e.g. the Llama
# receiver/sharer via C2C_RECEIVER/C2C_SHARER/C2C_FUSER_DIR) can be evaluated side by side
# without overwriting the first. Empty by default => the documented Qwen paths are unchanged.
RUN_TAG = os.environ.get("C2C_RUN_TAG", "")


def paths(data: str = "trivia") -> dict:
    """Per-dataset file locations, so the RAG harness is dataset-generic. A dataset just needs a
    `<data>_data.json` ({questions:[{q,aliases,norm_aliases,gold_ids}], corpus:[...]}) and a
    `<data>_corpus_emb.npy`; results land in `<data>_results<tag>/`, KV caches in
    `<data>_kvcache<tag>/`, offline summaries in `<data>_distilled<tag>*.json` (see RUN_TAG)."""
    return {"data": os.path.join(_HERE, f"{data}_data.json"),
            "emb": os.path.join(_HERE, f"{data}_corpus_emb.npy"),
            "results": os.path.join(_HERE, f"{data}_results{RUN_TAG}"),
            "kvcache": os.path.join(_HERE, f"{data}_kvcache{RUN_TAG}"),
            "distill_prefix": os.path.join(_HERE, f"{data}_distilled{RUN_TAG}")}


# ------------------------------------------------------------------------------------------
# Determinism. Decoding is greedy everywhere, so there is no sampling seed to set; the
# observed run-to-run drift (up to ~2.6 points on a TriviaQA cell) comes from nondeterministic
# CUDA kernels when batch composition differs between processes. This pins the kernels as far
# as PyTorch allows. It does NOT make the LLM judge deterministic -- judge verdicts are cached
# on disk as <tag>.judge*.json and should never be recomputed for a cell that already has them.
# ------------------------------------------------------------------------------------------
SEED = 0
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # read at cuBLAS init


def set_determinism(seed: int = SEED):
    import random
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # warn_only: a few fused attention kernels have no deterministic implementation; we prefer
    # running with a warning over refusing to run.
    torch.use_deterministic_algorithms(True, warn_only=True)


def top1(qemb, cemb, questions):
    """Top-1 retrieval. If a question carries `pool_ids` (built by prep.py --scope document),
    retrieval is restricted to the chunks of its own source document -- the realistic setting
    for question-answering over a known paper, and the one that makes paragraph-level retrieval
    tractable at all on QASPER/CovidQA. Otherwise it is argmax over the whole corpus."""
    import numpy as np
    if not questions or "pool_ids" not in questions[0]:
        return [int(i) for i in (qemb @ cemb.T).argmax(1)]
    out = []
    for i, q in enumerate(questions):
        pool = q["pool_ids"] or list(range(len(cemb)))
        sims = qemb[i] @ cemb[np.asarray(pool)].T
        out.append(int(pool[int(sims.argmax())]))
    return out
