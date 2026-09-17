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

def paths(data: str = "trivia") -> dict:
    """Per-dataset file locations, so the RAG harness is dataset-generic. A dataset just needs a
    `<data>_data.json` ({questions:[{q,aliases,norm_aliases,gold_ids}], corpus:[...]}) and a
    `<data>_corpus_emb.npy`; results land in `<data>_results/`, KV caches in `<data>_kvcache/`."""
    return {"data": os.path.join(_HERE, f"{data}_data.json"),
            "emb": os.path.join(_HERE, f"{data}_corpus_emb.npy"),
            "results": os.path.join(_HERE, f"{data}_results"),
            "kvcache": os.path.join(_HERE, f"{data}_kvcache")}


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


# ------------------------------------------------------------------------------------------
# Per-question keys. Question TEXT is not unique (CovidQA and QASPER repeat generic questions
# across papers), so every per-question score file is keyed by "<index>\t<question>", the
# sample's position in <data>_data.json plus its text. Every runner writes samples in data order
# (questions[:n_test]), so index i is questions[i].
# ------------------------------------------------------------------------------------------
def item_key(i, q):
    return f"{i}\t{q}"


def gold_aliases(data_questions, samples, field="norm_aliases"):
    """Full alias list for each sample, by position. Fails loudly if the samples are not in
    data order, which would silently pair generations with the wrong gold answers."""
    out = []
    for i, s in enumerate(samples):
        if i >= len(data_questions) or data_questions[i]["q"] != s["q"]:
            raise SystemExit(f"sample {i} ({s['q'][:60]!r}) is not question {i} of the eval set; "
                             f"the results file does not match <data>_data.json")
        out.append([a for a in data_questions[i][field] if a and a.strip()])
    return out


def write_scores(path, samples, values):
    """Per-question scores (judge verdicts, BERTScore) keyed by item_key."""
    import json
    json.dump({item_key(i, s["q"]): v for i, s, v in zip(range(len(samples)), samples, values)},
              open(path, "w"))


def load_scores(path, samples):
    """Read a per-question score file as {item_key: value}. Files written before the index key
    was introduced are keyed by question text alone; they are converted when the text is unique,
    and rejected when it is not, since such a file kept only the LAST verdict per text."""
    import json
    d = json.load(open(path))
    if all("\t" in k for k in d):
        return d
    texts = [s["q"] for s in samples]
    if len(set(texts)) != len(texts):
        raise SystemExit(f"{path} is keyed by question text, but this eval set repeats question "
                         f"texts, so verdicts for the repeats were lost. Re-run the judge/scorer.")
    return {item_key(i, q): d[q] for i, q in enumerate(texts) if q in d}


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
