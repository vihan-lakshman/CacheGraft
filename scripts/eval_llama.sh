#!/usr/bin/env bash
# Evaluate a trained Llama-1B-Instruct <- Llama-8B-Instruct fuser on the RAG tasks, reusing the
# dataset-generic harness via env overrides (common.py reads C2C_RECEIVER/SHARER/FUSER_DIR).
# Paths are derived from the script location, so it is machine-portable: just have the venv active.
#
#   C2C_FUSER_DIR=/path/to/<run>/final ./eval_llama.sh obqa    # OpenBookQA: no-retrieval + retrieved
#   C2C_FUSER_DIR=/path/to/<run>/final ./eval_llama.sh squad   # SQuAD-RAG: text_rag + c2c + judge
#   ./eval_llama.sh squad                                       # uses the default fuser dir below
# C2C_FUSER_DIR may also be a Hugging Face snapshot dir (snapshot_download of a published fuser).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # scripts/
REPO="$(cd "$HERE/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-$REPO/C2C}"             # the cloned C2C package (rosetta)
export C2C_RECEIVER="meta-llama/Llama-3.2-1B-Instruct"
export C2C_SHARER="meta-llama/Llama-3.1-8B-Instruct"
# Default to the 150k fuser under the C2C clone; override by exporting C2C_FUSER_DIR before calling.
export C2C_FUSER_DIR="${C2C_FUSER_DIR:-$REPO/C2C/local/checkpoints/llama1b_inst+8b_inst_150k/final}"
export USE_TF=0 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false

if [ ! -d "$C2C_FUSER_DIR" ]; then echo "fuser checkpoint not found: $C2C_FUSER_DIR" >&2; exit 1; fi
cd "$HERE"
task="${1:-obqa}"

if [ "$task" = "obqa" ]; then
  # Headline C2C test (no retrieval): does fusing the 8B's KV lift the 1B toward the 8B?
  python rag_obqa.py --method receiver --fact none        # 1B-Instruct floor
  python rag_obqa.py --method c2c_rag  --fact none         # C2C fusion (8B KV -> 1B)
  python rag_obqa.py --method sharer   --fact none         # 8B-Instruct ceiling
  # OpenBookQA-RAG (retrieved fact): text channel vs fusion at identical retrieval
  python rag_obqa.py --method text_rag --fact retrieved    # 1B reads the fact as TEXT
  python rag_obqa.py --method c2c_rag  --fact retrieved    # fuse [fact+Q+choices]
elif [ "$task" = "squad" ] || [ "$task" = "trivia" ]; then
  python rag_trivia.py --data "$task" --method receiver --fact none
  python rag_trivia.py --data "$task" --method text_rag --fact retrieved
  python rag_trivia.py --data "$task" --method text_rag --fact oracle
  python rag_trivia.py --data "$task" --method c2c --fact retrieved --fuse doc_only
  python rag_trivia.py --data "$task" --method c2c --fact oracle    --fuse doc_only
  python judge_trivia.py --data "$task" text_rag_retrieved c2c_retrieved_doc_only \
                         text_rag_oracle c2c_oracle_doc_only --model haiku
else
  echo "unknown task: $task (use 'obqa', 'squad', or 'trivia')" >&2; exit 1
fi
