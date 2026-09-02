#!/bin/bash
# Run the full experiment matrix for one dataset.
#
#   ./run_matrix.sh <data> <n_test>
#   ./run_matrix.sh squad 800
#
# Cells, all from one process tree so every number in a dataset comes from one environment:
#
#   SCREEN (no fuser, 4 batched cells) -- the two preconditions, measurable before any fusion run:
#     encoding headroom = sharer/oracle - text_rag/oracle   (does the receiver under-encode?)
#     answer-in-doc     = sharer/oracle - sharer/none       (is the answer in the passage?)
#   MATRIX:
#     text_rag          the baseline that carries the claim (same receiver, same passage, as text)
#     c2c doc_only      CacheGraft: fuse the document only, so the cache is query-independent
#     recv_kv           CONTROL: identical cache pipeline, fusion off -- must land on text_rag
#     sharer            quality ceiling
#
# Add --terse to the text_rag/c2c pair for the matched short-answer protocol (judge-free
# lexical comparison); see the README.
set -u
[ $# -eq 2 ] || { echo "usage: $0 <data> <n_test>"; exit 1; }
D=$1; N=$2
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:-$(cd .. && pwd)/C2C}"
export USE_TF=0
PY=${PY:-../.venv/bin/python}
run () { echo "### $(date +%H:%M:%S) :: $*"; "$@" 2>&1 | tail -2; }

echo "===== $D : screen (no fuser) ====="
run $PY rag_trivia.py --data $D --method receiver --fact none   --n_test $N
run $PY rag_trivia.py --data $D --method sharer   --fact none   --n_test $N
run $PY rag_trivia.py --data $D --method text_rag --fact oracle --n_test $N
run $PY rag_trivia.py --data $D --method sharer   --fact oracle --n_test $N

echo "===== $D : matrix ====="
run $PY rag_trivia.py --data $D --method text_rag --fact retrieved --n_test $N
run $PY rag_trivia.py --data $D --method sharer   --fact retrieved --n_test $N
run $PY rag_trivia.py --data $D --method c2c --fact retrieved --fuse doc_only --n_test $N
run $PY rag_trivia.py --data $D --method c2c --fact oracle    --fuse doc_only --n_test $N
run $PY rag_trivia_cached.py --data $D --fact retrieved --recv_kv --n_test $N
run $PY rag_trivia_cached.py --data $D --fact oracle    --recv_kv --n_test $N
rm -rf ${D}_kvcache_recv        # control caches are single-use and ~18 MB/passage

echo "### $D done $(date +%H:%M:%S)"
