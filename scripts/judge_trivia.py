"""LLM-as-judge scoring for TriviaQA-RAG, using the `claude` CLI (headless, structured output).

Motivation: token-F1/EM penalize C2C's verbosity (it restates the question with the answer
inside), even though the answer is correct. A reference-based judge reads the WHOLE answer and
rules correct/incorrect against the gold aliases -- verbosity-robust, so we can score the
ORIGINAL (non-terse) high-`contains` generations directly. Applied identically to every method.

  python judge_trivia.py c2c_retrieved_doc_only text_rag_retrieved \
                         c2c_oracle_doc_only   text_rag_oracle --model haiku

Writes trivia_results/<tag>.judge.json ({"<index>\\t<question>": correct_bool}) and prints judged accuracy +
agreement with `contains` (a validation check: the judge should mostly agree, fixing only the
verbosity false-negatives and the alias-incidental false-positives).
"""
import os, json, re, sys, subprocess, argparse
import common as C
HERE = os.path.dirname(__file__)

def normalize(s):
    s = s.lower(); s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s); return re.sub(r"\s+", " ", s).strip()

SCHEMA = json.dumps({"type": "object", "properties": {"verdicts": {"type": "array", "items": {
    "type": "object", "properties": {"id": {"type": "integer"},
    "verdict": {"type": "string", "enum": ["correct", "incorrect"]}},
    "required": ["id", "verdict"]}}}, "required": ["verdicts"]})

PREAMBLE = (
    "You are grading short-answer trivia responses. For each item you are given the QUESTION, the "
    "model's ANSWER, and a list of ACCEPTABLE gold answers. Mark \"correct\" if the ANSWER conveys "
    "one of the acceptable answers (ignore extra words, verbosity, or restating of the question); "
    "otherwise \"incorrect\". Judge correctness ONLY against the acceptable answers.\n\n")

def call_judge(items, model):
    """items: list of (id, question, answer, aliases) -> {id: 'correct'/'incorrect'}."""
    body = []
    for i, q, a, al in items:
        a = " ".join(a.split())[:300]
        body.append(f"Item {i}\nQUESTION: {q}\nANSWER: {a}\nACCEPTABLE: {json.dumps(al)}")
    prompt = PREAMBLE + "\n\n".join(body)
    out = subprocess.run(["claude", "-p", prompt, "--model", model,
                          "--output-format", "json", "--json-schema", SCHEMA],
                         capture_output=True, text=True, stdin=subprocess.DEVNULL)
    env = json.loads(out.stdout)
    so = env.get("structured_output") or {}
    return {v["id"]: v["verdict"] for v in so.get("verdicts", [])}, env.get("total_cost_usd", 0.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--data", default="trivia")
    args = ap.parse_args()
    RESULTS = C.paths(args.data)["results"]
    data_qs = json.load(open(C.paths(args.data)["data"]))["questions"]

    print(f"{'cell':<30}{'judged':>8}{'contains':>10}{'J&!C':>7}{'C&!J':>7}{'agree':>8}")
    total_cost = 0.0
    for tag in args.tags:
        d = json.load(open(os.path.join(RESULTS, f"{tag}.json")))
        S = d["samples"]
        verdicts = {}
        for i in range(0, len(S), args.batch):
            chunk = [(j, S[j]["q"], S[j]["gen"], S[j]["aliases"])
                     for j in range(i, min(i + args.batch, len(S)))]
            for _try in range(3):
                vd, cost = call_judge(chunk, args.model); total_cost += cost
                if all(c[0] in vd for c in chunk):
                    break                       # all ids returned -> good
                chunk = [c for c in chunk if c[0] not in vd]  # retry the missing ones
            verdicts.update(vd)
        # metrics + validation against `contains`
        jud = [verdicts.get(j) == "correct" for j in range(len(S))]
        cont = [any(a in normalize(s["gen"]) for a in al)
                for s, al in zip(S, C.gold_aliases(data_qs, S))]
        n = len(S)
        j_acc = sum(jud) / n
        c_acc = sum(cont) / n
        j_not_c = sum(1 for j, c in zip(jud, cont) if j and not c)   # judge fixes verbosity FN
        c_not_j = sum(1 for j, c in zip(jud, cont) if c and not j)   # judge rejects spurious hit
        agree = sum(1 for j, c in zip(jud, cont) if j == c) / n
        C.write_scores(os.path.join(RESULTS, f"{tag}.judge.json"), S, jud)
        print(f"{tag:<30}{j_acc*100:>7.1f}%{c_acc*100:>9.1f}%{j_not_c:>7}{c_not_j:>7}{agree*100:>7.1f}%")
    print(f"\njudge={args.model}  total_cost_usd={total_cost:.2f}")

if __name__ == "__main__":
    main()
