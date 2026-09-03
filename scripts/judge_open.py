"""Second-judge-family scoring with a LOCAL open-weights model.

`judge_trivia.py` grades with Claude Haiku via the `claude` CLI. A single grader is a single
point of failure for the paper's primary metric: the measured effect is entangled with output
length, and a length-tolerant judge would flatter the wordier system. This script replicates the
judgement with an independent, locally-run open-weights model (default microsoft/phi-4 -- a
different vendor, family and training pipeline), using the SAME reference-based prompt, so the
two judges can be compared cell-by-cell and item-by-item.

The judge sees only (question, answer, acceptable gold answers). It is blind to which system
produced the answer, and cells are scored with identical prompts.

  python judge_open.py --data squadpx_r text_rag_retrieved c2c_retrieved_doc_only

Writes `<tag>.judge_<short>.json` ({question: correct_bool}), the same schema as
judge_trivia.py's `<tag>.judge.json`, so downstream stats read either interchangeably.
"""
import os, json, re, argparse
os.environ.setdefault("USE_TF", "0")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import common as C

HERE = os.path.dirname(os.path.abspath(__file__))

# Same instructions as the Claude judge (judge_trivia.PREAMBLE), phrased for a single item.
INSTR = ("You are grading a short-answer response. You are given the QUESTION, the model's "
         "ANSWER, and a list of ACCEPTABLE gold answers. Reply \"correct\" if the ANSWER conveys "
         "one of the acceptable answers (ignore extra words, verbosity, or restating of the "
         "question); otherwise reply \"incorrect\". Judge correctness ONLY against the acceptable "
         "answers. Reply with exactly one word: correct or incorrect.")


def normalize(s):
    s = s.lower(); s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s); return re.sub(r"\s+", " ", s).strip()


def prompt_for(tok, q, ans, aliases):
    ans = " ".join(str(ans).split())[:300]
    body = (f"{INSTR}\n\nQUESTION: {q}\nANSWER: {ans}\n"
            f"ACCEPTABLE: {json.dumps(list(aliases))}")
    return tok.apply_chat_template([{"role": "user", "content": body}],
                                   tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--data", default="trivia")
    ap.add_argument("--model", default="microsoft/phi-4")
    ap.add_argument("--short", default=None, help="suffix for the output file (default: model stem)")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    short = args.short or args.model.split("/")[-1].replace(".", "").lower()
    # Pin the judge's kernels too. Without this, greedy judging still drifts on ~0.03% of
    # verdicts between runs on byte-identical generations (18/10400 on SQuAD), because CUDA
    # reductions are not bitwise-stable across processes.
    C.set_determinism()
    RESULTS = C.paths(args.data)["results"]

    tok = AutoTokenizer.from_pretrained(args.model); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="cuda").eval()

    print(f"judge={args.model}")
    print(f"{'cell':<40}{'judged':>8}{'contains':>10}{'J&!C':>7}{'C&!J':>7}{'agree':>8}{'unparsed':>9}")
    for tag in args.tags:
        d = json.load(open(os.path.join(RESULTS, f"{tag}.json")))
        S = d["samples"]
        prompts = [prompt_for(tok, s["q"], s["gen"], s["aliases"]) for s in S]
        verdicts, unparsed = [], 0
        for i in range(0, len(prompts), args.batch):
            enc = tok(prompts[i:i + args.batch], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                o = model.generate(**enc, do_sample=False, max_new_tokens=5,
                                   pad_token_id=tok.pad_token_id)
            for t in tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True):
                t = t.strip().lower()
                if "incorrect" in t:
                    verdicts.append(False)
                elif "correct" in t:
                    verdicts.append(True)
                else:                       # unparsable -> count as incorrect, but report it
                    verdicts.append(False); unparsed += 1
        jud = {S[j]["q"]: bool(verdicts[j]) for j in range(len(S))}
        cont = {}
        for s in S:
            al = [normalize(a) for a in s["aliases"]]
            p = normalize(s["gen"])
            cont[s["q"]] = any(a and a in p for a in al)
        n = len(S)
        j_acc = sum(jud.values()) / n
        c_acc = sum(cont.values()) / n
        j_not_c = sum(1 for q in jud if jud[q] and not cont[q])
        c_not_j = sum(1 for q in jud if cont[q] and not jud[q])
        agree = sum(1 for q in jud if jud[q] == cont[q]) / n
        json.dump(jud, open(os.path.join(RESULTS, f"{tag}.judge_{short}.json"), "w"))
        print(f"{tag:<40}{j_acc*100:>7.1f}%{c_acc*100:>9.1f}%{j_not_c:>7}{c_not_j:>7}"
              f"{agree*100:>7.1f}%{unparsed:>9}")


if __name__ == "__main__":
    main()
