"""Truncate-and-answer experiment. Requires a GPU, vllm, datasets.
Usage: python run_experiment.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --tasks 500 --seeds 3
Writes results.csv (task_id,seed,n,correct) and responses.jsonl (for the judge)."""
import argparse, csv, json, re
from vllm import LLM, SamplingParams
from datasets import load_dataset

SUFFIX = "\n\nFinal answer: \\boxed{"

def gold(ans):  # GSM8K gold is after '####'
    return ans.split("####")[-1].strip().replace(",", "")

def norm(s):
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return m[0] if m else s

def boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0: return None
    j, depth, out = i + 7, 1, ""
    while j < len(text) and depth:
        c = text[j]; depth += (c == "{") - (c == "}")
        if depth: out += c
        j += 1
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    ap.add_argument("--tasks", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--nmax", type=int, default=1024)
    ap.add_argument("--step", type=int, default=64)   # uniform blocks, k = step
    a = ap.parse_args()
    grid = list(range(0, a.nmax + 1, a.step))

    ds = load_dataset("gsm8k", "main", split="test").select(range(a.tasks))
    llm = LLM(a.model, dtype="bfloat16"); tok = llm.get_tokenizer()
    def prompt(q):
        m = [{"role": "user", "content": q + "\nReason step by step."}]
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)

    rows, resp = [], []
    for seed in range(a.seeds):
        sp = SamplingParams(temperature=0.6, max_tokens=a.nmax, seed=seed)
        prompts = [prompt(x["question"]) for x in ds]
        outs = llm.generate(prompts, sp)
        jobs = []  # (task, n, prompt_text, finished_text or None)
        for t, (x, o) in enumerate(zip(ds, outs)):
            ids, finished = list(o.outputs[0].token_ids), o.outputs[0].finish_reason == "stop"
            for n in grid:
                if finished and n >= len(ids):
                    jobs.append((t, n, None, o.outputs[0].text))
                else:
                    jobs.append((t, n, prompts[t] + tok.decode(ids[:n]) + SUFFIX, None))
        forced = [j for j in jobs if j[2] is not None]
        fo = llm.generate([j[2] for j in forced],
                          SamplingParams(temperature=0, max_tokens=24, stop=["}"]))
        fmap = {(j[0], j[1]): o.outputs[0].text for j, o in zip(forced, fo)}
        for t, n, p, full in jobs:
            pred = norm(boxed(full) or "") if full is not None else norm(fmap[(t, n)])
            ok = int(pred == norm(gold(ds[t]["answer"])))
            rows.append((t, seed, n, ok))
            text = full if full is not None else p[len(prompts[t]):] + fmap[(t, n)] + "}"
            resp.append({"task_id": t, "seed": seed, "n": n, "question": ds[t]["question"], "response": text})
    with open("results.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["task_id", "seed", "n", "correct"]); w.writerows(rows)
    with open("responses.jsonl", "w") as f:
        for r in resp: f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main()
