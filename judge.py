"""Pairwise LLM-judge win-rate of the response at budget n versus the response at budget 0,
with position swap. Any OpenAI-compatible endpoint (set OPENAI_BASE_URL / OPENAI_API_KEY).
Usage: python judge.py --judge-model <name>   -> writes judge.csv (task_id,seed,n,win)"""
import argparse, csv, json
from openai import OpenAI

PROMPT = ("Question:\n{q}\n\nResponse A:\n{a}\n\nResponse B:\n{b}\n\n"
          "Which response is better overall? Reply with exactly one letter: A or B.")

def ask(client, model, q, a, b):
    r = client.chat.completions.create(model=model, temperature=0, max_tokens=2,
        messages=[{"role": "user", "content": PROMPT.format(q=q, a=a, b=b)}])
    return r.choices[0].message.content.strip().upper()[:1]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--judge-model", required=True)
    a = ap.parse_args(); client = OpenAI()
    R = {}
    for line in open("responses.jsonl"):
        r = json.loads(line); R[(r["task_id"], r["seed"], r["n"])] = r
    with open("judge.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["task_id", "seed", "n", "win"])
        for (t, s, n), r in R.items():
            if n == 0: continue
            base = R[(t, s, 0)]
            w1 = ask(client, a.judge_model, r["question"], r["response"], base["response"]) == "A"
            w2 = ask(client, a.judge_model, r["question"], base["response"], r["response"]) == "B"
            w.writerow([t, s, n, (int(w1) + int(w2)) / 2])

if __name__ == "__main__":
    main()
