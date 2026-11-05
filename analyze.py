"""Compute a(n), running max abar(n), block gains g_j with bootstrap CIs (resampling tasks),
test Hypothesis 1(a) (prefix), and plot a(n) / abar(n) with the judge win-rate (Hypothesis 1(b)).
Usage: python analyze.py --lam 0.01 [--judge judge.csv]   (lam = accuracy-equivalent cost of one block)"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def curves(df, tasks_idx):
    # df pivot: tasks x n  (mean over seeds)
    m = df.loc[tasks_idx].mean(axis=0).values
    return m, np.maximum.accumulate(m)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.csv"); ap.add_argument("--judge", default="judge.csv")
    ap.add_argument("--lam", type=float, required=True); ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    d = pd.read_csv(a.results)
    P = d.groupby(["task_id", "n"])["correct"].mean().unstack()
    ns = P.columns.values; tasks = P.index.values
    rng = np.random.default_rng(0)
    a_hat, abar_hat = curves(P, tasks)
    gains_hat = np.diff(abar_hat)
    boots = np.array([np.diff(curves(P, rng.choice(tasks, len(tasks)))[1]) for _ in range(a.B)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    below = [j for j in range(len(lo)) if hi[j] < a.lam]
    above = [j for j in range(len(lo)) if lo[j] >= a.lam]
    viol = [(i, j) for i in below for j in above if i < j]
    print(f"a(0)={a_hat[0]:.3f}  abar(max)={abar_hat[-1]:.3f}  blocks={len(gains_hat)}  lambda={a.lam}")
    for j, g in enumerate(gains_hat):
        print(f"block {j} [{ns[j]}-{ns[j+1]}]  gain={g:+.4f}  CI=[{lo[j]:+.4f},{hi[j]:+.4f}]")
    print("Hypothesis 1(a) REFUTED (non-prefix pair found): %s" % viol if viol else
          "Hypothesis 1(a): no refuting pair at this lambda (not proof of the hypothesis).")

    fig, ax = plt.subplots(1, 2 if os.path.exists(a.judge) else 1, figsize=(9 if os.path.exists(a.judge) else 5, 3.4))
    ax = np.atleast_1d(ax)
    se = np.array([curves(P, rng.choice(tasks, len(tasks)))[0] for _ in range(a.B)])
    l, h = np.percentile(se, [2.5, 97.5], axis=0)
    ax[0].fill_between(ns, l, h, alpha=0.2); ax[0].plot(ns, a_hat, label="a(n)")
    ax[0].plot(ns, abar_hat, "--", label=r"$\bar a(n)$"); ax[0].set_xlabel("reasoning budget n (tokens)")
    ax[0].set_ylabel("accuracy"); ax[0].legend(); ax[0].set_title("Accuracy vs. budget")
    if os.path.exists(a.judge):
        J = pd.read_csv(a.judge).groupby(["task_id", "n"])["win"].mean().unstack()
        jm = J.mean(axis=0)
        jb = np.array([J.loc[rng.choice(J.index.values, len(J))].mean(axis=0).values for _ in range(a.B)])
        jl, jh = np.percentile(jb, [2.5, 97.5], axis=0)
        ax[1].fill_between(jm.index, jl, jh, alpha=0.2); ax[1].plot(jm.index, jm.values)
        ax[1].axhline(0.5, color="gray", lw=0.8); ax[1].set_xlabel("reasoning budget n (tokens)")
        ax[1].set_ylabel("judge win-rate vs. n=0"); ax[1].set_title("LLM-judge preference vs. budget")
    plt.tight_layout(); plt.savefig("figure.png", dpi=200); plt.savefig("figure.pdf")
    print("saved figure.png / figure.pdf")

if __name__ == "__main__":
    main()
