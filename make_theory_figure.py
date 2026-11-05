"""Illustration of Proposition 2 under an ASSUMED accuracy curve. Not experimental data."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
a0, tau, lam, beta, N = 0.3, 300.0, 5e-4, 5e-4, 6000
n = np.arange(0, N + 1)
a = a0 + (1 - a0) * (1 - np.exp(-n / tau))
u, r = a - lam * n, a + beta * n
nu, nr = u.argmax(), r.argmax()
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(n, u, label=r"true utility $u(n)=a(n)-\lambda n$")
ax.plot(n, r, label=r"judge reward $r(n)=a(n)+\beta n$")
ax.axhline(u[0], color="gray", ls=":", lw=1, label="$u(0)$: not reasoning at all")
ax.plot([nu], [u[nu]], "o", color="C0"); ax.plot([nr], [u[nr]], "o", color="C3")
ax.annotate("judge's choice:\nworse than $n=0$", (nr, u[nr]), (3700, 0.9), arrowprops=dict(arrowstyle="->"))
ax.set_xlabel("reasoning budget $n$ (tokens)"); ax.set_ylabel("utility / reward")
ax.set_title("Illustrative model (assumed $a(n)$), not experimental data", fontsize=9)
ax.legend(fontsize=8, loc="upper left"); plt.tight_layout()
plt.savefig("theory_figure.png", dpi=200); plt.savefig("theory_figure.pdf")
print("u-opt n=%d, judge-opt n=%d, u(judge)=%.2f, u(0)=%.2f" % (nu, nr, u[nr], u[0]))
