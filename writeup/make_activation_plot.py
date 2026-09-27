# GenAI disclosure: written by Claude (Anthropic) for Assignment 3, Section 3.2 figure.
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
x = np.linspace(-6, 6, 1201)
sig = 1/(1+np.exp(-x)); tanh = np.tanh(x); relu = np.maximum(0, x)
d_sig = sig*(1-sig); d_tanh = 1-tanh**2; d_relu = (x > 0).astype(float)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
panels = [("Sigmoid", sig, d_sig, r"$\sigma(x)$", r"$\sigma'(x)=\sigma(x)(1-\sigma(x))$"),
          ("Tanh", tanh, d_tanh, r"$\tanh(x)$", r"$\tanh'(x)=1-\tanh^2(x)$"),
          ("ReLU", relu, d_relu, r"$\mathrm{ReLU}(x)$", r"$\mathrm{ReLU}'(x)=\mathbb{1}[x>0]$")]
for ax, (name, f, df, lf, ldf) in zip(axes, panels):
    ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="gray", lw=0.6)
    if name == "ReLU":
        ax.plot(x, f, lw=2, label=lf)
        ax.plot(x[x < 0], df[x < 0], lw=2, ls="--", color="C1", label=ldf)
        ax.plot(x[x > 0], df[x > 0], lw=2, ls="--", color="C1")
        ax.plot([0], [0], "o", mfc="white", color="C1"); ax.plot([0], [1], "o", mfc="white", color="C1")
        ax.set_ylim(-0.5, 3)
    else:
        ax.plot(x, f, lw=2, label=lf); ax.plot(x, df, lw=2, ls="--", label=ldf)
    ax.set_title(name); ax.set_xlabel("x"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right" if name == "Tanh" else "upper left")
    ax.set_xlim(-6, 6)
fig.tight_layout(); fig.savefig("fig/activations.png", dpi=200)
print("saved")
