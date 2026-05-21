"""CML discrétisé — balayage 2D du plan de contrôle (r, eps) : le P4 sur substrat continu.

Le probe initial (4 points) a montré que le ratio mismatch/hasard N'EST PAS
uniformément 1.00 sur CML — contrairement aux ECA (§9). Ce balayage dense teste
où la barrière perceptuelle tient et où elle cède, et si la complexité
statistique C_mu pique au bord du chaos (§12).

Trace deux diagrammes de phase : ratio(r, eps) et C_mu(r, eps).
n=7 cellules, m=8 niveaux -> 8^7 = 2 097 152 états énumérés par point."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ddlab_bench import basinFieldFromSuccessor  # noqa: E402
from ddlab_bench.classify import (  # noqa: E402
    maryBasinMismatch,
    randomBasinMismatch,
    structuralSummary,
)
from ddlab_bench.cml import CoupledMapLattice  # noqa: E402
from ddlab_bench.renormalization import (  # noqa: E402
    bisimulationQuotient,
    statisticalComplexity,
)

N = 7
M = 8
R_VALUES = [3.5, 3.6, 3.7, 3.8, 3.9, 4.0]
EPS_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
OUT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data")


def runPoint(r, epsilon):
    cml = CoupledMapLattice(N, M, r, epsilon)
    succ = cml.successorArray()
    field = basinFieldFromSuccessor(succ, N, "cml")
    mismatch = maryBasinMismatch(field, M)
    baseline = randomBasinMismatch(field)
    quotient = bisimulationQuotient(succ, cml.densityLabel())
    return {
        "ratio": mismatch / baseline if baseline > 0 else 0.0,
        "cmu": statisticalComplexity(quotient),
        "basins": len(field.attractors),
        "goe": structuralSummary(field)["goeFraction"],
    }


def renderPhaseDiagram(ratio, cmu):
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    panels = [
        (axes[0], ratio, "ratio mismatch / hasard",
         "inferno", "barrière perceptuelle (§9)"),
        (axes[1], cmu, "C_mu  (bits)",
         "viridis", "complexité du quotient (§12)"),
    ]
    for ax, data, label, cmap, title in panels:
        im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(R_VALUES)))
        ax.set_xticklabels([f"{r:.1f}" for r in R_VALUES])
        ax.set_yticks(range(len(EPS_VALUES)))
        ax.set_yticklabels([f"{e:.1f}" for e in EPS_VALUES])
        ax.set_xlabel("r  (carte logistique)")
        ax.set_ylabel("eps  (couplage)")
        ax.set_title(title)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                        color="white", fontsize=8)
        fig.colorbar(im, ax=ax, label=label)
    fig.suptitle(f"CML discrétisé — diagramme de phase  (n={N}, m={M})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "cml_phase_diagram.png"), dpi=140)
    plt.close(fig)


def main():
    nr, ne = len(R_VALUES), len(EPS_VALUES)
    ratio = np.zeros((ne, nr))
    cmu = np.zeros((ne, nr))
    basins = np.zeros((ne, nr), dtype=int)
    goe = np.zeros((ne, nr))
    print(
        f"CML balayage (r, eps) — n={N}, m={M}, {M ** N} états/point, "
        f"{nr * ne} points\n",
        flush=True,
    )
    for i, epsilon in enumerate(EPS_VALUES):
        for j, r in enumerate(R_VALUES):
            p = runPoint(r, epsilon)
            ratio[i, j] = p["ratio"]
            cmu[i, j] = p["cmu"]
            basins[i, j] = p["basins"]
            goe[i, j] = p["goe"]
            print(
                f"  r={r:.2f} eps={epsilon:.2f}  ratio={p['ratio']:.3f}  "
                f"Cmu={p['cmu']:.2f}  #bas={p['basins']:>5}  "
                f"GoE={p['goe'] * 100:.1f}%",
                flush=True,
            )

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(
        os.path.join(OUT_DIR, "cml_phase.npz"),
        rValues=R_VALUES, epsValues=EPS_VALUES,
        ratio=ratio, cmu=cmu, basins=basins, goe=goe,
    )
    renderPhaseDiagram(ratio, cmu)
    print(f"\nÉcrit : {OUT_DIR}/cml_phase.npz + cml_phase_diagram.png", flush=True)


if __name__ == "__main__":
    main()
