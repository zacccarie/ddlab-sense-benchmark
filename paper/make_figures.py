"""Génère les figures du papier dans paper/figures/.

La falaise AMI/K reprend les chiffres des runs d'expérience commités
(Phases 7-8). Le mismatch perceptuel est recalculé exactement (instantané,
sans modèle). Exécuter depuis la racine du repo : python paper/make_figures.py"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ddlab_bench import computeBasinField  # noqa: E402
from ddlab_bench.classify import perceptualBasinMismatch, randomBasinMismatch  # noqa: E402

OUT = Path("paper/figures")


def figCliff():
    """Falaise AMI vs taille de codebook — données des Phases 7-8."""
    k = [16, 32, 64, 128, 256, 512]
    reconstruction = [-0.001, 0.014, 0.017, 0.017, 0.022, 0.951]
    predictive = [0.000, 0.000, 0.009, 0.032, 0.067, 0.993]

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.axhline(1.0, ls="--", color="0.55", lw=1.2,
               label="dynamical quotient (model-free)")
    ax.plot(k, reconstruction, "o-", color="#c0392b", label="reconstruction")
    ax.plot(k, predictive, "s-", color="#2c6fbb", label="next-image prediction")
    ax.set_xscale("log", base=2)
    ax.set_xticks(k)
    ax.set_xticklabels(k)
    ax.set_xlabel("codebook size $K$")
    ax.set_ylabel("AMI with true basin partition")
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("Learned tokenizers: a cliff, not a slope")
    ax.legend(fontsize=8, loc="center left")
    fig.tight_layout()
    fig.savefig(OUT / "ami_cliff.pdf")
    plt.close(fig)


def figMismatch():
    """Mismatch perceptuel-dynamique vs baseline aléatoire — recalcul exact."""
    rules = [(250, "ordered"), (110, "complex"), (30, "chaotic")]
    actual, baseline = [], []
    for rule, _ in rules:
        field = computeBasinField(rule, 9)
        actual.append(perceptualBasinMismatch(field))
        baseline.append(randomBasinMismatch(field))

    labels = [f"{name}\n(rule {rule})" for rule, name in rules]
    positions = range(len(rules))
    width = 0.38

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.bar([p - width / 2 for p in positions], actual, width,
           color="#c0392b", label="actual mismatch")
    ax.bar([p + width / 2 for p in positions], baseline, width,
           color="#b8b8b8", label="random baseline")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    ax.set_ylabel("1-bit perceptual$-$dynamical mismatch")
    ax.set_ylim(0, 1.0)
    ax.set_title("Basins are perceptually shattered (actual $=$ random)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "perceptual_mismatch.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    figCliff()
    figMismatch()
    print(f"figures écrites dans {OUT}/ : ami_cliff.pdf, perceptual_mismatch.pdf")


if __name__ == "__main__":
    main()
