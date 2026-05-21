"""Phase 7 — Pont 1 : balayage codebook, tokeniseur à objectif de RECONSTRUCTION.

On rend les états d'un ECA en images, on entraîne un VQ-VAE à reconstruire
l'image courante (tokeniseur aveugle à la dynamique), puis on projette la vraie
dynamique à travers la tokenisation et on compare le champ de bassins latent au
vrai. Balayage de K : à quel point de compression la structure casse ?

Objectif prédictif (alternative) : voir scripts/probe_predictive_tokenizer.py."""

from ddlab_bench import computeBasinField
from ddlab_bench.rendering import renderAll
from ddlab_bench.tokenizer_analysis import metricsForCode, trainVqvae

RULE = 110
N = 9
VQVAE_EPOCHS = 100
K_VALUES = [512, 256, 128, 64, 32, 16]


def main():
    device = "cpu"
    field = computeBasinField(RULE, N)
    images = renderAll(N)
    print(f"règle {RULE}, n={N} | {1 << N} états | tokeniseur de RECONSTRUCTION")
    print(f"vrais bassins : {len(field.attractors)}\n")

    header = (f"{'K':>5} {'codes':>7} {'collision':>10} {'cohérence':>10} "
              f"{'bassins lat.':>13} {'AMI':>8}")
    print(header)
    print("-" * len(header))
    for k in K_VALUES:
        model = trainVqvae(images, images, k, VQVAE_EPOCHS, device)
        code = model.encodeCodes(images, device)
        m = metricsForCode(code, field)
        print(f"{k:>5} {m['distinctCodes']:>7} {m['collisionRate']:>9.3f} "
              f"{m['consistent']:>10.3f} {m['latentBasins']:>13} {m['ami']:>8.3f}")
    print("\nLecture — AMI vs K : où la compression casse la structure causale.")


if __name__ == "__main__":
    main()
