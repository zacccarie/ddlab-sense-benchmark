"""Phase 8 — tokeniseur à objectif DYNAMIQUE.

Phase 7 a montré qu'un tokeniseur de reconstruction casse la structure dès la
première compression (falaise : AMI 0.95 -> 0.02 entre K=512 et K=256), parce
qu'il fusionne les états par similarité visuelle, pas par rôle dynamique.

Ici l'objectif n'est plus de reconstruire l'image courante mais de prédire
l'image SUIVANTE : le code doit retenir ce qui détermine le futur. Question —
une compression AVEC PERTE préserve-t-elle alors la structure causale ?
On balaye K et on compare la courbe AMI(K) à la falaise de reconstruction."""

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
    nextImages = images[field.succ]  # image de l'état successeur de chaque état
    print(f"règle {RULE}, n={N} | {1 << N} états | tokeniseur PRÉDICTIF "
          f"(cible = image suivante)")
    print(f"vrais bassins : {len(field.attractors)}\n")

    header = (f"{'K':>5} {'codes':>7} {'collision':>10} {'cohérence':>10} "
              f"{'bassins lat.':>13} {'AMI':>8}")
    print(header)
    print("-" * len(header))
    for k in K_VALUES:
        model = trainVqvae(images, nextImages, k, VQVAE_EPOCHS, device)
        code = model.encodeCodes(images, device)
        m = metricsForCode(code, field)
        print(f"{k:>5} {m['distinctCodes']:>7} {m['collisionRate']:>9.3f} "
              f"{m['consistent']:>10.3f} {m['latentBasins']:>13} {m['ami']:>8.3f}")
    print(
        "\nLecture — comparer la courbe AMI(K) à la falaise de reconstruction"
        "\n(Phase 7 : K=512 AMI 0.95, K<=256 AMI ~0.02). Si l'AMI tient à K réduit,"
        "\nl'objectif dynamique permet de compresser sans détruire le sens."
    )


if __name__ == "__main__":
    main()
