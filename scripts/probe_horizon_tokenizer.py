"""Phase 9 — tokeniseur à objectif MULTI-HORIZON.

Phases 7-8 : objectifs 1-pas (reconstruction, prédiction de l'image suivante) —
la compression casse la structure (falaise AMI 0.95+ -> 0.02), parce que pour
une dynamique déterministe l'information 1-pas ≈ l'information de l'état complet.
Le bassin est une propriété LONG-HORIZON.

Ici l'objectif du tokeniseur : prédire l'image de l'état H pas plus tard, avec
H >= transitoire maximal — l'état a alors atteint son cycle attracteur. Le code
doit retenir le DESTIN, pas l'apparence. Deux états du même bassin partagent ~le
même futur lointain → ils peuvent fusionner sans perdre le bassin. Question
décisive de §8 : la compression avec perte préserve-t-elle alors la structure ?"""

from ddlab_bench import computeBasinField
from ddlab_bench.basins import iterateSuccessor
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
    horizon = int(field.transient.max()) + 4
    targetImages = images[iterateSuccessor(field.succ, horizon)]

    print(f"règle {RULE}, n={N} | {1 << N} états | tokeniseur MULTI-HORIZON "
          f"(cible = image à H={horizon} pas)")
    print(f"vrais bassins : {len(field.attractors)}\n")

    header = (f"{'K':>5} {'codes':>7} {'collision':>10} {'cohérence':>10} "
              f"{'bassins lat.':>13} {'AMI':>8}")
    print(header)
    print("-" * len(header))
    for k in K_VALUES:
        model = trainVqvae(images, targetImages, k, VQVAE_EPOCHS, device)
        code = model.encodeCodes(images, device)
        m = metricsForCode(code, field)
        print(f"{k:>5} {m['distinctCodes']:>7} {m['collisionRate']:>9.3f} "
              f"{m['consistent']:>10.3f} {m['latentBasins']:>13} {m['ami']:>8.3f}")
    print(
        "\nLecture — comparer la courbe AMI(K) aux 1-pas :"
        "\n  Phase 7 reconstruction : K=512 AMI 0.95, K<=256 ~0.02"
        "\n  Phase 8 prédictif 1-pas : K=512 AMI 0.99, K<=256 ~0.07"
        "\nSi l'AMI tient ici à K réduit, l'objectif multi-horizon permet de"
        "\ncompresser sans détruire le sens — réponse à la question de §8."
    )


if __name__ == "__main__":
    main()
