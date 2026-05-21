"""Phase 9 (redesign) — isoler le phénomène : le quotient par horizon.

Phases 7-9 ont tenté d'APPRENDRE un tokeniseur et heurté des pathologies
d'entraînement (collapse VQ, conflit revival / objectif). On change d'objectif :
non plus « faire marcher le modèle » mais « isoler le phénomène ».

Sans aucun modèle. On construit la tokenisation directement par la dynamique :
code_H(s) = succ^H(s) — deux états partagent un code s'ils convergent vers le
même état en H pas. Or s'ils convergent, ils finissent dans le même bassin :
le quotient par horizon ne fusionne JAMAIS deux bassins. H=0 : bijection (aucune
compression) ; H grand : les états ont atteint leur cycle attracteur (forte
compression). Aucun modèle => aucun collapse possible. On isole proprement la
question : une discrétisation à long horizon compresse-t-elle SANS détruire la
structure causale ?"""

from ddlab_bench import computeBasinField
from ddlab_bench.basins import iterateSuccessor
from ddlab_bench.tokenizer_analysis import metricsForCode

RULE = 110
N = 9
HORIZONS = [0, 1, 2, 4, 8, 16, 32]


def main():
    field = computeBasinField(RULE, N)
    print(f"règle {RULE}, n={N} | {1 << N} états | quotient par horizon (sans modèle)")
    print(f"vrais bassins : {len(field.attractors)} | "
          f"transitoire max : {int(field.transient.max())}\n")

    header = (f"{'H':>4} {'codes':>7} {'collision':>10} {'cohérence':>10} "
              f"{'bassins lat.':>13} {'AMI':>8}")
    print(header)
    print("-" * len(header))
    for h in HORIZONS:
        code = iterateSuccessor(field.succ, h)
        m = metricsForCode(code, field)
        print(f"{h:>4} {m['distinctCodes']:>7} {m['collisionRate']:>9.3f} "
              f"{m['consistent']:>10.3f} {m['latentBasins']:>13} {m['ami']:>8.3f}")
    print(
        "\nLecture — le quotient par horizon compresse-t-il SANS casser l'AMI ?"
        "\n  AMI stable à 1 + collision qui monte => une discrétisation définie"
        "\n  par la dynamique (convergence à long horizon) compresse en préservant"
        "\n  exactement le sens. La tokenisation compressée préservante EXISTE."
        "\n  Le problème ouvert n'est donc pas son existence, mais qu'un tokeniseur"
        "\n  APPRIS depuis les pixels la découvre (Phases 7-9 : non, pas trivialement)."
    )


if __name__ == "__main__":
    main()
