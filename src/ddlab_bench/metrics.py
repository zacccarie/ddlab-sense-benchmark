"""Phase 3 — métriques d'extraction de sens.

inducedSuccessor applique le modèle à TOUT l'espace d'états: on obtient le graphe
de transition appris f-chapeau. Ses pré-images et son garden-of-Eden sont alors
comparés à la vérité-terrain DDLab. La prédiction forward est une fonction locale;
la structure inverse (pré-images, GoE) est non-locale et causale."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, v_measure_score

from ddlab_bench.basins import buildPreimageArray


def inducedSuccessor(model, n, device, chunk=4096):
    """Graphe de transition appris: induced[s] = état prédit par le modèle, pour tout s."""
    size = 1 << n
    induced = np.empty(size, dtype=np.int64)
    for start in range(0, size, chunk):
        states = np.arange(start, min(start + chunk, size))
        induced[states] = model.predictNext(states, device)
    return induced


def forwardEdgeAccuracy(induced, trueSucc):
    """Fraction d'arêtes du graphe de transition correctement prédites."""
    return float(np.mean(induced == trueSucc))


def preimageSetMetrics(induced, trueSucc):
    """Compare les ensembles de pré-images du graphe appris vs vérité-terrain."""
    inducedPre = buildPreimageArray(induced)
    truePre = buildPreimageArray(trueSucc)
    size = len(trueSucc)
    exact = 0
    jaccardSum = 0.0
    for s in range(size):
        learned = set(inducedPre[s])
        truth = set(truePre[s])
        if learned == truth:
            exact += 1
        union = learned | truth
        jaccardSum += 1.0 if not union else len(learned & truth) / len(union)
    return {
        "preimageExactMatch": exact / size,
        "preimageJaccardMean": jaccardSum / size,
    }


def goeDetectionMetrics(induced, trueSucc):
    """Détection du garden-of-Eden: in-degré 0 du graphe appris vs vérité-terrain.

    Un état GoE n'a aucun prédécesseur. Un modèle qui lui en invente un hallucine
    une histoire causale impossible (faux négatif); un modèle qui déclare GoE un
    état réellement atteignable efface une cause réelle (faux positif)."""
    size = len(trueSucc)
    inducedPre = buildPreimageArray(induced)
    truePre = buildPreimageArray(trueSucc)
    inducedGoe = np.fromiter(
        (len(inducedPre[s]) == 0 for s in range(size)), dtype=bool, count=size
    )
    trueGoe = np.fromiter(
        (len(truePre[s]) == 0 for s in range(size)), dtype=bool, count=size
    )
    truePos = int(np.sum(inducedGoe & trueGoe))
    falsePos = int(np.sum(inducedGoe & ~trueGoe))
    falseNeg = int(np.sum(~inducedGoe & trueGoe))
    precision = truePos / (truePos + falsePos) if (truePos + falsePos) else 0.0
    recall = truePos / (truePos + falseNeg) if (truePos + falseNeg) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "trueGoeCount": int(np.sum(trueGoe)),
        "inducedGoeCount": int(np.sum(inducedGoe)),
        "goePrecision": precision,
        "goeRecall": recall,
        "goeF1": f1,
    }


def basinHomomorphism(features, attractorId, seed=0, nInit=10):
    """Compare le partitionnement de l'espace de features (k-means) au vrai
    découpage en bassins d'attraction.

    k = nombre de bassins. Retourne AMI (Adjusted Mutual Information, ajusté
    pour le hasard) et V-measure. AMI élevé = les features regroupent les états
    par bassin — la représentation « découpe le réel aux jointures »."""
    k = int(attractorId.max()) + 1
    clusters = KMeans(n_clusters=k, random_state=seed, n_init=nInit).fit_predict(features)
    return {
        "ami": float(adjusted_mutual_info_score(attractorId, clusters)),
        "vMeasure": float(v_measure_score(attractorId, clusters)),
    }


def balancedAmi(reference, clusters, perClass=64, draws=5, seed=0):
    """AMI rééquilibrée pour des classes de référence déséquilibrées.

    L'AMI standard est dominée par les grosses classes : un bassin géant
    « écrase » la mesure. On échantillonne au plus `perClass` états par classe
    de référence — le géant est plafonné, les petites classes pleinement
    représentées —, on calcule l'AMI sur cet échantillon équilibré, et on
    moyenne sur `draws` tirages.

    Caveat : une classe plus petite que `perClass` reste sous-représentée ; la
    pondération supprime la domination du géant, pas tout déséquilibre."""
    reference = np.asarray(reference)
    clusters = np.asarray(clusters)
    rng = np.random.default_rng(seed)
    groups = [np.where(reference == c)[0] for c in np.unique(reference)]
    scores = []
    for _ in range(draws):
        sample = np.concatenate([
            rng.choice(group, size=min(perClass, len(group)), replace=False)
            for group in groups
        ])
        scores.append(adjusted_mutual_info_score(reference[sample], clusters[sample]))
    return float(np.mean(scores))
