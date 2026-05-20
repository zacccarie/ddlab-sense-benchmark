"""Phase 3 — sonde de représentation (test de H1).

Une sonde linéaire est entraînée à décoder, depuis un jeu de features, un fait
CAUSAL/INVERSE d'un état (appartenance au garden-of-Eden, in-degré) — fait que la
tâche forward n'exige jamais. La décodabilité linéaire est l'opérationnalisation
standard de « explicitement représenté » (Alain & Bengio). Si la représentation
d'un modèle forward convergé ne bat pas une représentation aléatoire, la
compétence forward ne porte pas la structure causale."""

import numpy as np
import torch
import torch.nn as nn


def goeLabels(field):
    """Étiquette binaire: 1 si l'état est garden-of-Eden (in-degré 0)."""
    labels = np.zeros(1 << field.n, dtype=np.int64)
    labels[np.asarray(field.goe, dtype=np.int64)] = 1
    return labels


def inDegreeLabels(field):
    """Étiquette 3 classes d'in-degré: 0 (GoE), 1, 2 (deux prédécesseurs ou plus)."""
    size = 1 << field.n
    degrees = np.fromiter(
        (len(field.preimages[s]) for s in range(size)), dtype=np.int64, count=size
    )
    return np.clip(degrees, 0, 2)


class LinearProbe(nn.Module):
    """Sonde linéaire: une seule couche affine, features -> classes."""

    def __init__(self, featureDim, numClasses):
        super().__init__()
        self.linear = nn.Linear(featureDim, numClasses)

    def forward(self, x):
        return self.linear(x)


class MLPProbe(nn.Module):
    """Sonde non-linéaire: perceptron à une couche cachée.

    Une sonde MLP peut recalculer la cible depuis n'importe quel jeu de features
    informatif; à sonde fixée, seule la comparaison forwardRep vs randomRep
    isole l'apport de l'entraînement."""

    def __init__(self, featureDim, numClasses, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(featureDim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, numClasses),
        )

    def forward(self, x):
        return self.net(x)


def _macroF1(predicted, trueLabels, numClasses):
    scores = []
    for c in range(numClasses):
        truePos = int(np.sum((predicted == c) & (trueLabels == c)))
        falsePos = int(np.sum((predicted == c) & (trueLabels != c)))
        falseNeg = int(np.sum((predicted != c) & (trueLabels == c)))
        precision = truePos / (truePos + falsePos) if (truePos + falsePos) else 0.0
        recall = truePos / (truePos + falseNeg) if (truePos + falseNeg) else 0.0
        scores.append(
            2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        )
    return float(np.mean(scores))


def trainProbe(features, labels, trainIdx, testIdx, numClasses, rng,
               epochs=50, batchSize=256, lr=5e-3, device="cpu", probeFactory=LinearProbe,
               seed=0):
    """Entraîne une sonde (linéaire ou MLP) et l'évalue sur le split de test.

    `seed` fixe l'initialisation de la sonde (reproductibilité).
    Retourne dict(testAccuracy, testMacroF1)."""
    torch.manual_seed(seed)
    probe = probeFactory(features.shape[1], numClasses).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    lossFn = nn.CrossEntropyLoss()

    xTrain = torch.tensor(features[trainIdx], dtype=torch.float32, device=device)
    yTrain = torch.tensor(labels[trainIdx], dtype=torch.int64, device=device)

    for _ in range(epochs):
        probe.train()
        order = rng.permutation(len(trainIdx))
        for start in range(0, len(order), batchSize):
            batchIdx = torch.from_numpy(order[start:start + batchSize])
            optimizer.zero_grad()
            loss = lossFn(probe(xTrain[batchIdx]), yTrain[batchIdx])
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        xTest = torch.tensor(features[testIdx], dtype=torch.float32, device=device)
        predicted = probe(xTest).argmax(dim=1).cpu().numpy()
    trueTest = labels[testIdx]
    return {
        "testAccuracy": float(np.mean(predicted == trueTest)),
        "testMacroF1": _macroF1(predicted, trueTest, numClasses),
    }


def majorityBaseline(labels, trainIdx, testIdx, numClasses):
    """Précision du classifieur trivial: classe majoritaire du train, évaluée en test."""
    majority = np.bincount(labels[trainIdx], minlength=numClasses).argmax()
    return float(np.mean(labels[testIdx] == majority))


def extractRepresentations(model, n, device, chunk=4096):
    """Représentation cachée du modèle pour tous les états de l'espace [0, 2^n)."""
    size = 1 << n
    feats = np.empty((size, model.gru.hidden_size), dtype=np.float32)
    for start in range(0, size, chunk):
        states = np.arange(start, min(start + chunk, size))
        feats[states] = model.representation(states, device)
    return feats
