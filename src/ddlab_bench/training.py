"""Entraînement du modèle A (GRU séquentiel)."""

import numpy as np
import torch
import torch.nn as nn

from ddlab_bench.models.sequence_model import SequenceModel, stateToBits


def makeBatches(traj, n, rng, batchSize):
    """Découpe les trajectoires en batches (entrée, cible) décalés d'un pas."""
    bits = stateToBits(traj, n)
    inputs = bits[:, :-1, :]
    targets = bits[:, 1:, :]
    order = rng.permutation(len(inputs))
    for start in range(0, len(order), batchSize):
        idx = order[start:start + batchSize]
        yield inputs[idx], targets[idx]


def trainSequenceModel(trainTraj, n, hidden, epochs, rng, device="cpu",
                       batchSize=128, lr=2e-3, verbose=False, seed=0):
    """Entraîne un GRU à prédire l'état suivant. Retourne le modèle entraîné.

    `seed` fixe l'initialisation des poids (reproductibilité)."""
    torch.manual_seed(seed)
    model = SequenceModel(n, hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lossFn = nn.BCEWithLogitsLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        epochLoss, batchCount = 0.0, 0
        for inputs, targets in makeBatches(trainTraj, n, rng, batchSize):
            x = torch.tensor(inputs, device=device)
            y = torch.tensor(targets, device=device)
            optimizer.zero_grad()
            loss = lossFn(model(x), y)
            loss.backward()
            optimizer.step()
            epochLoss += loss.item()
            batchCount += 1
        if verbose and (epoch == 1 or epoch % 5 == 0):
            print(f"  epoch {epoch:>3} | loss {epochLoss / batchCount:.4f}")
    return model


def forwardAccuracy(model, states, succ, device, chunk=4096):
    """Précision un-pas exacte (état entier correct) sur un pool d'états."""
    states = np.asarray(states)
    correct = 0
    for start in range(0, len(states), chunk):
        block = states[start:start + chunk]
        predicted = model.predictNext(block, device)
        correct += int(np.sum(predicted == succ[block]))
    return correct / len(states)


def parameterCount(model):
    """Nombre de paramètres entraînables — proxy de capacité du modèle."""
    return sum(p.numel() for p in model.parameters())
