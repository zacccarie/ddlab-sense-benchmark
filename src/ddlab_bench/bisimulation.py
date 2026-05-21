"""Objectif de renormalisation dynamique appris — le contre-test positif du §11.

Le §9 montre qu'aucun objectif perceptuel ne récupère le quotient dynamique.
Le §11 en propose le contre-test positif : un encodeur phi entraîné à
satisfaire l'équation de point fixe de la métrique de bisimulation

    d(phi(x), phi(x'))  ~=  1[l(x) != l(x')]  +  gamma * d(phi(f(x)), phi(f(x')))

Le terme de base 1[l != l'] ancre phi sur l'observable macroscopique l (et
interdit l'effondrement phi == cste) ; le terme récursif (gamma < 1,
contractant) propage l'équivalence le long de la dynamique. Aucune
reconstruction : phi ne vit jamais dans l'espace d'observation au-delà de l.

L'objectif est système-agnostique — il ne consomme que des vecteurs d'état, un
graphe de transition et un observable de base. Validé sur ECA contre le
quotient exact de `renormalization.bisimulationQuotient` ; applicable tel quel
aux RBN et aux CML."""

import numpy as np
import torch
import torch.nn as nn


class BisimulationEncoder(nn.Module):
    """phi : vecteur d'état -> plongement latent. MLP simple."""

    def __init__(self, featureDim, latentDim=16, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(featureDim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latentDim),
        )

    def forward(self, features):
        return self.net(features)


def trainBisimulationEncoder(stateFeatures, succ, label, latentDim=16,
                             gamma=0.9, epochs=600, lr=1e-3, tau=0.02,
                             device="cpu", seed=0, verbose=False):
    """Entraîne phi sur l'équation de point fixe de bisimulation, en batch
    complet — l'espace d'états est énumérable, toutes les paires tiennent.

    stateFeatures : (size, featureDim) — vecteur d'entrée par état.
    succ          : (size,) — graphe de transition.
    label         : (size,) — observable de base l.
    gamma         : poids récursif ; gamma=0 -> phi n'encode que l.
    tau           : taux d'EMA du réseau cible.

    La distance des successeurs est lue sur un RÉSEAU CIBLE — copie à moyenne
    glissante de phi. Sans lui, la cible (qui dépend de phi) et l'estimation
    courante se renforcent et l'échelle du plongement diverge ; le réseau cible
    découple la cible de l'encodeur vif et rend la récursion contractante
    (correctif standard, cf. deep bisimulation for control). Retourne
    (encoder, lossHistory)."""
    torch.manual_seed(seed)
    size, featureDim = stateFeatures.shape
    features = torch.tensor(np.asarray(stateFeatures, dtype=np.float32), device=device)
    succIdx = torch.tensor(np.asarray(succ), dtype=torch.long, device=device)
    label = np.asarray(label)
    labelMismatch = torch.tensor(
        (label[:, None] != label[None, :]).astype(np.float32), device=device
    )

    encoder = BisimulationEncoder(featureDim, latentDim).to(device)
    targetEncoder = BisimulationEncoder(featureDim, latentDim).to(device)
    targetEncoder.load_state_dict(encoder.state_dict())
    for param in targetEncoder.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        encoder.train()
        z = encoder(features)
        dCurrent = torch.cdist(z, z)                          # d(phi(x), phi(x'))
        with torch.no_grad():
            zTarget = targetEncoder(features)[succIdx]
            dSucc = torch.cdist(zTarget, zTarget)             # d cible des successeurs
            target = labelMismatch + gamma * dSucc
        loss = torch.mean((dCurrent - target) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # réseau cible : moyenne glissante exponentielle de l'encodeur vif
        with torch.no_grad():
            for pTarget, pLive in zip(targetEncoder.parameters(), encoder.parameters()):
                pTarget.mul_(1.0 - tau).add_(tau * pLive)
        history.append(float(loss.item()))
        if verbose and (epoch == 1 or epoch % 50 == 0):
            print(f"  epoch {epoch:>3} | loss {loss.item():.5f}")
    return encoder, history


def embedAllStates(encoder, stateFeatures, device="cpu"):
    """Plongement latent de tous les états — array (size, latentDim)."""
    encoder.eval()
    with torch.no_grad():
        features = torch.tensor(np.asarray(stateFeatures, dtype=np.float32), device=device)
        return encoder(features).cpu().numpy()


def trainInvarianceEncoder(stateFeatures, succ, latentDim=16, epochs=600,
                           lr=1e-3, varCoef=1.0, covCoef=1.0,
                           device="cpu", seed=0):
    """Entraîne phi à être INVARIANT par la dynamique : phi(x) ~= phi(f(x)).

    Un encodeur succ-invariant est constant sur chaque bassin — en itérant,
    phi(x) = phi(f^k(x)) = phi(attracteur). C'est la formulation spectrale du
    §11 : l'espace propre de valeur propre 1 de l'opérateur de Koopman est
    engendré par les indicatrices de bassin. L'effondrement phi == cste
    (l'invariant trivial) est interdit par les termes de variance et de
    covariance, style VICReg.

    Cible : la partition en BASSINS — là où l'objectif de bisimulation récupère
    le quotient, plus fin. Retourne (encoder, lossHistory)."""
    torch.manual_seed(seed)
    features = torch.tensor(np.asarray(stateFeatures, dtype=np.float32), device=device)
    succIdx = torch.tensor(np.asarray(succ), dtype=torch.long, device=device)

    encoder = BisimulationEncoder(features.shape[1], latentDim).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    history = []
    for _ in range(epochs):
        encoder.train()
        z = encoder(features)
        zSucc = encoder(features[succIdx])
        invariance = torch.mean((z - zSucc) ** 2)
        # anti-collapse VICReg : variance par dimension >= 1, dimensions décorrélées
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        varLoss = torch.mean(torch.relu(1.0 - std))
        zc = z - z.mean(dim=0)
        cov = (zc.T @ zc) / (z.shape[0] - 1)
        covLoss = ((cov ** 2).sum() - (torch.diagonal(cov) ** 2).sum()) / latentDim
        loss = invariance + varCoef * varLoss + covCoef * covLoss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))
    return encoder, history
