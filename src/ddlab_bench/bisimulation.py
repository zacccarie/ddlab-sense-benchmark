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

from ddlab_bench.models.sequence_model import stateToBits
from ddlab_bench.models.trajectory_transformer import TrajectoryTransformer


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

    Deux stabilisateurs rendent la récursion sûre, jusqu'à gamma -> 1 :
    - RÉSEAU CIBLE — la distance des successeurs est lue sur une copie à moyenne
      glissante de phi ; sans lui, cible et estimation courante se renforcent
      (correctif standard, cf. deep bisimulation for control).
    - CIBLE NORMALISÉE — (1-gamma)*base + gamma*d_succ. Le point fixe non
      normalisé croît comme 1/(1-gamma) et fait diverger l'échelle du plongement
      quand gamma -> 1 ; normalisé, il reste borné dans [0,1] pour tout gamma.
      La partition récupérée est inchangée : le point fixe normalisé vaut
      (1-gamma) fois le non normalisé — même géométrie, échelle bornée.

    Retourne (encoder, lossHistory)."""
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
            target = (1.0 - gamma) * labelMismatch + gamma * dSucc
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


def orbitWindows(succ, n, windowLength):
    """orbitBits[s] = bits des `windowLength` états de l'orbite avant de s :
    s, succ(s), succ^2(s), ... — l'entrée du transformer de trajectoire."""
    succ = np.asarray(succ)
    size = len(succ)
    windows = np.empty((size, windowLength), dtype=np.int64)
    cur = np.arange(size)
    for k in range(windowLength):
        windows[:, k] = cur
        cur = succ[cur]
    return stateToBits(windows.reshape(-1), n).reshape(size, windowLength, n)


def trainTrajectoryTransformer(orbitBits, succ, label, gamma=0.7, epochs=400,
                               lr=1e-3, tau=0.02, device="cpu", seed=0):
    """Entraîne un transformer de trajectoire sur l'équation de point fixe de
    bisimulation (§11).

    Même objectif que `trainBisimulationEncoder` — cible normalisée, réseau
    cible — mais l'encodeur est un transformer qui lit la fenêtre d'orbite de
    chaque état (séquence temporelle), au lieu d'un MLP qui ne lit que l'état.
    C'est le §11 porté sur l'architecture des world models. Retourne
    (encoder, lossHistory)."""
    torch.manual_seed(seed)
    orbits = torch.tensor(np.asarray(orbitBits, dtype=np.float32), device=device)
    size, windowLength, n = orbits.shape
    succIdx = torch.tensor(np.asarray(succ), dtype=torch.long, device=device)
    label = np.asarray(label)
    labelMismatch = torch.tensor(
        (label[:, None] != label[None, :]).astype(np.float32), device=device
    )

    encoder = TrajectoryTransformer(n, windowLength).to(device)
    targetEncoder = TrajectoryTransformer(n, windowLength).to(device)
    targetEncoder.load_state_dict(encoder.state_dict())
    for param in targetEncoder.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    history = []
    for _ in range(epochs):
        encoder.train()
        z = encoder(orbits)
        dCurrent = torch.cdist(z, z)
        with torch.no_grad():
            zTarget = targetEncoder(orbits)[succIdx]
            dSucc = torch.cdist(zTarget, zTarget)
            target = (1.0 - gamma) * labelMismatch + gamma * dSucc
        loss = torch.mean((dCurrent - target) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for pTarget, pLive in zip(targetEncoder.parameters(), encoder.parameters()):
                pTarget.mul_(1.0 - tau).add_(tau * pLive)
        history.append(float(loss.item()))
    return encoder, history
