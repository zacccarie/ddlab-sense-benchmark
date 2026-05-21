# ddlab-sense-benchmark

**Auditing representation learners against exact causal ground truth.**

Companion code for the paper *Perception Is Not Causation: Auditing
Representation Learners Against Exact Causal Ground Truth* (`paper/paper.tex`).

## What this is

Probing studies report that a representation encodes some property at *X%*
accuracy — but without the true structure of the data, that number cannot be
calibrated. This project removes the gap. It studies discrete dynamical
systems small enough to enumerate completely — elementary cellular automata,
random Boolean networks, discretized coupled map lattices. For such a system
the *entire* dynamics — every attractor, basin of attraction, garden-of-Eden
state, pre-image, and the bisimulation quotient — is computed exactly, without
sampling, and used as ground truth to audit what a representation learner
actually recovers.

## Key results

- **Forward training recovers local causal structure.** A next-state model
  develops a representation in which *local* causal facts (garden-of-Eden
  membership, in-degree) become linearly decodable — but global ones do not.
- **A structure-preserving compression exists.** The dynamical quotient
  compresses the state space while preserving basin structure exactly.
- **Perceptual objectives fail to recover it.** Five learned-tokenizer
  objectives all collapse under compression: the dynamical quotient is
  *perceptually shattered* — as dispersed in observation space as a random
  partition.
- **A recursive objective recovers it.** An objective defined recursively on
  the dynamics — a learned bisimulation metric — recovers the quotient
  (adjusted mutual information up to 0.95) where every observation-proximity
  objective stays near zero. The failure is in the objective, not the data.

## Install

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```sh
uv venv
uv pip install -e .
```

## Layout

```
src/ddlab_bench/      core library
  eca.py              elementary cellular automata
  rbn.py              random Boolean networks (Kauffman model)
  cml.py              discretized coupled map lattices
  basins.py           exact basin-of-attraction fields
  classify.py         Wuensche Z parameter, perceptual mismatch
  renormalization.py  refinement operator R, bisimulation quotient + metric, Cmu
  bisimulation.py     learned recursive objectives (bisimulation, invariance)
  metrics.py          probing and structure-recovery metrics
  models/             GRU sequence model, VQ-VAE tokenizer
scripts/              experiment probes — one per result
tests/                test suite
paper/                the paper (LaTeX source and figures)
```

## Reproduce

Each script is a self-contained experiment:

```sh
uv run python scripts/probe_representation.py        # local causal structure
uv run python scripts/probe_horizon_quotient.py      # the quotient exists
uv run python scripts/probe_perceptual_gap.py        # the perceptual shatter
uv run python scripts/probe_bisimulation.py          # perceptual vs recursive objectives
uv run python scripts/probe_bisimulation_robust.py   # the recursive result, solidified
uv run python scripts/probe_gamma_stability.py       # discount-factor sweep
uv run python scripts/probe_renormalization.py       # edge-of-chaos sweep
uv run python scripts/probe_cml.py                   # continuous substrate
```

## Tests

```sh
uv run pytest
```

## Paper

`paper/paper.tex` — build with `tectonic paper.tex`.

## License

Not yet chosen — to be set before public release.
