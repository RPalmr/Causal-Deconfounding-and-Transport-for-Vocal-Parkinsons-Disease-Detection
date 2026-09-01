# `old/`

Legacy core module for the dissertation's causal-inference pipeline. This directory is
named `old/` because it predates the top-level `main.py`, but it is still actively
imported at runtime.

## Contents

- `causal.py` — single module containing:
  - Graph-string dictionary `GRAPH_MODELS` for every back-door adjustment set used
    in the dissertation (`age`, `grbas`, `updrs`, `both`, `g-as[a]`, `ug-as[a]`,
    `xy`). See Chapter 6, §6.2.5 ("Causal graph set").
  - Corpus loaders `load_nv` (NeuroVoz) and `load_pc` (ParkCeleb), which return the
    `(X, y, age, sex, …)` tuples consumed by every downstream arm. The
    `task="PETACA"` filter on `load_nv` is what restricts NeuroVoz to the Petaca
    Blanca elicitation so both corpora share the same feature space
    (Chapter 6, §6.2.1).
  - Causal-bootstrapping weight routines `_cb` and `_tcb_weights`, which call into
    the `causal_assistant` wrapper around GRAPL (closed-form interventional
    distribution) and CausalBootstrapping (KDE-based density ratio).
  - Stratified speaker-level split helper `_speaker_split` and the shared
    `GradientBoostingClassifier` factory `_gbc` (100 trees, lr 0.1, depth 3) used
    uniformly across confounded / de-confounded arms (Chapter 6, §6.2.3).

## How the dissertation uses it

Imported as `import old.causal as c` by `main.py`, by `tests/transport_u_duration.py`
(UGAS source-side causal-bootstrapping weights), by `tests/transport_sid_proxy.py`,
and by `paper_visuals/gen_fig_app_cb_weights.py`. The `GRAPH_MODELS` keys
`g-as[a]` and `ug-as[a]` are the only substantive NeuroVoz arms executed; both
reduce to the same Age+Sex back-door graph string
(`"y;X;Age;Sex; y->X; Age->X; Age->y; Sex->X; Sex->y;"`), so the
causal-weighting step is identical under either label (Chapter 6, §6.4.1).