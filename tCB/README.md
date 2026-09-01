# `tCB/`

transportable Causal Bootstrapping (tCB) — a vendored implementation of the
Bareinboim–Pearl transportability framework used as the conceptual reference for
the dissertation's cross-corpus transport arms.

## Background

The text file
`A General Algorithm for Deciding Transportability of Experimental Results.txt`
is Bareinboim & Pearl (2013), *Journal of Causal Inference* 1(1):107–134,
doi:10.1515/jci-2012-0004, which introduces the `sID` (selection-identification)
algorithm and motivates treating domain shift as selection on unobserved `S`-nodes.
Chapter 5, §5.4 of the dissertation cites this as the foundation for the sID-proxy
transport arm, and Chapter 6, §6.5.2 motivates the data-driven PCA-proxy
implementation that approximates the latent selection variable numerically.

## Contents

- `transport.py` — loaders for NeuroVoz and ParkCeleb, plus a PCA-based domain-
  confounder estimator `estimate_confound` and a `compute_tcb_weights` density-
  ratio helper. This is the closest vendored analogue of the dissertation's
  sID-proxy method.
- `sID_expr.py`, `sID_expr_moreDo.py` — symbolic `sID` algorithm and graph-mutilation
  helpers (`remove_incoming`, `remove_outgoing`, `ismsep_set`, `isdsep_set`, `sid`)
  built on top of GRAPL's `grapl.expr`. The dissertation replaces these symbolic
  computations with a numerical PCA proxy rather than running `sID` symbolically on
  a selection diagram.
- `expr_extend.py` — `LocalDOExpr` extension for GRAPL `expr`.
- `experiment/` — standalone tCB experiments on synthetic data
  (`syn_classification/`: back-door, front-door, non-identifiable settings) and on
  bgMNIST (`semi_syn_bgMNIST/`). Reference implementations only; not invoked by
  `main.py`.
- `debug.py`, `test_scr.py` — scratch / debugging scripts.

## How the dissertation uses it

Not invoked directly at runtime. `old/causal.py:sys.path` adds this directory so
that `_cb` / `_tcb_weights` can import the upstream `causal_assistant` interface,
which in turn imports the tCB kernel-density helpers. The dissertation's actual
transport arms (UGAS in `tests/transport_u_duration.py`, sID-proxy in
`tests/transport_sid_proxy.py`) are written from scratch on top of scikit-learn +
scipy and do not call the vendored `compute_tcb_weights`; this directory is kept
as the reference implementation that motivated the design.