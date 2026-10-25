# causal-inference-pd

## layout

```
main.py                         pipeline runner (see "run" below)
extraction/                     data-prep scripts
  downsample_neurovoz_audio.py  resamples neurovoz wavs to 32 khz pcm_16 mono
  grbas.py                      normalises raw grbas csvs in data/grbas/
  nisqav2.py                    runs nisqa mos/noi/dis/col/loud on parkceleb
  updrs.py                      builds data/updrs/updrs.csv from neurovoz metadata
  parkceleb/                    upstream Favaro et al. ParkCeleb repo
                                (download + interpretable/non_interpretable
                                 feature scripts; used as a reference, not
                                 re-run by this pipeline)
old/                           legacy core module (still imported at runtime)
  causal.py                     graph dictionary, load_nv/load_pc, _cb,
                                _gbc, _tcb_weights, _speaker_split, metrics
  README.md                     module-level docs for old/
libraries/                     third-party causal packages
  GRAPL/                        closed-form interventional distribution
  causal_assistant/             python wrapper that unifies GRAPL +
                                CausalBootstrapping behind one API
  CausalBootstrapping/          kde-based density-ratio reweighting
tCB/                           reference code for transportable causal
                                bootstrapping (sID_expr*, transport.py)
tests/                         cross-corpus and ablations arms
  transport_u_age.py            stage-2 raw-feature NV→PC sweep (chapter 7)
  transport_u_duration.py       ugas age+sex back-door transfer
  transport_sid_proxy.py        pca speaker-id proxy transfer (--transport-only)
  age_transport_intervals.py    age/sex weight-formulation diagnostic
  ssl_speaker_control.py        wav2vec2 speaker-level control
  ssl_transfer_compare.py       wav2vec2 vs hand-crafted features transfer
paper_visuals/                 matplotlib + tikz figure generators
  gen_fig*.py                   each writes the matching png in this folder
  gen_ch3_tikz.py, gen_tikz_to_png.py
                                tikz diagrams from the dissertation
Palmer-RXP528/                 the dissertation manuscript itself
  Dissertation.tex              master file, includes all chapters
  Chapter1/ … Chapter10/        chapter sources
  Appendices/                   appendix sources
  Preamble/                     shared latex preamble
  paper_visuals/                copy of the figure pngs used by the manuscript
results/                       pipeline output csvs (one per arm)
data/                          inputs (see "data" above)
papers/                        reference pdfs/txts (e.g. bareinboim 2014)
```

## pipeline at a glance

`main.py` runs five groups of arms and writes incremental csvs to
`results/`:

1. **replication** (`results/pipeline_replication_pc.csv`,
   `results/pipeline_replication_nv.csv`). fits a standard scaler +
   gradient-boosted classifier on each corpus with a 10-fold
   speaker-stratified split to reproduce baseline balanced accuracy.
2. **neurovoz graph/variant arms**
   (`results/pipeline_variants_nv.csv`). for every (graph ∈ {control,
   ug-as[a]}) × (variant ∈ {bootstrap, weights}) combination, computes
   the 2-d class-probability weights from `old/causal._cb` using the
   back-door graph `y;X;Age;Sex; y->X; Age->X; Age->y; Sex->X; Sex->y;`,
   and reports per-fold confounded vs deconfounded balanced accuracy
   plus a ks p-value.
3. **parkceleb same-corpus arms**
   (`results/pipeline_samecorpus_pc.csv`, `…_nv.csv`). same protocol
   but the confounder is the observed variable (age, accent, nisqa
   quality) instead of a graph.
4. **parkceleb age/acc/quality arms** — same file as (3), one row per
   variable. accent uses the nationality×ethnicity cluster index from
   `accent_map()`; quality uses `extraction/nisqav2.py` mos/noi/dis/col/loud.
5. **cross-corpus transfer** (`results/pipeline_transfer.csv`). the
   dissertation's main result. ugas source-side causal-bootstrapping
   weights from the age+sex graph are multiplied by an exact
   laplace-smoothed (age × sex) pmf transport ratio between neurovoz
   petaca blanca and each parkceleb disease-duration interval
   (`-2`, `-1`, `1`, `2`); a feature-selected gbm is bootstrapped with
   and without the resulting weights, and per-interval confounded vs
   deconfounded speaker-balanced accuracy is recorded.

two additional transport ablations live in `tests/`:

- `tests/transport_sid_proxy.py` (run with `python main.py
  --transport-only`) replaces the ugas age+sex graph with a single
  pca component of parkceleb intensity features most orthogonal to the
  pd label, then runs the same per-interval protocol with importance
  weights w = p(u|src) / (p(u|trg) · p(y|src,u)).
- `tests/transport_u_age.py` is the stage-2 raw-feature sweep (chapter
  7): transported / sID-weighted / feature-aligned / placebo / no-u /
  age-only variants on the same NV→PC split.

## requirements

- python 3.12
- core: `numpy`, `pandas`, `scipy`, `scikit-learn`, `openpyxl`, `tqdm`
- extra arms: `xgboost` (ssl transfer), `torch` (nisqa), `soundfile`
  + `ffmpeg` (downsample), `matplotlib` (figure scripts)

```
pip install numpy pandas scipy scikit-learn openpyxl tqdm xgboost matplotlib
```

## run

```
python main.py
```

the default run executes all five arm groups above with `--pc-reps 50`,
`--nv-reps 200`, `--xfer-reps 200`, `--cb-repeats 10`, and the seed
defined in `old/causal.py` (42). results are appended to the csvs in
`results/`, deduplicated by their key columns, so re-running with
different `--reps` only fills the missing rows.

### command-line arguments

```
--pc-reps INT      parkceleb repetitions (default 50)
--nv-reps INT      neurovoz repetitions (default 200)
--xfer-reps INT    transfer repetitions (default 200)
--cb-repeats INT   bootstrap repeats (default 10)
--reps INT         sets pc-reps, nv-reps, xfer-reps to one value
--seed INT         random seed (default 42, taken from old/causal.py)
--transport-only   skip everything except tests/transport_sid_proxy.py
```

### reproducing a single arm

every arm is also runnable as a standalone script:

```
python tests/transport_u_duration.py          # ugas transfer (chapter 6)
python tests/transport_sid_proxy.py           # sid-proxy transfer
python tests/transport_u_age.py               # stage-2 raw-feature sweep
python tests/age_transport_intervals.py 30 pb 10  # age-weight diagnostic
python tests/ssl_transfer_compare.py          # wav2vec2 vs 66-dim
python tests/ssl_speaker_control.py           # speaker-level ssl control
```

## output schema

| csv                                | key columns                                | value columns                                       |
|------------------------------------|--------------------------------------------|-----------------------------------------------------|
| `pipeline_replication_pc.csv`      | `interval`                                 | `n`, `speakers`, `ba`                               |
| `pipeline_replication_nv.csv`      | `model`                                    | `ba`                                                |
| `pipeline_variants_nv.csv`         | `graph`, `variant`                         | `conf`, `deconf`, `dacc`, `p_ks`, `reps`, `cb_repeats` |
| `pipeline_samecorpus_pc.csv`       | `variable`                                 | `n`, `conf`, `deconf`, `dacc`, `p_ks`, `reps`, `cb_repeats` |
| `pipeline_samecorpus_nv.csv`       | `graph`                                    | `conf`, `deconf`, `dacc`, `p_ks`, `reps`, `cb_repeats` |
| `pipeline_transfer.csv`            | `interval`                                 | `conf`, `deconf`, `dacc`, `p_ks`, `reps`, `cb_repeats` |
| `pipeline_transfer_sid_proxy.csv`  | `interval`                                 | `conf`, `deconf`, `dacc`, `p_ks`, `reps`, `cb_repeats` |

`dacc = deconf - conf`; `p_ks` is the two-sided ks test on the per-rep
balanced-accuracy distributions. each csv is keyed so that re-runs
append rather than overwrite.

## figures

every png in `paper_visuals/` is generated by a sibling `gen_*.py`
script:

- `fig1_semisynth.png` — semi-synthetic causal weight illustration.
- `fig2_nv_deconf.png` — neurovoz within-corpus deconfounding summary.
- `fig3_pc_deconf.png` — parkceleb within-corpus deconfounding summary.
- `fig5_ugas_transport.png` — ugas transfer results per interval.
- `fig6_method_compare.png` — ugas vs sid-proxy vs no-u comparison.
- `fig_app_cb_weights.png`, `fig_app_nv_demographics.png`,
  `fig_app_pc_intervals.png` — appendix figures.

`Palmer-RXP528/` contains the manuscript sources. `cd` into it and
run `latexmk -pdf Dissertation.tex` (or your preferred latex build)
to regenerate `Dissertation.pdf`; figures are pulled from
`Palmer-RXP528/paper_visuals/`, a copy of the pngs at the repo root.

## module map

- `old/causal.py` — single source of truth for the back-door graph
  dictionary `GRAPH_MODELS`, neurovoz and parkceleb corpus loaders,
  causal-bootstrapping weight routine, gbm factory, and speaker-level
  split helper. imported by `main.py` and every transport arm in
  `tests/`.
- `libraries/causal_assistant` — wraps `GRAPL` (closed-form
  interventional distribution) and `CausalBootstrapping` (kde density
  ratio) behind a single `causal_weights(causal_graph, cause_var,
  effect_var, fit_method, X, y, …)` call. `old/causal._cb` and
  `tests/transport_u_duration.py:source_cb_weights` both go through it.
- `tCB/` — reference implementation of transportable causal
  bootstrapping (the `sID_expr*.py` and `transport.py` files). kept
  for traceability against the bareinboim & correa 2020 algorithm.
- `tests/` — each script is a self-contained arm and prints a summary
  table at the end. some of them also write notes to `notes/`.
