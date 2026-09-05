# CLAUDE.md — Jet Scaling Laws Replication

## What this project is

A 2-day, rigor-first replication of the scaling-law methodology in Hartman et al.
(ATL-SOFT-PUB-2026-002), executed on the public Top Quark Tagging Reference
Dataset instead of ATLAS-internal data. We test the *qualitative claims*
(structure of the fitted exponents, cost-model deformations), not the numerical
values. The artifact is a public repo a researcher can skim in 5 minutes:
pre-registration doc, clean sweep, parametric fit with bootstrap CIs, two
headline figures, and a 2-page README written like a mini-paper.

**Claims under test (pre-registered):**
1. L(N, D) = L_inf + A/N^α + B/D^β fits the sweep, and α ≫ β (model exponent
   much larger than data exponent, matching Hartman's α≈0.677 vs β≈0.077 gap).
2. The power-law form beats an exponential-decay form on held-out grid points.
3. Under the extended cost model C = 6ND + kD + mN, switching on m > 0 pushes
   the compute-optimal N down at fixed budget; switching on k > 0 adds a
   constant data-cost offset. (Reproduced analytically — no training needed.)

## Non-negotiable rules

- **First commit = pre-registration.** `docs/preregistration.md` (claims table,
  planned fit procedure, fallbacks, empty deviations log) must be committed
  BEFORE any training run. Do not backfill it.
- **No leaderboard chasing.** Loss values need to be internally consistent
  across the sweep, not competitive with published taggers. Do not tune
  architecture per-run. One recipe, swept systematically.
- **Every deviation from the reference methodology goes in
  `docs/deviations.md`** with a classification: benign / potentially
  consequential / unknown.
- **Symmetric reporting.** If a claim doesn't hold, report it with candidate
  explanations. Never silently drop a failed check.
- Config-driven everything. No hardcoded hyperparameters in training code.
- Every run writes one JSON to `results/runs/` — never overwrite, never edit.

## Repo structure to scaffold

```
jet-scaling-replication/
├── CLAUDE.md                  # this file
├── README.md                  # mini-paper, written last (Day 2)
├── pyproject.toml
├── .gitignore                 # ignore data/, results/runs can be committed
├── docs/
│   ├── preregistration.md     # FIRST COMMIT
│   └── deviations.md
├── configs/
│   ├── grid.yaml              # the 4x4 (N, D) grid + seed-repeat spec
│   └── model_sizes.yaml       # depth/width per N target
├── data/                      # gitignored; HDF5 files live here
├── src/jetscaling/
│   ├── __init__.py
│   ├── data.py                # HDF5 load, feature building, D-subsampling
│   ├── model.py               # set transformer (no positional embedding)
│   ├── train.py               # single run: config in -> results JSON out
│   ├── sweep.py               # launches the grid from configs/grid.yaml
│   ├── fit.py                 # parametric fit: Huber on log-L + bootstrap
│   └── costmodel.py           # C = 6ND + kD + mN analytics (zero-compute)
├── notebooks/
│   ├── 01_sanity_checks.ipynb # load data, plot a jet, overfit 1k jets
│   ├── 02_costmodel.ipynb     # isoFLOP deformations w/ Hartman's constants
│   └── 03_results.ipynb       # final fits, CIs, headline figures
├── results/
│   ├── runs/                  # one JSON per training run
│   └── figures/
└── tests/
    └── test_fit.py            # fit recovers known synthetic exponents
```

## Environment setup

Python ≥ 3.10. Prefer `uv`; fall back to venv + pip.

```bash
uv init --python 3.11
uv add torch numpy h5py pandas scipy scikit-learn matplotlib tqdm pyyaml einops
uv add --dev pytest jupyter ipykernel
```

(pip equivalent: `pip install torch numpy h5py pandas scipy scikit-learn
matplotlib tqdm pyyaml einops pytest jupyter ipykernel`)

Package roles:
- `torch` — models + training (install the CUDA build matching the local GPU;
  check `nvidia-smi` first and use the correct index URL if needed)
- `h5py` — the dataset ships as HDF5 (pandas-style tables; `pandas.read_hdf`
  also works and is the easiest path)
- `scipy` — `scipy.optimize` for the Huber-loss parametric fit
- `scikit-learn` — AUC / background-rejection sanity metrics only
- `pyyaml` — run configs
- `einops` — optional, readable attention reshapes

## Dataset

Top Quark Tagging Reference Dataset (Kasieczka, Plehn, Thompson, Russel 2019).
- Zenodo DOI: 10.5281/zenodo.2603256  → https://zenodo.org/records/2603256
- Files: `train.h5` (1.2M jets), `val.h5` (400k), `test.h5` (400k)
- Format: pandas HDF5 table; per jet up to 200 constituents as (E, px, py, pz),
  zero-padded, plus `is_signal_new` label (1 = top, 0 = QCD).
- Download with wget/curl from the Zenodo record page into `data/`.
- Companion paper for generation details + reference numbers: arXiv:1902.09914.
- Cite BOTH the Zenodo DOI and arXiv:1902.09914 in README.

Feature building in `data.py`: from constituent 4-vectors compute per-particle
(log pT, Δη, Δφ, log E) relative to the jet axis; drop zero-padded entries via
a mask; cap at the first ~64 constituents by pT (record this cap in
deviations.md). Evaluation set: one fixed ~200k-jet slice of test.h5, identical
for every run in the sweep.

## Model spec (one recipe, do not tune per-run)

Set transformer: per-particle linear embed → N_layers pre-LN transformer
encoder blocks, NO positional embedding → masked mean-pool → linear → 1 logit.
BCE-with-logits loss. AdamW, cosine schedule with warmup, schedule length
matched to each run's token budget (this matters for scaling fits — note the
Kaplan-vs-Chinchilla LR-schedule lesson). Report final val loss in nats.

`configs/model_sizes.yaml` targets (adjust width/depth to hit param counts,
then record EXACT counts — embeddings included — in each run JSON):
- ~50k params   (e.g. 2 layers, d=32)
- ~200k params  (e.g. 3 layers, d=64)
- ~800k params  (e.g. 4 layers, d=96)
- ~3M params    (e.g. 6 layers, d=128)

## Sweep spec (`configs/grid.yaml`)

- D ∈ {50k, 150k, 500k, 1.2M} jets, subsampled with a fixed seed from train.h5
- Single pass over each subsample (single-epoch regime where possible)
- Full grid = 16 runs
- Seed repeats: 3 seeds at (50k params, 150k jets) and (800k params, 500k jets)
  for the seed-variance check → 6 extra runs
- Each run JSON: {params_exact, D, seed, final_val_loss, val_auc, wall_time_s,
  lr_schedule, git_commit}

## Fit spec (`fit.py`) — follow Hoffmann et al. Appendix D

- Parametric form: L(N, D) = L_inf + A/N^α + B/D^β
- Objective: Huber loss (δ = 1e-3) on log-space residuals
- Optimizer: L-BFGS (or scipy minimize) from a GRID of initializations; keep
  the best final objective; REPORT sensitivity to initialization
- Uncertainty: bootstrap over the 16 grid points (≥1000 resamples), percentile
  CIs on α, β, L_inf
- Falsification check: fit L = L_inf + A·exp(-N/n0) + B·exp(-D/d0); compare
  held-out residuals (leave-one-out over grid points); power law should win
- Validity check: refit excluding the largest-N column; check extrapolation
- `tests/test_fit.py`: generate synthetic (N, D, L) from known (α, β), confirm
  the fitter recovers them within bootstrap CI

## Cost-model notebook (`costmodel.py` + notebook 02) — ZERO GPU NEEDED

Implement C = 6ND + kD + mN. Using Hartman's published constants
(L_inf = 0.619, α = 0.677, β = 0.077) plot isoFLOP contours and the
compute-optimal frontier for: (k=0, m=0), (k=0, m=1e9), (k=1e9, m=0).
Verify the pre-registered qualitative deformations. On Day 2, overlay the
same plots with OUR fitted constants. Build this while the sweep runs.

## Build order (matches the 2-day plan)

1. Scaffold repo + env; write `docs/preregistration.md`; **commit** (must be
   the first commit, before any training).
2. Download dataset; notebook 01: load, plot one jet, overfit 1k jets.
3. `model.py` + `train.py`: one config → one results JSON.
4. `sweep.py`: launch 4 smallest runs, debug, then full 16 + seed repeats.
5. While sweep runs: `costmodel.py` + notebook 02 with Hartman's constants.
6. `fit.py` + test on synthetic data; first-pass fit on completed runs.
7. Final fits + bootstrap CIs; falsification + extrapolation checks;
   headline figures → `results/figures/`.
8. README mini-paper: claims tested, deviations table, results w/ CIs,
   symmetric outcomes, JetClass named as stage 2.

## Pre-committed fallbacks (also copy into preregistration.md)

- 3M-param row too slow → drop it; 3×4 grid; log in deviations.md.
- Fit unstable on 16 points → report instability explicitly (cf. Besiroglu
  et al. arXiv:2404.10102 on Approach-3 fragility); show init sensitivity.
- β differs numerically from Hartman's → expected (different dataset/task);
  the pre-registered claim is the α ≫ β structure, not the value.
