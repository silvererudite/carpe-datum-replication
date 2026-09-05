# Pre-registration

**Frozen: 2026-09-05, in the first commit of this repository, before any training run.**

This document fixes what will be measured, how it will be fitted, and what counts as a
pass or a failure — *before* any loss value exists. Nothing here may be edited after the
first training run. Changes go in [`deviations.md`](deviations.md), dated and classified.

Author: shamima2hossain@gmail.com · Repository: this one · Analysis plan version: 1

---

## 0. What is being replicated, and what is not

**Target.** Hartman, Vigl et al., *Carpe Datum: Scaling behavior of transformers for
heavy hadron flavor identification*, ATL-SOFT-PUB-2026-002 (CERN CDS record
[2953659](https://cds.cern.ch/record/2953659), released 2026-01-30). Reference constants
used throughout, as recorded in `CLAUDE.md`: **L_inf = 0.619, α = 0.677, β = 0.077**.

**Closest public sibling**, by the same group, on public data: Vigl, Hartman, Kagan,
Heinrich, *Neural Scaling Laws for Boosted Jet Tagging*,
[arXiv:2602.15781](https://arxiv.org/abs/2602.15781) (JetClass). It is prior work, not a
target: it is read for method, and its numbers are *not* used to tune anything here.

**This replication differs from the target in four ways, all of them material:**

| Axis | Target | Here |
|---|---|---|
| Task | heavy-flavour (b/c/light) jet tagging | boosted top vs QCD tagging |
| Data | ATLAS-internal simulation | public Top Quark Tagging Reference Dataset, 1.2M train jets |
| Scale | "unprecedented" for HEP | 5.6e4 – 3.1e6 parameters, ≤ 1.2M jets, one laptop GPU |
| Inputs | full ATLAS track/vertex features | constituent four-vectors only |

Therefore **numerical agreement is not predicted and its absence is not a failure.**
What is predicted is *structure*: which exponent is larger, which functional form wins
out of sample, and which way the compute-optimal point moves when the cost model
changes. Those are the pre-registered claims below.

**Everything in §0 that is attributed to the target note comes from `CLAUDE.md`, not
from the note itself.** The note has not been read directly in preparing this document.
If reading it later contradicts anything here, that is a deviation (`D-002`), reported.

---

## 1. Claims and pass/fail criteria

Each criterion is a number chosen now, with no loss values in hand.

### C1 — The power law fits, and the model exponent dominates the data exponent

> L(N, D) = L_inf + A/N^α + B/D^β describes the 16-point sweep, and α ≫ β.

| | |
|---|---|
| **Estimator** | multi-start Huber fit of §5 on all 16 grid points |
| **Passes if** | (a) *adequacy*: max over the 16 points of \|L̂ − L\|/L ≤ 0.03, **and** the signs of the residuals show no row/column pattern (two-sided sign test over rows and over columns, p > 0.05); **and** (b) *dominance*: the bootstrap 95% percentile CI of (α − β) excludes 0 **and** the point estimate satisfies α/β ≥ 3 |
| **Fails if** | either the adequacy or the dominance condition is not met |
| **Reported either way** | α, β, L_inf with bootstrap CIs; the init-sensitivity spread; the residual table |

Rationale for α/β ≥ 3: the reference gap is 0.677/0.077 ≈ 8.8. A threshold of 3 is
deliberately looser than the reference ratio, because the task, data and scale all
differ — it tests the *structural* claim (parameters bind harder than data in this
regime) without smuggling in the reference's numbers.

### C2 — The power law beats an exponential-decay form out of sample

> L = L_inf + A·exp(−N/n0) + B·exp(−D/d0) is the falsification target. Same number of
> free parameters (5), same objective, same multi-start protocol.

| | |
|---|---|
| **Estimator** | leave-one-out over the 16 grid points; score each held-out point by \|log L̂ − log L\| |
| **Passes if** | the power law has the smaller mean held-out absolute log residual **and** is the better form on ≥ 12 of 16 held-out points (one-sided sign test, p = 0.038) |
| **Fails if** | the exponential wins on either count, or the two are within 12/16 of each other (an inconclusive result is reported as inconclusive, not as a pass) |

### C3 — Extended cost models deform the compute-optimal frontier as predicted

> Under C = 6ND + kD + mN, minimising L(N, D) at fixed C:
>
> - **C3a** m > 0 (per-parameter cost, e.g. inference in a trigger) pushes **N\* down**.
> - **C3b** k > 0 (per-example cost, e.g. simulating a jet) pushes **D\* down, N\* up**.
> - **C3c** both raise the achievable loss at fixed budget — the frontier moves up.

| | |
|---|---|
| **Estimator** | numerical constrained minimisation, zero training, at budgets {1e17, 1e19, 1e21} FLOPs, with (k, m) ∈ {(0,0), (0,1e9), (1e9,0)} |
| **Passes if** | the sign of every predicted shift holds at all three budgets, using the reference constants **and** again using our fitted constants |
| **Fails if** | any sign is wrong at any budget, or the optimum is not interior (the solver hits a bound) |

**Honest note on C3's strength.** For α, β ∈ (0,1) the *signs* of these shifts can be
derived on paper from the Lagrange conditions, so C3 is primarily a check that the
implementation and the constants are right, plus a report of the *magnitudes* (how many
dex N\* moves per decade of m). It is a weaker test than C1 and C2 and is reported as
such.

C3 needs no GPU and is checked first, on Day 1. `CLAUDE.md` states C3a and describes the
k-term as "a constant data-cost offset"; C3b sharpens that into a directional prediction
about D\*, and C3c is added here. Both additions are made before any data exists.

---

## 2. Data plan

- **Source.** Zenodo [10.5281/zenodo.2603256](https://doi.org/10.5281/zenodo.2603256),
  files `train.h5` (1.2M jets), `val.h5` (400k), `test.h5` (400k), md5s pinned in
  `scripts/get_data.py`. Not `pd4ml` — its mirrors are dead (`D-001`).
- **Training pool.** `train.h5` only. `val.h5` is *not* used: with a single-epoch recipe
  and no per-run tuning there is nothing to select on, and an unused split is one fewer
  way to leak.
- **Evaluation set.** The **first 200,000 rows of `test.h5`**, fixed, identical for all
  22 runs. Every loss entering the fit is measured on exactly these jets.
  *Contingency, committed now:* a head-of-file slice is only valid if the file is
  shuffled. If the signal fraction of that slice is outside [0.45, 0.55], the eval set
  becomes a random 200k drawn with seed 20260905 instead — decided before any run, and
  logged as a deviation. It is fixed once either way and never revisited.
- **Preprocessing.** Leading **64** constituents by pT (`D-003`); constituent real iff
  E > 0; 7 features per constituent
  (log pT, log E, Δη, Δφ, log pT_rel, log E_rel, ΔR) relative to the jet axis,
  following the ParticleNet/pd4ml convention. Feature list frozen — no additions later.
- **D-subsampling.** One permutation of the 1.2M training rows drawn with
  `subsample_seed = 20260905`, held fixed; each D is a **prefix** of it, so the D
  columns are nested and a difference between columns cannot be a lucky draw.
  The subsample seed is independent of the run seed.

## 3. Model and training recipe — frozen

One recipe. Only size varies. Any per-run tuning invalidates the sweep.

| Rung | layers | d_model | heads | predicted params |
|---|---|---|---|---|
| tiny | 2 | 48 | 4 | ~5.6e4 |
| small | 3 | 72 | 4 | ~1.9e5 |
| medium | 4 | 128 | 8 | ~7.9e5 |
| large | 7 | 192 | 8 | ~3.1e6 |

Set transformer, pre-LN, **no positional embedding**, masked mean-pool, one logit,
BCE-with-logits. AdamW (0.9, 0.95), wd 0.01, grad clip 1.0, peak LR 3e-4, 5% warmup,
cosine to 0.1×peak, **T_max = that run's own step count**, batch 256, one epoch, fp32.
Full spec: `configs/grid.yaml`. The N entering the fit is the **measured** parameter
count including embeddings, recorded per run — never the target from the table.

## 4. Sweep

16 cells = {tiny, small, medium, large} × D ∈ {50k, 150k, 500k, 1.2M}, seed 0.
Plus 6 seed repeats: seeds {1,2,3} at (tiny, 150k) and (medium, 500k). **22 runs.**
One immutable JSON per run in `results/runs/`, keys listed in `configs/grid.yaml`.
Never overwritten, never edited; a repeat gets a new `run_id` and both files stay.

**Exclusion rule, fixed now:** a run is excluded from the fit *only* if it diverged —
final training loss is NaN, or final eval loss ≥ log 2 = 0.693 (no better than a
constant predictor). Any exclusion is logged in `deviations.md` with the run JSON path.
No other exclusion, for any reason, at any time.

## 5. Fit procedure — frozen (Hoffmann et al. 2203.15556, appendix D)

- **Parameterisation** L̂ = exp(e) + exp(a − α·log N) + exp(b − β·log D), θ = (e,a,b,α,β),
  all unconstrained.
- **Objective** Huber, δ = 1e-3, on **log-space** residuals, summed over grid points.
- **Optimiser** L-BFGS from a grid of 4500 starts:
  a,b ∈ {0,5,10,15,20,25}; α,β ∈ {0,0.5,1,1.5,2}; e ∈ {−1,−0.5,0,0.5,1}. Keep the best
  final objective.
- **Init sensitivity is a reported result, not a diagnostic**: number of starts reaching
  within 1e-6 of the best objective, and the α/β spread across the best decile of starts.
  (Besiroglu et al. [2404.10102](https://arxiv.org/abs/2404.10102) re-fit Hoffmann's own
  approach-3 data and found the reported uncertainties implausibly tight; a replication
  that hides this sensitivity has replicated a number, not a method.)
- **Uncertainty** bootstrap over the **16 grid points** (the unit of observation),
  ≥ 1000 resamples, percentile CIs on L_inf, α, β, and the derived (α−β) and (α/β).
  Non-converged resamples are counted and reported, not dropped silently.
- **Pre-committed prediction about the CIs:** with 16 points they will be wide. Width is
  the honest result; it is not to be narrowed by adding points after seeing the fit.

## 6. Validity checks — all reported whatever they show

1. **Falsification (C2).** Exponential alternative, leave-one-out, sign test.
2. **Extrapolation.** Refit excluding the entire largest-N column; predict it; report
   predicted vs observed per cell. A law that only interpolates is not a law.
3. **Seed variance.** σ across seeds at the two repeat cells, compared with the spread
   of losses along a grid row. If σ_seed is of the same order, the fit is noise-limited
   and every conclusion above is reported as noise-limited.
4. **Synthetic recovery.** `tests/test_fit.py` generates (N, D, L) from known (α, β) with
   realistic noise; the fitter must recover them inside the bootstrap CI. If this test
   does not pass, no fit on real data is reported at all.

## 7. What will not be done

- No adding, re-running or dropping grid cells after seeing a fit, except under the
  §4 exclusion rule.
- No LR, architecture, batch or schedule change after any real run — that restarts the
  whole sweep from zero and is logged.
- No swapping the evaluation slice, ever.
- No reporting of α or β without their CIs and their init-sensitivity spread.
- No dropping a failed claim. A failed claim is a result and gets a README section with
  candidate explanations.

## 8. Pre-committed fallbacks

| Trigger | Fallback | Logged as |
|---|---|---|
| `large` row too slow on this hardware | drop it; fit a 3×4 grid; report the reduced N-lever explicitly | deviation |
| Fit unstable across inits | report the instability as the headline result, with the init-spread plot (cf. 2404.10102) | result, not deviation |
| β differs numerically from 0.077 | expected; C1 tests the α ≫ β structure, not the value | no action |
| A run diverges | exclude per §4, log it, and report the fit both with and without it | deviation |
| Dataloader-bound small runs | precompute the feature cache; if throughput still differs > 2× across cells, report that wall-clock FLOP estimates are unreliable | deviation |

## 9. Compute environment (declared, because it constrains the grid)

Apple M4, 10 cores, 16 GB unified memory, macOS 24.6.0; PyTorch **2.14.0** on the
**MPS** backend, fp32 (bf16 autocast on MPS is not trusted here); Python **3.11.13** via
`uv`. No CUDA. Verified at freeze time: `torch.backends.mps.is_available() == True`.
Exact package versions are pinned in `uv.lock`; `torch.__version__` and the resolved
device string are written into every run JSON.

## 10. Deviations log

See [`deviations.md`](deviations.md). It is already non-empty at freeze time: D-001
(data source), D-002 (task and reference-constant provenance), D-003 (constituent cap),
D-004 (model ladder widths), D-005 (hardware), D-006 (added dependency).
