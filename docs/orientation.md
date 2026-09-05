# Orientation — the landscape you are about to build in

Written for someone who wants to *do* this replication, not commission it. It covers:
where the method came from, what the fitted numbers actually mean, the six places this
kind of study goes wrong, and a build ladder where each rung has a check you can run and
a question you should be able to answer before climbing to the next.

Nothing here is a rule — the rules are in `CLAUDE.md` and
[`preregistration.md`](preregistration.md). This is the map.

---

## 1. The whole project in one paragraph

Train the same jet tagger at 4 model sizes × 4 dataset sizes. You get 16 loss numbers.
Fit a 5-parameter surface **L(N, D) = L_inf + A/N^α + B/D^β** to them. The three fitted
quantities that matter are **L_inf** (the loss you would reach with infinite model and
infinite data — the irreducible difficulty of the task), **α** (how fast loss falls as
you add parameters) and **β** (how fast it falls as you add data). Then ask three
questions: does the surface actually fit? does it beat a different functional form on
points it never saw? and what does it say about how to spend a fixed budget — especially
when the budget includes the cost of *generating* simulated jets? That last question is
why an ATLAS group cares, and it is where the physics differs from the LLM literature.

---

## 2. The five papers, in reading order

| # | Read before | Paper | What to take from it |
|---|---|---|---|
| 1 | anything | Kaplan et al. 2020, [2001.08361](https://arxiv.org/abs/2001.08361) | The original claim that test loss is a clean power law in N, D and C. Also the cautionary tale — see §5.1. |
| 2 | writing `fit.py` | Hoffmann et al. 2022 ("Chinchilla"), [2203.15556](https://arxiv.org/abs/2203.15556) | **Appendix D is the method you are implementing**: the parametric form, Huber on log residuals, the multi-start grid. §3 explains the three approaches. |
| 3 | reporting any number | Besiroglu et al. 2024, [2404.10102](https://arxiv.org/abs/2404.10102) | A replication of Hoffmann's *approach 3* that found the published confidence intervals implausibly tight and the fit fragile. This is the standard your error bars are held to. |
| 4 | writing `data.py` | Kasieczka et al. 2019, [1902.09914](https://arxiv.org/abs/1902.09914) | The dataset you are using and what good performance on it looks like. Read §2 (data) and skim the tagger table. |
| 5 | Day 2 write-up | Vigl, Hartman, Kagan, Heinrich 2026, [2602.15781](https://arxiv.org/abs/2602.15781) | Same group as the target note, public data (JetClass), **boosted jet tagging — your task family**. Compute-optimal laws, data-repetition effects, and how the constants move with input features. The closest thing to a ground truth you can check against. |

The target itself: **ATL-SOFT-PUB-2026-002**, *Carpe Datum: Scaling behavior of
transformers for heavy hadron flavor identification*
([CDS 2953659](https://cds.cern.ch/record/2953659)). Read it before Day 2 and settle
`D-002` in the deviations log. Note the title: *seize the data* — §4 below explains why
that is the conclusion these exponents point to.

---

## 3. What the fitted numbers mean

**L_inf** is the floor. Two jets with identical constituent four-vectors, one from a top
and one from QCD, are genuinely indistinguishable — no model can separate them. L_inf is
where that irreducible ambiguity lives. It is also where every *un*-modelled limitation
hides: your 64-constituent cap raises it (`D-003`), and so does anything else you throw
away. A fitted L_inf near 0 usually means the fit is extrapolating badly, not that the
task is easy.

**α and β** are not "importance" — they are *rates of diminishing return*. Large α means
loss drops quickly as you add parameters, so the benefit of more parameters is exhausted
quickly. Small β means loss drops slowly with data, so data keeps paying, slowly, for a
long time.

That has a consequence people get backwards. Minimise L subject to C = 6ND:

> N ∝ C^(β/(α+β)),  D ∝ C^(α/(α+β))
>
> *Sketch:* Lagrange gives αA·N^(−α) = βB·D^(−β), so D ∝ N^(α/β); substitute into
> ND ∝ C. Do this derivation yourself before writing `costmodel.py` — it is the whole
> module in two lines.

Check it against Chinchilla: α ≈ 0.34, β ≈ 0.28 → N ∝ C^0.45, D ∝ C^0.55, which is
Hoffmann's "scale both about equally". Now put in the reference constants for the target
note, α = 0.677, β = 0.077:

> **N ∝ C^0.10,  D ∝ C^0.90**

So "α ≫ β" — the structure claim C1 — means that in this regime almost all marginal
compute should go into **data**, not parameters. *Carpe datum.* And in HEP, data is not
scraped, it is *simulated*, at real CPU cost per jet. That is exactly the k·D term in the
extended cost model, and it is why C = 6ND is the wrong budget for a physics
collaboration. The whole arc of the project is in that chain.

---

## 4. The three approaches (so you know which one you are doing)

Hoffmann et al. estimate the compute-optimal frontier three ways:

1. **Minimum over training curves** — fix model sizes, vary tokens, take the envelope.
2. **IsoFLOP profiles** — fix several compute budgets, sweep model size within each, find
   the minimum of each parabola.
3. **Parametric loss fit** — fit L(N, D) to all runs and derive everything analytically.

**You are doing approach 3**, and it is the one Besiroglu et al. showed is the most
fragile: it fits 5 parameters to few points, over a narrow range, using an optimiser that
can land in different basins from different starts. Approaches 1 and 2 need far more runs
than a laptop weekend allows. This is a known, accepted, pre-registered weakness — which
is why the init-sensitivity spread is a *reported result* and not a footnote.

---

## 5. Six ways this goes wrong

### 5.1 The learning-rate schedule (the Kaplan/Chinchilla lesson)
If your cosine schedule is longer than the run, the run stops mid-decay at an inflated
loss. Your D axis spans 24×, so every column is a different run length. Get this wrong
and short runs look artificially bad — a pure schedule artefact that lands directly on
**β**. `configs/grid.yaml` sets `schedule: cosine_to_end`, T_max = *that run's* steps.
This one bug is thought to be a large part of why Kaplan's and Hoffmann's conclusions
differed.

### 5.2 Counting N wrong
Fitting a power law in N and then feeding it a nominal parameter count is a silent
disaster. Count the *exact* number, embeddings and biases included, from the built model.
With four N values spanning ~1.7 dex, a 10% systematic error in N is not noise.

### 5.3 The evaluation set moving
Every loss in the fit must be measured on the *same jets*. Resample the eval slice
between runs and you add noise that looks like curvature in the surface. Fix it once,
assert its size and label sum in every run.

### 5.4 Masking bugs that leak the label
Two masks matter: attention (padded keys → −inf before softmax) and mean-pool (divide by
the real constituent count). Break either and the network can read *how many pad slots
this jet has* — which correlates with jet type. You get a suspiciously good tagger and a
scaling law about nothing. Test: shuffle constituents within a jet; logits must not move.
Test: pad the same jet to different lengths; logits must not move.

### 5.5 Noise-limited grids
With 16 points and a 5-parameter fit, seed noise can dominate the signal you are fitting.
That is why 6 of the 22 runs are seed repeats. If σ_seed at a cell is comparable to the
loss difference *between* adjacent grid cells, the honest conclusion is "this grid cannot
resolve the exponents" — and you report that, with the numbers.

### 5.6 Dataloader-bound small runs
The `tiny` model on 50k jets does so little arithmetic that pandas/HDF5 overhead can
dominate the step. Wall time then measures your I/O, not the model, and any FLOP-based
claim built on it is wrong. Build the float16 feature cache first; then compare measured
throughput across cells and report if it varies more than ~2×.

---

## 6. The build ladder

Each rung: what to write, how to know it works, and a question. If you cannot answer the
question, the rung is not finished — the code running is not the same as the rung being
done.

### 6.0 — Scaffold *(done)*
Repo, environment, configs, pre-registration committed before any training.
**Q:** Why must the pre-registration be the *first* commit and not a Day-2 write-up?

### 6.1 — Data (`data.py`, notebook 01) — build step 2
Read HDF5 row slices → 4-vectors → 7 features + mask. Then the feature cache, then the
nested D-subsampling.
**Checks:** plot one jet in (Δη, Δφ) with marker size ∝ pT — a top jet should look
three-prong-ish, QCD more like one blob. Constituent multiplicity histogram. Fraction of
jet pT captured by the leading 64 constituents (this number settles `D-003`).
Reconstructed jet mass from the summed four-vectors should peak near 175 GeV for signal
and be steeply falling for background — that single plot validates your whole feature
pipeline at once.
**Q:** Your Δφ is a difference of angles. What did you do at the ±π wrap, and how would
you *see* it in the data if you got it wrong?

### 6.2 — Model (`model.py`) — build step 3
Pre-LN set transformer, no positional embedding, masked mean-pool, one logit.
**Checks:** `permutation_invariance_check` < 1e-5. Parameter counts within ~25% of the
ladder targets. Overfit 1,000 jets to near-zero loss — if it cannot, the bug is in the
model or the masking, and you find it now rather than after 22 runs.
**Q:** Why does a *set* transformer suit a jet, and what exactly would a positional
embedding let the model learn that you do not want it to?

### 6.3 — Train (`train.py`) — build step 3
One config → one immutable run JSON. Schedule resolved from the run's own length.
**Checks:** same seed twice → identical loss. Plot the realised LR against step for the
shortest and the longest run: both must complete a full cosine.
**Q:** Your run JSON records `params_exact` and `tokens_seen`. Which fit input would be
corrupted if each were wrong, and by how much?

### 6.4 — Sweep (`sweep.py`) — build step 4
Four cheapest cells first, debug, then all 22. Resumable, never overwriting.
**Check:** loss should fall monotonically along each row and each column. A
non-monotonicity is either seed noise (compare against σ_seed) or a bug — decide which,
in writing, before continuing.
**Q:** One cell comes out clearly off-trend. What does the pre-registration permit you to
do about it, and what does it forbid?

**Budget arithmetic, so you know what you are signing up for.** With 6·N·D·64 FLOPs per
cell (64 tokens per jet), the 16-cell grid costs ~3.0e15 FLOPs and the 6 seed repeats add
~4.6e14, so **~3.5e15 FLOPs total** — of which the single `large` × 1.2M cell is ~1.4e15,
40% of everything. An M4 GPU is ~4 TFLOP/s peak fp32; small transformers on MPS realise
maybe 10–25% of that, so expect **roughly 2–6 hours of wall clock** for the whole sweep,
dominated by that one cell. Two consequences: run the four cheapest cells first, and get
your real throughput number from the 1,000-jet overfit in rung 6.2 (steps/second × the
step counts here) before committing to the `large` row — the fallback to a 3×4 grid is
pre-committed, and it is much cheaper to take it on Day 1 than on Day 2.

### 6.5 — Cost model (`costmodel.py`, notebook 02) — build step 5, no GPU
C = 6ND + kD + mN; isoFLOP curves; compute-optimal frontier at (k,m) = (0,0), (0,1e9),
(1e9,0).
**Check:** at k = m = 0 your numerical optimum must match the closed form of §3 to ~1e-6.
That is a free, exact test — write it as an assertion, not a plot you eyeball.
**Q:** For α, β ∈ (0,1) the *signs* in claim C3 can be derived on paper. So what is C3
actually testing — and what would you have to report for it to be interesting rather than
merely correct?

### 6.6 — Fit (`fit.py`, `tests/test_fit.py`) — build step 6
Huber on log residuals, 4500 starts, bootstrap. **`tests/test_fit.py` first**: generate
(N, D, L) from known (α, β), add realistic noise, and require recovery inside the
bootstrap CI. A fitter that cannot recover synthetic truth cannot be trusted on real data
— and on 16 real points you have no way to notice it is lying.
**Checks:** count how many of the 4500 starts reach the best objective. Re-fit with 1%
noise added to the losses and see how far α moves.
**Q:** Your bootstrap resamples 16 grid points with replacement. What does that CI
actually cover — and which sources of uncertainty in your α does it entirely miss?

### 6.7 — Results and write-up (notebook 03, README) — build steps 7–8
Final fits, CIs, falsification, extrapolation, two headline figures, mini-paper.
**Q:** Suppose C1 fails — α/β comes out at 1.4 with a CI straddling 1. Write the three
most plausible explanations *now*, before you know the answer. That list is worth more
than a confirmation.

---

## 7. Symbols

| | |
|---|---|
| **N** | trainable parameters, exact, embeddings included |
| **D** | training jets (not constituents, not tokens — `tokens_seen = D × 64` is recorded separately) |
| **C** | cost. 6ND is the FLOP convention; the extended model adds kD + mN |
| **L** | mean BCE on the fixed eval slice, in **nats** |
| **L_inf, α, β** | irreducible loss, model exponent, data exponent |
| **k, m** | per-example and per-parameter cost coefficients, in FLOP-equivalents |
| **σ_seed** | std of final loss across seeds at a fixed (N, D) |

---

## 8. Open questions I could not settle from the outside

1. **Does the note fit this exact functional form?** L_inf + A/N^α + B/D^β is assumed from
   `CLAUDE.md`. Unverified (`D-002`).
2. **What are A and B in the reference?** Not available, so the reference curve in
   notebook 02 can only be anchored or plotted shape-invariantly.
3. **Is the reference D measured in jets or in constituents/tokens?** This changes β's
   meaning by a constant factor inside the log and is worth one sentence of the note.
4. **Does 2602.15781 report α and β for JetClass boosted jet tagging?** If yes, that is a
   far better structural comparison for our task than the flavour-tagging note, and the
   Day-2 write-up should use both.
