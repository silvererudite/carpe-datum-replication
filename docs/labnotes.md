# Lab notes — predictions, surprises, decisions

A running log, newest last. Three kinds of entry, and the first kind is the one that
makes the other two worth anything.

**PREDICTION** — written *before* a measurement, with numbers and a confidence.
Unfalsifiable predictions ("it'll probably scale") are not entries. A prediction you
would not be embarrassed to be wrong about is not specific enough.

**SURPRISE** — a measurement that violated a prediction. Record the gap, then the
candidate explanations *before* you test any of them. This is where the research is:
a result that matches expectation adds a decimal place, a result that violates one
adds knowledge.

**DECISION** — a judgement call and its reasoning. If it changes the pre-registered
plan it also goes in `deviations.md`; this file records *why*, that file records *what*.

Why bother: surprise is only detectable against a stated expectation. Without this file
you will read every result as "about what I thought", because memory reconstructs
predictions to match outcomes. That is not a character flaw, it is how memory works —
which is why the prediction has to be written down before the number exists.

---

## Template

```
### [PREDICTION | SURPRISE | DECISION] YYYY-MM-DD — one-line title
**Context:** what I was about to do / had just done.
**Claim:** the specific statement, with numbers.
**Confidence:** low / medium / high, and what would change it.
**Outcome:** (filled in later — leave blank when writing a PREDICTION)
```

---

## Entries

### How to fill a prediction row (worked example — not one of your predictions)

The unit of a prediction is a **range**, not a number. State an interval you are ~90%
sure contains the truth, plus a best guess inside it. A surprise is then defined
mechanically: the actual lands outside your range.

That definition is what makes this calibratable. Over the project you will make ~10
predictions; roughly 1 should land outside its 90% range. **Never** wrong means your
ranges are so wide they assert nothing. **Often** wrong means you are overconfident.
Both are worth knowing about yourself before you start interpreting fits.

Worked example — L_inf, reasoned from things already known:

> **Bracket it from above and below.** BCE in nats on a balanced binary task: chance is
> ln 2 = 0.693, perfect is 0. So L_inf lies in (0, 0.693) and — since L_inf is the
> asymptote — strictly below the best loss any of my 22 runs achieves.
>
> **Find an anchor.** Strong published top taggers on this exact dataset reach ~94%
> accuracy / ~0.986 AUC. A calibrated classifier at that accuracy has BCE roughly
> 0.15–0.25. Those taggers are large, use up to 200 constituents, and are trained on the
> full 1.2M — so they are an upper bound on what the *task* permits, not a floor.
>
> **Adjust for my handicaps.** I cap at 64 constituents (throws away real substructure
> information, raises the floor) and use kinematics only. But L_inf is the
> infinite-N, infinite-D limit, which is *lower* than anything I will measure.
> These push in opposite directions.
>
> **Do NOT anchor on the reference's 0.619.** That is a different task and possibly a
> different number of classes — if flavour tagging is 3-class, its chance baseline is
> ln 3 = 1.099, not 0.693, and the two L_inf values are not on the same scale at all.
> (Worth confirming when the note is read — see D-002.)
>
> → **90% range 0.08–0.30, best guess 0.17, confidence low.**

The reasoning line matters more than the number. When you are later wrong, the line tells
you *which step* of the reasoning failed — and that is the finding.

---

## Entries

### PREDICTION — before writing any of the pipeline

Fill this in before `read_four_vectors` returns its first array. Ranges, not adjectives.
Twenty minutes. Low confidence is a fine answer; a blank row is not.

The **kind** column says what sort of reasoning each row needs. Eight of the ten are ML,
statistics or systems questions — physics enters in exactly two places, and one of those
you can measure in five minutes.

| Quantity | Kind | 90% range | Best guess | One-line reasoning | Actual |
|---|---|---|---|---|---|
| Median fraction of jet pT in the leading 64 constituents | physics (but measurable in 5 min) | | | | |
| L_inf (nats) | physics anchor + ML | | | | |
| α (model exponent) | ML / scaling laws | | | | |
| β (data exponent) | ML / scaling laws | | | | |
| α/β | ML / scaling laws | | | | |
| σ_seed at (tiny, 150k), nats | ML / noise budget | | | | |
| Loss gap between adjacent cells in one row, nats | ML / arithmetic | | | | |
| Which form wins the LOO test, and on how many of 16 | statistics | | | | |
| Fraction of the 4500 fit inits reaching the best objective | optimisation | | | | |
| Wall clock for all 22 runs, hours | systems | | | | |

Anchors you already have (use them, adjust from them, say which way and why):

- chance-level BCE = ln 2 = 0.693; your N range spans 1.74 dex, your D range 1.38 dex
- Kaplan et al.: α_N ≈ 0.076, α_D ≈ 0.095 — LLMs, huge N range, known schedule bug
- Chinchilla: α ≈ 0.34, β ≈ 0.28 — the "scale both equally" regime
- the reference note: α = 0.677, β = 0.077 — flavour tagging, much larger scale
- rows 6 and 7 together decide whether your grid can resolve anything at all

### Second worked example — σ_seed, by decomposing the noise budget

No physics in this one at all. Where can run-to-run variation in final eval loss come
from?

1. **Eval-set sampling noise.** Per-jet BCE has a spread of order 0.5 nats, so the
   standard error of the mean over 200k jets is ~0.5/sqrt(2e5) ≈ 0.001 nats. But the eval
   set is *fixed and identical across runs*, so this is a common offset, not run-to-run
   variance — **it cancels in every comparison you care about.** (That is a second, less
   obvious reason the fixed eval slice matters.)
2. **Init + shuffle order.** What is actually left. For a small model on a short
   single-epoch run, this is the whole budget.
3. **Nondeterministic kernels.** MPS reductions are not bit-reproducible; tiny, and
   swamped by (2).

So σ_seed ≈ the training-stochasticity term alone, and your prior for that comes from
every small model you have ever trained twice. Note the direction: **short runs are
noisier**, so σ_seed at (tiny, 150k) is an upper bound for the better-resolved cells.

Then do the arithmetic that actually matters. If β ≈ 0.1 and the D-term contributes
~0.3 nats, then across the 24× D range the loss moves by 24^0.1 ≈ 1.37×, i.e. ~0.09 nats
total, ~0.03 per adjacent column. Against σ_seed of 0.005 that is a signal-to-noise of 6
— resolvable. Against σ_seed of 0.02 it is 1.5, and your grid measures almost nothing.
**Two guesses, one division, and you know whether the experiment can work.** Do this
before running it, not after.

Two in prose, and these are the ones worth the most:

**What would have to be true for C1 to fail?** Write it now, while you have no stake in
the answer.

**If α comes out near β — CI on α−β straddling zero — what are the three most likely
causes, ranked?** Ranking them now, before you know, is what stops the first plausible
story from becoming the explanation later.
