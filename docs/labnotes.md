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

### PREDICTION 2026-09-05 — before writing any of the pipeline

Fill these in before `read_four_vectors` returns its first array. Numbers, not adjectives.

| Quantity | My prediction | Confidence | Actual |
|---|---|---|---|
| Median fraction of jet pT in the leading 64 constituents | | | |
| L_inf (irreducible BCE, nats) | | | |
| α (model exponent) | | | |
| β (data exponent) | | | |
| σ_seed at (tiny, 150k) | | | |
| Which functional form wins the LOO test | | | |
| Will the residuals show an N×D interaction pattern? | | | |
| Wall clock for the full 22-run sweep | | | |

Then, the two that matter most, in prose:

**What would have to be true for claim C1 to fail?** (write it now, while you have no
stake in the answer)

**If α comes out near β — same order, CI on α−β straddling zero — what are the three
most likely causes, ranked?**
