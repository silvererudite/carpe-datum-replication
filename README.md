# Jet Scaling Laws — Replication

**Status: Day 1, step 1 of 8 complete (scaffold + environment + pre-registration).
No training has been run. No results exist yet.**

A rigor-first replication of the scaling-law methodology in **Hartman, Vigl et al.,
*Carpe Datum: Scaling behavior of transformers for heavy hadron flavor identification*,
ATL-SOFT-PUB-2026-002** ([CDS 2953659](https://cds.cern.ch/record/2953659)), executed on
the **public Top Quark Tagging Reference Dataset** instead of ATLAS-internal data.

We test the *qualitative structure* of the fitted scaling law — is the model exponent
much larger than the data exponent, does the power-law form beat an exponential, how do
extended cost models deform the compute-optimal frontier — not the numerical values.

This README becomes the mini-paper on Day 2. Until then, the documents that matter are:

| Document | What it is |
|---|---|
| [`docs/preregistration.md`](docs/preregistration.md) | **Read first.** The claims, the pass/fail criteria, and the full analysis plan — frozen before any training run. |
| [`docs/orientation.md`](docs/orientation.md) | The landscape: the five papers this sits on, where the methodology is fragile, and a build ladder with self-checks. |
| [`docs/deviations.md`](docs/deviations.md) | Every departure from the reference methodology, classified. Already non-empty. |
| [`CLAUDE.md`](CLAUDE.md) | The project brief and non-negotiable rules. |

## Setup

```bash
uv sync                     # Python 3.11 venv + all dependencies
uv run pytest               # scaffold self-check
uv run python -c "import torch; print(torch.backends.mps.is_available())"
```

## Data

```bash
uv run python scripts/get_data.py --all      # ~1.7 GB from Zenodo, md5-verified
```

Top Quark Tagging Reference Dataset (Kasieczka, Plehn, Thompson, Russel 2019),
Zenodo DOI [10.5281/zenodo.2603256](https://doi.org/10.5281/zenodo.2603256),
companion paper [arXiv:1902.09914](https://arxiv.org/abs/1902.09914). **Cite both.**

> The `pd4ml` package (arXiv:2107.00656) also ships this dataset, but **all of its
> download links are dead as of 2026-09-05** — see `docs/deviations.md` D-001 for the
> evidence and the fallback shim.

## Layout

```
docs/        pre-registration, orientation, deviations log
configs/     the sweep grid and the model-size ladder -- all hyperparameters live here
src/         jetscaling package: data, model, train, sweep, fit, costmodel
notebooks/   01 sanity checks, 02 cost model, 03 results
results/     runs/ (one immutable JSON per training run), figures/
tests/       test_fit.py -- the fitter must recover known synthetic exponents
scripts/     get_data.py
```

## Citations

- Kasieczka, Plehn, Thompson, Russel, *Top Quark Tagging Reference Dataset*, Zenodo 10.5281/zenodo.2603256 (2019)
- Kasieczka et al., *The Machine Learning Landscape of Top Taggers*, [arXiv:1902.09914](https://arxiv.org/abs/1902.09914)
- Hartman, Vigl et al., ATL-SOFT-PUB-2026-002, [CDS 2953659](https://cds.cern.ch/record/2953659)
- Vigl, Hartman, Kagan, Heinrich, *Neural Scaling Laws for Boosted Jet Tagging*, [arXiv:2602.15781](https://arxiv.org/abs/2602.15781)
- Hoffmann et al., *Training Compute-Optimal Large Language Models*, [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- Besiroglu et al., *Chinchilla Scaling: A Replication Attempt*, [arXiv:2404.10102](https://arxiv.org/abs/2404.10102)
