# Deviations log

Every departure from the reference methodology, or from `CLAUDE.md`, with a
classification. Append-only: entries are added and amended with dated notes, never
deleted.

**Classification**
- **benign** — cannot plausibly change any pre-registered conclusion.
- **potentially consequential** — could move a number or an exponent; must be discussed
  in the README results section.
- **unknown** — cannot be assessed without the measurement.

| ID | Date | Deviation | Class |
|---|---|---|---|
| D-001 | 2026-09-05 | Dataset fetched from Zenodo, not the `pd4ml` package | benign |
| D-002 | 2026-09-05 | Different task (top tagging, not flavour tagging); reference constants taken from `CLAUDE.md`, not read from the note | potentially consequential |
| D-003 | 2026-09-05 | Constituents capped at 64 (of up to 200) | potentially consequential |
| D-004 | 2026-09-05 | Model-ladder widths recomputed; `CLAUDE.md` examples undershoot their targets | benign |
| D-005 | 2026-09-05 | Apple M4 / MPS / fp32 instead of a CUDA GPU | unknown |
| D-006 | 2026-09-05 | Added `tables` (PyTables) to the dependency list | benign |
| D-007 | 2026-09-05 | `pandas` pinned to < 3; pandas 3.x cannot read this dataset | benign |

---

## D-001 — Data source: Zenodo, not `pd4ml`

**Class: benign.** *(It is the same dataset, from its canonical home.)*

`pd4ml` ("Physics Data for Machine Learning", the package released with
[arXiv:2107.00656](https://arxiv.org/abs/2107.00656),
<https://github.com/erum-data-idt/pd4ml>) ships this dataset as `TopTagging`, and its
`TopTagging.load()` API would have been the more convenient route. It was evaluated and
rejected on evidence:

1. **Every download link in the package is dead.** All five datasets are served from
   DESY sync-and-share links hardcoded in `pd4ml/pd4ml.py`. Checked 2026-09-05:

   ```
   404  .../TWqT3E9j6q5yYSF/download/1_top_tagging_2M.npz   (TopTagging)
   404  .../RN9AYWLgKHEPEfF/download/2_spinodal_29k.npz     (Spinodal)
   404  .../NXojejGjKdSJSR2/download/3_EOSL_or_EOSQ_180k.npz (EOSL)
   ```

   The share root itself 404s. This has happened before — issue
   [#27](https://github.com/erum-data-idt/pd4ml/issues/27) ("Unable to download
   datasets?") reports the same failure in January 2023, answered by the maintainer with
   "our server structure changed and therefore the links expired", fixed in commit
   *"fixed downloads"* on 2023-01-19. That is the **last commit to the repository**; the
   links have since expired again with no maintainer activity for 3.5 years. The package
   is also not on PyPI.
   Re-check any time with `uv run python scripts/get_data.py --probe-pd4ml`.

2. **Its preprocessing path will not install on Python 3.11.** `pd4ml.load_data` (the
   `graph=True` / feature-building route) imports `awkward` 0.x `JaggedArray` and
   `uproot-methods`, both unmaintained since 2020 and unavailable for modern Python.
   Only the raw `Dataset.load()` path (numpy + requests + rich + pandas) is usable.

3. **Even if it worked, the shape is worse for this study.** `pd4ml` distributes one
   `.npz` holding all ~2M jets as a dense `(n, 200, 4)` float array. Materialising the
   1.6M-jet training split costs ~5 GB of RAM in one allocation, on a 16 GB machine,
   with no way to read a row range — whereas the Zenodo files are pandas HDF5 tables
   that `pandas.read_hdf(..., start=, stop=)` slices lazily, which is exactly what the
   D-axis subsampling needs.

4. **`pd4ml` also merges the original train and validation splits** into a single
   `train`, exposing the original assignment only through a `ttv` flag array. The
   Zenodo files keep the official three-way split.

The Zenodo files are the same data these arrays were derived from, so the physics is
unaffected: `pd4ml`'s array is `(E, px, py, pz)` per constituent, pT-ordered and
zero-padded — identical in content and ordering to the `E_i, PX_i, PY_i, PZ_i` columns
of the Zenodo HDF5 (see `pd4ml/preprocessing_utils.py::convert`, which reconstructs
exactly those column names from the array).

`scripts/get_data.py` keeps the pd4ml URL and md5 and a one-command liveness probe, so
switching back is a small edit if the mirror is ever restored.

## D-002 — Different task; reference constants unverified against the source

**Class: potentially consequential.**

Two distinct issues, filed together because they share a cause (the target note has not
been read directly).

*Task.* ATL-SOFT-PUB-2026-002 studies **heavy-hadron flavour identification** (b/c/light
jets, with track and vertex information). This replication studies **boosted top vs QCD
tagging** from constituent four-vectors. Different label semantics, different intrinsic
difficulty, different irreducible error — so L_inf, A, B, and plausibly α and β, have no
reason to match numerically. Only the *structure* of the fit is pre-registered (C1–C3).

*Provenance.* L_inf = 0.619, α = 0.677, β = 0.077 are taken from `CLAUDE.md`. They have
not been checked against the note (<https://cds.cern.ch/record/2953659>), and the
functional form the note actually fits has not been verified to be
L_inf + A/N^α + B/D^β. A and B are not available at all, which is why notebook 02 either
anchors the reference curve to a stated reference point or plots only the shape-invariant
quantities — recorded in the notebook.

**Action before Day 2 write-up:** read the note, and amend this entry with what was
confirmed and what was not. The closest public sibling,
[arXiv:2602.15781](https://arxiv.org/abs/2602.15781) (same group, JetClass, boosted jet
tagging), is a better structural comparison for our task and should be read alongside it.

## D-003 — Constituents capped at 64

**Class: potentially consequential.**

Jets carry up to 200 constituents; we keep the leading 64 by pT. Attention is O(C²), so
64 vs 200 is roughly a 10× difference in attention cost per jet — this is what makes 22
runs feasible on one laptop.

Why it may matter: the soft, wide-angle radiation that the tail of the constituent list
represents is not noise for top tagging — it carries some of the three-prong substructure
signal. Truncation can therefore **raise L_inf** (a harder task) and could plausibly
change how much extra data helps, i.e. β. Published top taggers commonly use 100–200
constituents, so our absolute losses are not comparable to the literature — which the
no-leaderboard-chasing rule already accepts.

Because the cap is identical for all 22 runs, it shifts the whole loss surface rather
than tilting one axis, so the *comparisons* the claims rest on stay valid.

**Cheap check worth doing:** in notebook 01, plot the distribution of constituent
multiplicity and the fraction of jet pT captured by the leading 64. If that fraction is
> 99% for nearly all jets, downgrade this entry to benign with the number quoted.

## D-004 — Model-ladder widths recomputed

**Class: benign.**

`CLAUDE.md` suggests (2, d=32), (3, d=64), (4, d=96), (6, d=128) for ~50k / 200k / 800k /
3M parameters. Under the block used here — 4d² attention + 8d² feed-forward ≈ 12d² per
layer — those give ~25k, ~147k, ~442k, ~1.2M: the top rung is off by 2.5×, which would
compress the N lever the whole C1 claim depends on. Widths were recomputed
(`configs/model_sizes.yaml`) to (2, 48), (3, 72), (4, 128), (7, 192) ≈ 5.6e4, 1.9e5,
7.9e5, 3.1e6 — roughly 0.57 dex apart. The **measured** count goes into every run JSON
and into the fit; the targets are only the plan.

## D-005 — Apple M4 / MPS / fp32

**Class: unknown.**

`CLAUDE.md` assumes a CUDA GPU (`nvidia-smi`, CUDA index URL). This machine is an Apple
M4 with 16 GB unified memory; PyTorch runs on the MPS backend in fp32 (bf16 autocast on
MPS is not trusted here).

Why unknown rather than benign: MPS kernels are not bit-identical to CUDA, small models
can be *slower* on MPS than on CPU because of dispatch overhead, and 16 GB of unified
memory is shared with the OS. None of that should change a converged loss by more than
run-to-run seed noise — but "should" is exactly what the seed-variance check (§6.3 of the
pre-registration) exists to measure. Resolve this entry once σ_seed is known.

The practical risk is wall-clock: if the `large` × 1.2M cell does not fit the two-day
budget, the pre-committed fallback is to drop the row and fit a 3×4 grid.

## D-006 — `tables` added to the dependency list

**Class: benign.**

`CLAUDE.md`'s package list omits `tables` (PyTables), but `pandas.read_hdf` — the
recommended read path for these pandas-format HDF5 files — requires it. Added to
`pyproject.toml`.

## D-007 — `pandas` pinned to < 3

**Class: benign.**

`CLAUDE.md` calls `pandas.read_hdf` "the easiest path". It is — but not on pandas 3.x.
These files are pandas `frame_table`s written with pandas 0.15.2 (2015), whose HDF5
attributes are byte strings; pandas 3.0.5 does `if "table" not in pt` against those bytes
and raises `TypeError: a bytes-like object is required, not 'str'`. Verified on
`data/test.h5`, 2026-09-05. pandas 2.3.3 reads the same file correctly, so
`pyproject.toml` pins `pandas>=2.2,<3`.

Separately, and worth knowing before reaching for it: **h5py cannot read these files at
all.** The table is blosc-compressed and h5py does not ship the filter plugin — it fails
with `can't open directory .../hdf5/2.0.0-arm64/lib/plugin`. PyTables ships blosc and
reads it fine, which is the real reason `tables` is a hard dependency (D-006), not just a
pandas backend.

The fast path the feature cache should use is PyTables directly (~10× faster than
`read_hdf`, 200k rows in ~0.2 s); the block column order was verified identical to the
DataFrame column order, so `values_block_0[:, :800].reshape(-1, 200, 4)` is exact. Both
paths were cross-checked to return identical values. Details in `src/jetscaling/data.py`.

---

# Verified at scaffold time

Facts checked against the real files on 2026-09-05, before any training. These are not
deviations — they are the assumptions the scaffold rests on, confirmed rather than
assumed.

| Assumption | Status |
|---|---|
| HDF5 key is `table`, format is `frame_table` | confirmed |
| 806 columns: 800 constituent + `truthE/PX/PY/PZ` + `ttv` + `is_signal_new` | confirmed |
| Constituent components ordered (E, px, py, pz), pT-descending, zero-padded | confirmed |
| `test.h5` row count | 404,000 (not exactly 400k) |
| **Eval-slice gate:** signal fraction of the first 200,000 rows of `test.h5` | **0.4994 — PASS** ([0.45, 0.55] window; head-of-file slice is legitimate, seed-pinned fallback not triggered) |
| File is shuffled | yes, but imperfectly: per-10k-block signal fraction spans 0.479–0.546, ~3× wider than i.i.d. shuffling predicts. Harmless at 200k; do not take a small contiguous slice as a balanced debug set. |
| Zenodo md5s match the record | `test.h5` confirmed; `train.h5`/`val.h5` pending download completion |
