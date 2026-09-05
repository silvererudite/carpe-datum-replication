"""Data pipeline: HDF5 -> (features, mask, label), and the D-axis subsampling.

STATUS: contract only. Implement in build step 2/3 (see docs/orientation.md 6.1).

------------------------------------------------------------------------------
The raw format (Zenodo 10.5281/zenodo.2603256) -- VERIFIED 2026-09-05 on test.h5
------------------------------------------------------------------------------
Pandas `frame_table` (pandas_version 0.15.2) under key "table". 806 columns:

    E_0, PX_0, PY_0, PZ_0, ..., E_199, PX_199, PY_199, PZ_199   (800 float32)
    truthE, truthPX, truthPY, truthPZ                           (4 float32)
    ttv, is_signal_new                                          (2 int64)

Constituents are pT-ordered, highest first, and zero-padded: a constituent is real
iff E > 0. `is_signal_new`: 1 = top, 0 = QCD. test.h5 holds 404,000 rows.

TWO READ PATHS, both verified to give identical values:

1. `pandas.read_hdf(path, key="table", start=i, stop=j)` -- the documented route.
   REQUIRES pandas < 3: pandas 3.x raises `TypeError: a bytes-like object is
   required, not 'str'` on this file's bytes-valued `pandas_type` attribute.
   Pinned in pyproject.toml (deviation D-007).

2. PyTables directly -- ~10x faster and what the feature cache should use:

       with tables.open_file(path) as f:
           rows = f.root.table.table.read(start, stop)
       p4     = rows["values_block_0"][:, :800].reshape(-1, 200, 4)   # (E,px,py,pz)
       labels = rows["values_block_1"][:, 1]                          # is_signal_new

   Block column order was verified identical to the DataFrame column order, so the
   reshape above is exact. 200k rows read in ~0.2 s.

h5py CANNOT read these files: the table is blosc-compressed and h5py does not ship
the filter plugin ("can't open directory .../hdf5/.../plugin"). PyTables does.

------------------------------------------------------------------------------
The features we build (fixed by the pre-registration -- do not add features later)
------------------------------------------------------------------------------
For each surviving constituent i, with the jet four-vector p_jet = sum_i p_i:

    pT_i    = hypot(px_i, py_i)
    eta_i   = arcsinh(pz_i / pT_i)
    phi_i   = arctan2(py_i, px_i)
    dEta_i  = (eta_i - eta_jet) * sign(eta_jet)     # sign flip: fold the two hemispheres
    dPhi_i  = wrap_to_pi(phi_i - phi_jet)
    dR_i    = hypot(dEta_i, dPhi_i)

    f_i = [ log pT_i, log E_i, dEta_i, dPhi_i, log(pT_i/pT_jet), log(E_i/E_jet), dR_i ]

That 7-feature set is the ParticleNet/pd4ml convention (pd4ml
`preprocessing_utils._transform`), which keeps us comparable to the published
top-tagging literature. F = 7 is recorded in every run JSON.

Zero-padded slots must be masked, not zero-filled-and-ignored: log(0) = -inf will
poison the mean-pool if you forget.

------------------------------------------------------------------------------
Why a feature cache
------------------------------------------------------------------------------
The small models in this sweep are dataloader-bound, not compute-bound: recomputing
these features from 4-vectors every epoch would make the tiny-N runs measure pandas
throughput rather than model quality -- and that would show up as a fake data
exponent. Build the features ONCE into a float16 memmap and have every run read
slices of it. Cache key must include (max_constituents, feature list version).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    "log_pt",
    "log_e",
    "delta_eta",
    "delta_phi",
    "log_pt_rel",
    "log_e_rel",
    "delta_r",
)
N_FEATURES = len(FEATURE_NAMES)


@dataclass(frozen=True)
class JetArrays:
    """One materialised slice of the dataset."""

    features: np.ndarray  # (n_jets, max_constituents, N_FEATURES) float32
    mask: np.ndarray  # (n_jets, max_constituents) bool -- True = real constituent
    labels: np.ndarray  # (n_jets,) int8 -- 1 = top, 0 = QCD


def read_four_vectors(
    path: str | Path, start: int = 0, stop: int | None = None, max_constituents: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Read a row slice as (p4, labels).

    Returns
    -------
    p4 : (n, max_constituents, 4) float32, ordered (E, px, py, pz), pT-descending,
         zero-padded. Truncated to the leading `max_constituents` (deviation D-003).
    labels : (n,) int8 from `is_signal_new`.
    """
    raise NotImplementedError("build step 2 -- see module docstring for the format")


def build_features(p4: np.ndarray) -> JetArrays:
    """Turn (n, C, 4) four-vectors into the 7-feature representation + mask.

    Vectorise this; a per-jet Python loop over 1.2M jets will dominate the sweep.
    Verify against the formulas in the module docstring, and sanity-check that
    `delta_r` is < ~0.8 for essentially every constituent (anti-kT R=0.8 jets).
    """
    raise NotImplementedError("build step 2")


def build_feature_cache(
    h5_path: str | Path,
    cache_path: str | Path,
    max_constituents: int = 64,
    chunk_rows: int = 50_000,
) -> dict:
    """Stream the HDF5 file through `build_features` into a float16 memmap.

    Writes `cache_path` (features), `cache_path.with_suffix('.mask.npy')`,
    `cache_path.with_suffix('.labels.npy')`, and a JSON sidecar recording
    n_jets, max_constituents, FEATURE_NAMES, source md5 and the code version.
    Returns the sidecar dict.
    """
    raise NotImplementedError("build step 2")


def subsample_indices(
    n_total: int, d: int, seed: int, nested: bool = True, all_d: tuple[int, ...] = ()
) -> np.ndarray:
    """Pick the D jets for one grid column.

    `nested=True` means the D=50k set is a strict subset of the D=150k set, and so on:
    draw ONE permutation with `seed` and take prefixes. This removes draw-to-draw
    variation from the data axis, so a difference between two D columns is a data-size
    effect and not a lucky sample. The seed here is `data.subsample_seed` from
    configs/grid.yaml and must NOT be the run seed.
    """
    raise NotImplementedError("build step 2")


def fixed_eval_slice(cfg: dict) -> JetArrays:
    """The evaluation set: one fixed slice of test.h5, IDENTICAL for all 22 runs.

    This is the single most load-bearing invariant in the sweep -- every loss value in
    the fit is a number measured on exactly these jets. Assert the returned shape and
    the label sum against values recorded in the run JSON of the first run.

    GATE, ALREADY MEASURED (2026-09-05): signal fraction of the first 200,000 rows of
    test.h5 is 0.4994 -- PASSES the pre-registered [0.45, 0.55] window, so the
    head-of-file slice is legitimate and the seed-pinned fallback is not triggered.
    Assert this value (tolerance ~1e-3) at load time so a future file swap is caught.

    Caveat worth knowing: the file is shuffled but not perfectly. Per-10k-block signal
    fractions range 0.479-0.546, a spread ~3x wider than i.i.d. shuffling predicts.
    Irrelevant for a 200k slice; a real trap if you grab the first 1k rows for a quick
    debug set and get a lopsided sample. Shuffle small debug slices yourself.
    """
    raise NotImplementedError("build step 2")
