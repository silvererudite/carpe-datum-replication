"""Data pipeline: HDF5 -> (features, mask, label), and the D-axis subsampling.

STATUS: contract only. Implement in build step 2/3 (see docs/orientation.md 6.1).

------------------------------------------------------------------------------
The raw format (Zenodo 10.5281/zenodo.2603256)
------------------------------------------------------------------------------
train.h5 / val.h5 / test.h5 are *pandas* HDF5 tables (key "table"), so
`pandas.read_hdf(path, key="table", start=i, stop=j)` gives you row slices without
loading 1.2M jets into RAM. Columns per jet:

    E_0, PX_0, PY_0, PZ_0, ..., E_199, PX_199, PY_199, PZ_199   (800 float32)
    truth-top four-momentum, ttv, is_signal_new                 (metadata, 6 columns)

(806 columns total. The Zenodo record documents the truth-momentum columns only
informally as "truth_px etc." -- print `df.columns[-8:]` on first read and pin the exact
spelling here. The pandas key is "table"; if that raises, list the keys with
`pandas.HDFStore(path).keys()`.)

Constituents are pT-ordered, highest first, and zero-padded: a constituent is real
iff E > 0. Label `is_signal_new`: 1 = top, 0 = QCD.

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

    CHECK BEFORE TRUSTING IT: a contiguous head-of-file slice is only valid if the file
    is shuffled. Assert the signal fraction is within [0.45, 0.55]. If test.h5 turns out
    to be ordered by label, the first 200k rows would be one class and every loss in the
    sweep would be meaningless -- fall back to a seed-pinned random 200k (contingency is
    pre-registered in preregistration.md section 2) and log the deviation.
    """
    raise NotImplementedError("build step 2")
