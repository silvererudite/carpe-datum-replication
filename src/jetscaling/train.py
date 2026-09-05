"""One config in -> one immutable results JSON out.

STATUS: contract only. Implement in build step 3 (docs/orientation.md 6.3).

    uv run python -m jetscaling.train --config <run.yaml> --out results/runs/

Rules this module enforces (from CLAUDE.md):
  * every hyperparameter comes from the config -- no literals in the training loop;
  * the run JSON contains every key in grid.yaml:run_json_required_keys, validated
    BEFORE the file is written;
  * a run JSON is never overwritten and never edited. If a run must be repeated, it
    gets a new run_id and both files stay on disk.

The LR schedule is the subtle part
----------------------------------
Cosine decay with T_max = *this run's* total step count, warmup = warmup_frac of it.
Kaplan et al. (2001.08361) used a schedule longer than many of their runs, which left
those runs stopped mid-decay at an inflated loss; Hoffmann et al. (2203.15556, sec. 3
and app. C) showed that fixing this changes the fitted exponents materially. Our D
axis spans 24x, so every D column is a different run length: if the schedule does not
follow the run length, we reproduce the exact bug that made Kaplan's data exponent
wrong. Record the resolved (peak_lr, warmup_steps, total_steps) in the run JSON so a
reader can verify this was done.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunResult:
    run_id: str
    params_exact: int
    N_name: str
    D: int
    seed: int
    final_val_loss: float  # nats, mean BCE on the fixed eval slice
    val_auc: float
    wall_time_s: float
    lr_schedule: dict
    git_commit: str
    config_hash: str
    device: str
    tokens_seen: int


def resolve_schedule(total_steps: int, recipe: dict) -> dict:
    """-> {peak_lr, min_lr, warmup_steps, total_steps, kind}. Pure function; unit-test it."""
    raise NotImplementedError("build step 3")


def evaluate(model, eval_data, batch_size: int, device: str) -> tuple[float, float]:
    """-> (mean BCE in nats, ROC AUC) on the fixed eval slice.

    Nats, not bits: torch's BCEWithLogitsLoss is already natural log. If you ever
    report bits, divide by ln 2 and say so -- an unlabelled unit change looks exactly
    like a shifted L_inf in the fit.
    """
    raise NotImplementedError("build step 3")


def train_one(config: dict) -> RunResult:
    """The whole single-run pipeline. Deterministic given (config, seed)."""
    raise NotImplementedError("build step 3")


def write_run_json(result: RunResult, out_dir: str) -> str:
    """Validate against grid.yaml:run_json_required_keys, then write atomically.

    Refuse to overwrite an existing run_id: raise, do not clobber. The runs directory
    is the evidence trail for the whole replication.
    """
    raise NotImplementedError("build step 3")


def main() -> None:
    raise NotImplementedError("build step 3")


if __name__ == "__main__":
    main()
