"""Launch the (N, D) grid described by configs/grid.yaml.

STATUS: contract only. Implement in build step 4 (docs/orientation.md 6.4).

    uv run python -m jetscaling.sweep --dry-run          # print the 22 runs, launch nothing
    uv run python -m jetscaling.sweep --only tiny:50000  # one cell, for debugging
    uv run python -m jetscaling.sweep                    # the whole grid, resumable

Resumability is not a convenience here: a sweep that cannot resume tempts you to
delete a half-finished run and start over, and deleting run JSONs breaks the evidence
trail. Skip cells whose run JSON already exists; never re-run silently.

Run the four cheapest cells first (tiny/small x 50k/150k). If something is wrong with
the pipeline -- a mask bug, a leaky eval slice, a schedule that does not follow the run
length -- it is visible in ten minutes of those four rather than after the 1.2M-jet run.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunSpec:
    run_id: str  # e.g. "tiny_d50000_s0" -- deterministic, used as the JSON filename
    N_name: str
    D: int
    seed: int
    is_seed_repeat: bool


def iter_runs(grid_cfg: dict) -> list[RunSpec]:
    """The full 16 grid cells + 6 seed repeats, in cheapest-first order.

    Assert len == 22 (or whatever the config implies) so a config typo that silently
    drops a row cannot pass unnoticed.
    """
    raise NotImplementedError("build step 4")


def estimate_flops(params: int, d: int, tokens_per_jet: int) -> float:
    """6 * N * (D * tokens_per_jet), the usual fwd+bwd estimate. Used for ordering,
    for the x-axis of the isoFLOP plots, and for a sanity check against wall time:
    if measured throughput differs wildly between two cells, something other than the
    model is dominating (usually the dataloader)."""
    return 6.0 * params * d * tokens_per_jet


def main() -> None:
    raise NotImplementedError("build step 4")


if __name__ == "__main__":
    main()
