#!/usr/bin/env python3
"""Fetch the Top Quark Tagging Reference Dataset.

    uv run python scripts/get_data.py --all           # train + val + test (~1.7 GB)
    uv run python scripts/get_data.py --files test.h5 # just one
    uv run python scripts/get_data.py --check         # verify md5 of what is on disk
    uv run python scripts/get_data.py --probe-pd4ml   # is the pd4ml mirror alive again?

Stdlib only, so it runs before (or without) the project environment. Downloads
resume via HTTP Range, and every file is md5-verified against the Zenodo record.

Source: Zenodo 10.5281/zenodo.2603256 (Kasieczka, Plehn, Thompson, Russel 2019).
See docs/deviations.md D-001 for why this is not the pd4ml package.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

ZENODO_RECORD = "https://zenodo.org/records/2603256"
ZENODO_FILES = {
    # name: (md5, size_bytes)
    "train.h5": ("45663819f47c13724f67eb0fd80bfa5c", 1_038_496_555),
    "val.h5": ("dca4b7248027618f041f9baa86d360fc", 347_378_076),
    "test.h5": ("13163479dee30a5fe546e4536cc3d04d", 347_849_376),
}
ZENODO_URL = "https://zenodo.org/api/records/2603256/files/{name}/content"

# The pd4ml alternative, kept here so the shim is one edit away if it comes back.
PD4ML_URL = (
    "https://syncandshare.desy.de/index.php/s/TWqT3E9j6q5yYSF/download/1_top_tagging_2M.npz"
)
PD4ML_MD5 = "708a8369d75ceff2229bd8c46b47afea"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNK = 1 << 20


def md5sum(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _fetch_once(name: str, part: Path, expected_size: int) -> int:
    """One attempt. Appends to `part`, resuming if it already has bytes.
    Returns the size of `part` afterwards. A short read is not an error here --
    the caller decides whether to retry."""
    have = part.stat().st_size if part.exists() else 0
    url = ZENODO_URL.format(name=name)
    req = urllib.request.Request(url, headers={"User-Agent": "jet-scaling-replication/0.1"})
    if have:
        req.add_header("Range", f"bytes={have}-")

    with urllib.request.urlopen(req) as resp:
        if have and resp.status != 206:
            # Server ignored Range: restart cleanly rather than append a second copy.
            part.unlink(missing_ok=True)
            have = 0
        mode = "ab" if have else "wb"
        with part.open(mode) as out:
            done = have
            try:
                while block := resp.read(CHUNK):
                    out.write(block)
                    done += len(block)
                    print(f"\r{name}: {done / 1e6:8.0f} / {expected_size / 1e6:.0f} MB"
                          f"  {100.0 * done / expected_size:5.1f}%", end="", flush=True)
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
                print(f"\n{name}: stream broke at {done / 1e6:.0f} MB ({exc})")
    print()
    return part.stat().st_size


def download(name: str, dest_dir: Path, expected_md5: str, expected_size: int,
             max_attempts: int = 6) -> Path:
    """Download with resume and retry. Verifies size BEFORE md5.

    A truncated transfer and a corrupt transfer need opposite responses -- resume the
    first, discard the second -- and md5 alone cannot tell them apart, because the
    checksum of a short file is wrong for the boring reason. So size is checked first.
    """
    dest = dest_dir / name
    part = dest.with_suffix(dest.suffix + ".part")

    if dest.exists():
        print(f"{name}: present, verifying md5 ...", flush=True)
        if md5sum(dest) == expected_md5:
            print(f"{name}: OK")
            return dest
        print(f"{name}: md5 MISMATCH on a complete file -- discarding and refetching")
        dest.unlink()

    for attempt in range(1, max_attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        if have > expected_size:
            print(f"{name}: partial is larger than expected -- discarding")
            part.unlink(missing_ok=True)
            have = 0
        if have:
            print(f"{name}: resuming at {have / 1e6:.0f} MB (attempt {attempt}/{max_attempts})")

        try:
            size = _fetch_once(name, part, expected_size)
        except urllib.error.URLError as exc:
            print(f"{name}: connection failed ({exc}); attempt {attempt}/{max_attempts}")
            continue

        if size < expected_size:
            print(f"{name}: TRUNCATED at {size / 1e6:.0f} / {expected_size / 1e6:.0f} MB"
                  f" -- attempt {attempt}/{max_attempts}, will resume")
            continue

        got = md5sum(part)
        if got != expected_md5:
            # Full size, wrong bytes: resuming cannot repair this. Start over.
            print(f"{name}: md5 {got} != {expected_md5} at full size -- corrupt, restarting")
            part.unlink(missing_ok=True)
            continue

        part.rename(dest)
        print(f"{name}: OK ({dest})")
        return dest

    raise RuntimeError(f"{name}: failed after {max_attempts} attempts; rerun to resume")


def check(dest_dir: Path) -> int:
    bad = 0
    for name, (expected, size) in ZENODO_FILES.items():
        path = dest_dir / name
        if not path.exists():
            print(f"{name}: MISSING")
            bad += 1
            continue
        actual_size = path.stat().st_size
        got = md5sum(path)
        ok = got == expected and actual_size == size
        print(f"{name}: {'OK' if ok else 'CORRUPT'}  size={actual_size}  md5={got}")
        bad += 0 if ok else 1
    return bad


def probe_pd4ml() -> int:
    """Is the pd4ml DESY mirror alive? It was 404 on 2026-09-05 (deviations D-001)."""
    req = urllib.request.Request(PD4ML_URL, headers={"Range": "bytes=0-1023"})
    try:
        with urllib.request.urlopen(req) as resp:
            print(f"pd4ml mirror: HTTP {resp.status} -- ALIVE. Reconsider D-001:")
            print("  pip install git+https://github.com/erum-data-idt/pd4ml")
            print("  from pd4ml import TopTagging; X, y = TopTagging.load('train', path='data')")
            print("  X[0] is (n, 200, 4) float32 ordered (E, px, py, pz); y is 0=QCD, 1=top.")
            print("  NOTE: pd4ml merges the original train+val into one 'train' split and")
            print("  exposes the original assignment in X[1] ('ttv': 0=train, 1=test, 2=val).")
            return 0
    except urllib.error.HTTPError as exc:
        print(f"pd4ml mirror: HTTP {exc.code} -- still dead. Staying on Zenodo (D-001).")
        return 1
    except urllib.error.URLError as exc:
        print(f"pd4ml mirror: unreachable ({exc.reason}). Staying on Zenodo (D-001).")
        return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="download train.h5, val.h5 and test.h5")
    ap.add_argument("--files", nargs="+", choices=sorted(ZENODO_FILES), help="download specific files")
    ap.add_argument("--check", action="store_true", help="md5-verify files already on disk")
    ap.add_argument("--probe-pd4ml", action="store_true", help="test whether the pd4ml mirror is back")
    ap.add_argument("--dir", type=Path, default=DATA_DIR)
    args = ap.parse_args()

    args.dir.mkdir(parents=True, exist_ok=True)

    if args.probe_pd4ml:
        return probe_pd4ml()
    if args.check:
        return check(args.dir)

    names = sorted(ZENODO_FILES) if args.all else (args.files or [])
    if not names:
        ap.print_help()
        return 2

    total = sum(ZENODO_FILES[n][1] for n in names)
    print(f"Zenodo record: {ZENODO_RECORD}")
    print(f"Downloading {len(names)} file(s), {total / 1e9:.2f} GB -> {args.dir}\n")
    failed = []
    for name in names:
        expected_md5, size = ZENODO_FILES[name]
        try:
            download(name, args.dir, expected_md5, size)
        except Exception as exc:  # keep going: one bad transfer should not block the rest
            print(f"{name}: FAILED -- {exc}")
            failed.append(name)

    if failed:
        print(f"\nIncomplete: {', '.join(failed)}. Rerun to resume from the .part files.")
        return 1
    print("\nDone. Cite BOTH the Zenodo DOI and arXiv:1902.09914.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
