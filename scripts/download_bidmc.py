#!/usr/bin/env python3
"""Download all 53 BIDMC PhysioNet records to data/bidmc/.

Run once before using any scripts that load local signal files.
Annotations are fetched on-the-fly via wfdb pn_dir='bidmc'.

Usage:
    python scripts/download_bidmc.py
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DL_DIR = REPO_ROOT / "data" / "bidmc"


def main():
    try:
        import wfdb
    except ImportError:
        print("ERROR: wfdb not installed. Run: pip install wfdb")
        sys.exit(1)

    DL_DIR.mkdir(parents=True, exist_ok=True)
    records = [f"bidmc{i:02d}" for i in range(1, 54)]

    existing = {f.stem for f in DL_DIR.glob("*.hea")}
    to_download = [r for r in records if r not in existing]

    if not to_download:
        print(f"All {len(records)} records already present in {DL_DIR}")
        return

    print(f"Downloading {len(to_download)} / {len(records)} records to {DL_DIR} ...")
    for i, rec in enumerate(to_download, 1):
        print(f"  [{i:2d}/{len(to_download)}] {rec} ...", end=" ", flush=True)
        wfdb.dl_database("bidmc", dl_dir=str(DL_DIR), records=[rec])
        print("done")

    print(f"\nDone. {len(records)} records in {DL_DIR}")
    total_mb = sum(f.stat().st_size for f in DL_DIR.glob("*.dat")) / 1e6
    print(f"Total signal data: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
