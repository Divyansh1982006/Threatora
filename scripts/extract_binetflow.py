"""Pull only the .binetflow files out of the CTU-13 archive.

Run:
    python scripts/extract_binetflow.py

`tar --wildcards '*.binetflow'` should do this in one line, and on Linux it
does. Under Git-for-Windows' tar it sat at 33 seconds of CPU across 25 minutes
without writing a file, so this replaces it with something that reports what it
is doing.

The archive still has to be decompressed end to end - bzip2 is a stream format,
there is no index to seek with - but only the flow files are written. That
matters: extracting everything unpacks roughly 54 GB of per-scenario PCAPs that
this project never reads, since the packet-level path uses a separate capture.
"""

from __future__ import annotations

import argparse
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARCHIVE = ROOT / "data" / "raw" / "CTU-13-Dataset.tar.bz2"
DEFAULT_OUT = ROOT / "data" / "raw"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--suffix", default=".binetflow")
    args = parser.parse_args()

    if not args.archive.exists():
        raise SystemExit(f"Archive not found: {args.archive}")

    start = time.time()
    written = skipped = scanned = 0

    # Streaming mode ("r|bz2") rather than random access: it never builds the
    # full member index, so extraction starts immediately and memory stays flat.
    with tarfile.open(args.archive, "r|bz2") as tar:
        for member in tar:
            scanned += 1
            if scanned % 200 == 0:
                print(f"  [{time.strftime('%H:%M:%S')}] scanned {scanned} members, "
                      f"wrote {written}", flush=True)

            if not member.isfile() or not member.name.endswith(args.suffix):
                continue

            target = args.out / member.name
            if target.exists() and target.stat().st_size > 0:
                skipped += 1
                print(f"  have {member.name}", flush=True)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                continue
            with target.open("wb") as fh:
                while chunk := src.read(1 << 20):
                    fh.write(chunk)
            written += 1
            print(f"  wrote {member.name} "
                  f"({target.stat().st_size / 1e6:.0f} MB)", flush=True)

    print(f"\ndone in {time.time() - start:.0f}s: {written} written, "
          f"{skipped} already present, {scanned} members scanned")


if __name__ == "__main__":
    main()
