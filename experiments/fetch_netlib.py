#!/usr/bin/env python3
"""Fetch a small, fixed Netlib selection; never execute downloaded content.

The source revision is pinned. Subsequent downloads must match the committed
manifest byte-for-byte. Data are public LP benchmarks, not personal records.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
REVISION = "f1cc423067407d55d579c9c35fb01edf860dbc24"
BASE = f"https://raw.githubusercontent.com/coin-or-tools/Data-Netlib/{REVISION}/"
CASES = {
    "adlittle": (225494.96316, "Established Netlib LP supplied by Stanford SOL."),
    "afiro": (-464.75314286, "Small established Netlib LP supplied by Stanford SOL."),
    "blend": (-30.812149846, "Oil-refinery model variant documented by Netlib."),
    "scfxm1": (18416.759028, "Staircase LP from the Ho-Loute benchmark collection."),
    "stocfor1": (-41131.976219, "Deterministic member of Netlib's seven-period forestry family."),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "netlib")
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    path = args.data_dir / "manifest.json"
    recorded = json.loads(path.read_text()) if path.exists() else None
    manifest = {"upstream_repository": "https://github.com/coin-or-tools/Data-Netlib",
                "revision": REVISION, "reference": "https://www.netlib.org/lp/data/readme",
                "downloaded_utc": datetime.now(timezone.utc).isoformat(), "datasets": []}
    for name, (optimum, description) in CASES.items():
        url = BASE + name + ".mps.gz"
        compressed = urlopen(url, timeout=30).read()
        content = gzip.decompress(compressed)
        entry = {"name": name, "url": url, "sha256_gzip": sha(compressed),
                 "sha256_mps": sha(content), "bytes_mps": len(content),
                 "reference_objective": optimum, "description": description}
        if recorded:
            previous = next(d for d in recorded["datasets"] if d["name"] == name)
            if entry != previous:
                raise ValueError(f"Source differs from committed manifest: {name}")
        destination = args.data_dir / f"{name}.mps"
        if destination.exists() and destination.read_bytes() != content:
            raise ValueError(f"Refusing to overwrite differing data: {destination.name}")
        if not destination.exists():
            destination.write_bytes(content)
        manifest["datasets"].append(entry)
        print(f"Verified {name}: {len(content)} bytes, sha256={sha(content)}", flush=True)
    license_content = urlopen(BASE + "LICENSE", timeout=30).read()
    license_path = args.data_dir / "UPSTREAM_LICENSE.txt"
    if license_path.exists() and license_path.read_bytes() != license_content:
        raise ValueError("Existing upstream license differs")
    license_path.write_bytes(license_content)
    if recorded is None:
        path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
