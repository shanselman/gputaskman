"""Read-only host check: two device samples, no process scan or workload launch."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gputaskman import drm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pci", required=True, help="full GPU PCI address, for example 0000:04:00.0")
    args = parser.parse_args()
    found = [card for card in drm.cards() if card.pdev.lower() == args.pci.lower() and card.driver == "xe"]
    if len(found) != 1:
        parser.error("no unique Xe GPU at that PCI address")
    card = found[0]
    endpoints = []
    for path in sorted(Path("/sys/class/intel_pmt").glob("telem[0-9]*"))[:256]:
        try:
            if Path(card.sysfs).resolve() not in path.resolve().parents:
                continue
            node = path / "telem"
            info = node.stat()
            endpoints.append({"path": str(path), "guid": (path / "guid").read_text().strip(),
                              "size": (path / "size").read_text().strip(), "mode": oct(info.st_mode & 0o777),
                              "uid": info.st_uid, "gid": info.st_gid, "readable": os.access(node, os.R_OK)})
        except OSError:
            continue
    samples = []
    try:
        samples.append(dataclasses.asdict(drm.read_device(card, 0)))
        time.sleep(0.5)
        samples.append(dataclasses.asdict(drm.read_device(card, 0)))
    finally:
        drm.DrmCounters([card]).close()
    json.dump({"pci": card.pdev, "driver": card.driver, "kernel": os.uname().release,
               "effective_uid": os.geteuid(), "pmt_endpoints": endpoints, "samples": samples,
               "scope": "Device counters only; no process scan, inference or workload was started."},
              sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
