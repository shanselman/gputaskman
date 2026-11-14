"""Publish fresh Intel device counters for unprivileged gputaskman; optionally record the full workload."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import dataclasses
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


class DiscardOutput:
    def write(self, value):
        return len(value)

    def flush(self):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pci', required=True, help='full PCI address, for example 0000:04:00.0')
    parser.add_argument('--interval', type=float, default=1.0)
    parser.add_argument('--duration', type=float, help='seconds to collect; default: until Ctrl+C')
    parser.add_argument('--record', action='store_true', help='also write full gputaskman JSON Lines to stdout')
    args = parser.parse_args()
    if not sys.platform.startswith('linux'):
        parser.error('the host reader requires native Linux')
    if not math.isfinite(args.interval) or args.interval < 0.1:
        parser.error('--interval must be finite and at least 0.1 seconds')
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error('--duration must be finite and greater than zero')
    if os.geteuid() != 0:
        parser.error('run this reader as administrator on the GPU host; gputaskman itself can remain unprivileged')
    if not any(Path('/dev/dri').glob('renderD*')):
        parser.error('GPU render nodes are unavailable in this environment; start this on the GPU host')
    # Prevent an older feed from becoming its own input, including startup priming.
    os.environ['GPUTASKMAN_DISABLE_HOST_FEED'] = '1'
    from gputaskman import drm
    from gputaskman.collector import enumerate_adapters, ordered_adapters
    from gputaskman.intel_host import DIRECTORY, FeedWriter, ReaderRunning, filename
    from gputaskman.record import run, source

    cards = [card for card in drm.cards() if card.pdev.lower() == args.pci.lower() and card.driver == 'xe']
    if len(cards) != 1:
        parser.error('no unique Xe GPU at that PCI address')
    card = cards[0]
    adapters = ordered_adapters(enumerate_adapters())
    indexes = [i for i, adapter in enumerate(adapters) if adapter.luid == card.key and adapter.vendor_id == card.vendor_id]
    if len(indexes) != 1:
        parser.error('cannot uniquely match that PCI device to a monitor adapter')
    try:
        writer = FeedWriter(card.pdev, card.vendor_id, card.device_id)
    except ReaderRunning:
        if not args.record:
            raise
        writer = None
        print('A host reader is already running; recording without replacing its live feed.', file=sys.stderr, flush=True)
    if writer is not None:
        print(f'Host reader for {card.pdev}: {DIRECTORY / filename(card.pdev)}', file=sys.stderr, flush=True)
        print('Normal gtm will use fresh samples automatically. Ctrl+C stops this reader.', file=sys.stderr, flush=True)
    count = 0
    try:
        def device_sample():
            device = drm.read_device(card, indexes[0])
            return {'taken': time.time(), 'device': dataclasses.asdict(device)}
        context = source(indexes[0], system=True) if args.record else nullcontext(device_sample)
        with context as sample:
            def publish():
                nonlocal count
                frame = sample()
                device = frame.get('device')
                if device is not None and (device.get('pci_bus') != card.pdev or device.get('source') != 'Intel sysfs'):
                    raise RuntimeError('selected GPU identity changed; restart the host reader')
                if writer is not None:
                    writer.publish(device, taken=frame['taken'])
                count += 1
                if count == 1 or count % 10 == 0:
                    d = device or {}
                    from gputaskman.nvml import format_metric
                    from gputaskman.report import memory_bandwidth_label
                    from gputaskman.telemetry import DeviceTelemetry

                    bandwidth = memory_bandwidth_label(DeviceTelemetry(**d)) if d else ''
                    used = d.get('mem_used')
                    memory = f'{used / 1024**3:.2f} GiB' if used is not None else 'n/a'
                    print(f"Sample {count}: GPU {format_metric(d.get('util_gpu'), '.1f', '%')} / "
                          f"VRAM {memory} / {bandwidth or 'memory bandwidth awaiting a reading'}", file=sys.stderr, flush=True)
                return frame
            run(publish, sys.stdout if args.record else DiscardOutput(), interval=args.interval, duration=args.duration)
    finally:
        try:
            drm.DrmCounters([card]).close()
        finally:
            try:
                if writer is not None:
                    writer.publish(None, state='stopped')
            finally:
                if writer is not None:
                    writer.close()
    print(f"{'Host reader stopped' if writer is not None else 'Recording finished'} after {count} samples.",
          file=sys.stderr, flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(str(error))
