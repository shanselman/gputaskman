"""gputaskman - an htop for what's *inside* your GPU."""

from __future__ import annotations

from .i18n import cli_language, tr

import argparse
import math
import sys
import time


def _language_option(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--lang", choices=("ru", "en"), help=tr("interface language (default: ru; or GPUTASKMAN_LANG)"))


@cli_language
def diff_main(argv: list[str]) -> None:
    import json

    ap = argparse.ArgumentParser(prog="gputaskman diff",
                                 description=tr("Compare two --json snapshots, for example before and after loading a model."))
    ap.add_argument("before", help=tr("earlier --json snapshot"))
    ap.add_argument("after", help=tr("later --json snapshot"))
    ap.add_argument("--json", action="store_true", help=tr("print the comparison as JSON"))
    _language_option(ap)
    args = ap.parse_args(argv)

    from .diff import diff, load, to_text

    try:
        result = diff(load(args.before), load(args.after))
    except (OSError, ValueError) as error:
        ap.error(tr(str(error)))
    print(json.dumps(result, indent=2) if args.json else to_text(result))


def _check_platform() -> None:
    if sys.platform not in ("win32", "darwin") and not sys.platform.startswith("linux"):
        sys.exit(tr("gputaskman supports Windows (WDDM), Linux/WSL (NVML, or amdgpu/Intel DRM) and macOS (Apple Silicon) only."))


def _check_gpu(ap: argparse.ArgumentParser, index: int | None) -> None:
    if index:
        from .collector import enumerate_adapters, select_adapter

        try:
            select_adapter(enumerate_adapters(), index)
        except ValueError as error:
            ap.error(tr(str(error)))


@cli_language
def record_main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="gputaskman record",
                                 description=tr("Append one --json snapshot per interval to a JSON Lines file, "
                                             "for example to keep what a long run did to the GPU."))
    ap.add_argument("-o", "--output", required=True, help=tr("file to append to, one --json snapshot per line"))
    ap.add_argument("--gpu", type=int, default=0, help=tr("GPU to record (default 0; gputaskman --list-gpus shows the indexes)"))
    ap.add_argument("-i", "--interval", type=float, default=1.0, help=tr("seconds between samples (default 1)"))
    ap.add_argument("--duration", type=float, help=tr("stop after this many seconds (default: until Ctrl+C)"))
    ap.add_argument("--no-host", action="store_true",
                    help=tr("WSL: don't read Windows host GPU counters through interop (WSL VM total, Windows apps)"))
    ap.add_argument("--no-system", action="store_true", help=tr("don't sample system RAM/CPU: no system section in the records"))
    _language_option(ap)
    args = ap.parse_args(argv)
    if args.gpu < 0:
        ap.error(tr("--gpu must be a non-negative index"))
    if not math.isfinite(args.interval) or args.interval <= 0:
        ap.error(tr("--interval must be a finite number greater than zero"))
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        ap.error(tr("--duration must be a finite number greater than zero"))
    _check_platform()
    _check_gpu(ap, args.gpu)

    from .record import append_to, run, source

    try:
        out = append_to(args.output)
    except OSError as error:
        ap.error(tr(f"cannot write {args.output}: {error}"))
    count = 0
    try:
        with out, source(args.gpu, host=not args.no_host, system=not args.no_system) as sample:
            span = tr(f" for {args.duration:g} s") if args.duration is not None else ""
            print(tr(f"Recording GPU {args.gpu} every {args.interval:g} s{span} to {args.output}; "
                  "press Ctrl+C to stop."), file=sys.stderr)
            count = run(sample, out, interval=args.interval, duration=args.duration)
    except KeyboardInterrupt:
        pass  # Ctrl+C while starting up: nothing was recorded yet
    except OSError as error:
        ap.exit(1, tr(f"{ap.prog}: recording to {args.output} failed: {error}\n"))
    print(tr(f"Recorded {count} sample(s) to {args.output}"), file=sys.stderr)


@cli_language
def main() -> None:
    argv = sys.argv[1:]
    command = 0
    while command < len(argv):
        if argv[command] == "--lang" and command + 1 < len(argv):
            command += 2
        elif argv[command].startswith("--lang="):
            command += 1
        else:
            break
    if command < len(argv) and argv[command] in ("diff", "record"):
        handler = diff_main if argv[command] == "diff" else record_main
        return handler(argv[:command] + argv[command + 1:])
    ap = argparse.ArgumentParser(prog="gputaskman", description=tr("What's inside your GPU: per-process VRAM, engines, and LLM weights/KV-cache breakdown."),
                                 epilog=tr("gputaskman diff BEFORE.json AFTER.json compares two --json snapshots; "
                                        "gputaskman record -o FILE appends one --json snapshot per interval to FILE."))
    ap.add_argument("--once", action="store_true", help=tr("print one snapshot and exit (no TUI)"))
    ap.add_argument("--json", action="store_true", help=tr("print one JSON snapshot and exit (implies --once)"))
    ap.add_argument("--gpu", type=int, help=tr("show only this GPU (TUI default: all; one-shot default: 0)"))
    ap.add_argument("--list-gpus", action="store_true", help=tr("list selectable GPU adapters and exit"))
    ap.add_argument("--no-host", action="store_true",
                    help=tr("WSL: don't read Windows host GPU counters through interop (WSL VM total, Windows apps)"))
    ap.add_argument("--no-system", action="store_true",
                    help=tr("don't sample system RAM/CPU: no TUI panel or RAM/CPU%% columns, no system section in --once/--json"))
    ap.add_argument("-i", "--interval", type=float, default=1.0, help=tr("refresh seconds (default 1)"))
    _language_option(ap)
    ap.add_argument("--no-tray", action="store_true", help=tr("disable the Windows taskbar panel, notification icon and compact window title"))
    ap.add_argument("--no-taskbar", action="store_true", help=tr("hide the always-visible taskbar readings (Windows)"))
    args = ap.parse_args(argv)
    if args.gpu is not None and args.gpu < 0:
        ap.error(tr("--gpu must be a non-negative index"))
    if not math.isfinite(args.interval) or args.interval <= 0:
        ap.error(tr("--interval must be a finite number greater than zero"))

    _check_platform()

    if args.list_gpus:
        from .collector import enumerate_adapters, ordered_adapters
        from .tcc import TccAdapter, tcc_discovery_status

        for i, adapter in enumerate(ordered_adapters(enumerate_adapters())):
            if isinstance(adapter, TccAdapter):  # its LUID is made up from the PCI identity, so show the address
                bus, device, function = adapter.pci_address
                pci = f"{adapter.pci_domain:08x}:" if adapter.pci_domain else ""
                print(f"{i}: {adapter.name} ({adapter.vendor}, {tr('TCC compute-only')}, "
                      f"PCI {pci}{bus:02x}:{device:02x}.{function:x})")
            else:
                print(f"{i}: {adapter.name} ({adapter.vendor}, LUID 0x{adapter.luid & 0xffffffffffffffff:016x})")
            if adapter.identity_status:
                print(f"   {tr(adapter.identity_status)}")
        status = tcc_discovery_status()
        if status:
            print(tr("TCC discovery: {0}").format(tr(status)), file=sys.stderr)
        return

    _check_gpu(ap, args.gpu)

    if args.once or args.json:
        from .collector import Collector
        from .host import HostSampler
        from .report import dump

        try:
            c = Collector(args.gpu or 0, scan_budget=None, host=not args.no_host)  # one-shot: identify everything now
        except ValueError as error:
            ap.error(tr(str(error)))
        try:
            if c.host is not None:
                c.host.wait(15.0)  # WSL: the first Windows counter sample takes several seconds
            if hasattr(c.counters, "warm_up"):
                c.counters.warm_up(15.0)  # macOS: per-process GPU memory comes from a background footprint pass
            sampler, system, system_error = None, None, None
            if not args.no_system:
                try:
                    sampler = HostSampler()
                    sampler.sample()  # per-process CPU% is a rate too; this primes it
                except Exception as error:  # system RAM/CPU is extra: its failure must not cost the GPU report
                    sampler, system_error = None, f"{type(error).__name__}: {error}"
            time.sleep(1.0)  # engine utilization is a rate; needs two samples
            snapshot = c.sample()
            if sampler:
                try:
                    system = sampler.sample()
                except Exception as error:
                    system_error = f"{type(error).__name__}: {error}"
            print(dump(snapshot, args.json, system, system_error))
        finally:
            c.close()
        return

    from .app import GpuTaskManager

    GpuTaskManager(gpu_index=args.gpu, interval=args.interval, host=not args.no_host,
                   system=not args.no_system, tray=not args.no_tray, taskbar=not args.no_taskbar).run()
