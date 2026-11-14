"""Device-only handoff from an explicitly started privileged Intel host reader.

No elevation, sockets, service installation or workload control. Readers accept
only a recent sample from this boot and exact PCI device. Public handoff files
contain device counters only; process records stay in the user's private trace.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import stat
import time

DIRECTORY = Path('/run/gputaskman')
MAX_BYTES = 65536
MAX_AGE = 6.0
XE_TOTAL = 'Xe query (driver-reported usable VRAM)'
XE_USED = 'Xe query (privileged driver estimate)'
CLIENT_USED = 'visible DRM clients (estimated)'
PCI_TOTAL = 'PCI window (estimated)'
MEMORY = ('mem_read_bps', 'mem_write_bps', 'mem_bandwidth_max_bps', 'mem_bandwidth_pct')
PCIE = ('pcie_rx_kbs', 'pcie_tx_kbs')
LINK = ('pcie_gen', 'pcie_width')
NUMERIC = {'mem_total', 'mem_used', 'util_gpu', 'video', 'temp', 'vram_temp', 'power_w', 'power_limit_w',
           'fan', 'fan_rpm', 'sm_clock', 'sm_clock_max', 'gfx_clock', 'mem_clock', 'video_clock', *MEMORY, *PCIE, *LINK}
INTEGER = {'mem_total', 'mem_used', 'mem_bandwidth_max_bps', 'fan_rpm', 'pcie_gen', 'pcie_width'}
SCALARS = NUMERIC - {'mem_total', 'mem_used', 'util_gpu', 'video', *MEMORY, *PCIE, *LINK}
_FLAGS = os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NOFOLLOW', 0)


class ReaderRunning(RuntimeError):
    pass


def filename(pci: str) -> str:
    if not re.fullmatch(r'[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]', pci):
        raise ValueError('invalid full PCI address')
    return 'intel-' + pci.lower().replace(':', '_') + '.json'


def boot_id() -> str:
    try:
        return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except OSError:
        return ''


def _root_uid() -> int:
    # Host root may map to the overflow uid inside a user namespace. The owner
    # of the mounted /run is the trusted root identity in that filesystem view.
    try:
        return Path('/run').stat().st_uid
    except OSError:
        return 0


def _secure_directory(path: Path, uid: int) -> int:
    fd = os.open(path, _FLAGS | getattr(os, 'O_DIRECTORY', 0))
    info = os.fstat(fd)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != uid or info.st_mode & 0o022:
        os.close(fd)
        raise ValueError('host reader directory has an unexpected owner or writable permissions')
    return fd


class HostDeviceFeed:
    def __init__(self, pci: str, vendor_id: int, device_id: int, *, directory=DIRECTORY,
                 trusted_uid=None, boot=None, clock=time.time):
        self.pci, self.vendor_id, self.device_id = pci.lower(), vendor_id, device_id
        self.directory, self.name = Path(directory), filename(pci)
        self.uid = _root_uid() if trusted_uid is None else trusted_uid
        self.boot, self.clock = boot_id() if boot is None else boot, clock
        self.seen = False

    def read(self) -> tuple[dict | None, str]:
        directory_fd = fd = None
        try:
            directory_fd = _secure_directory(self.directory, self.uid)
            fd = os.open(self.name, _FLAGS | getattr(os, 'O_NONBLOCK', 0), dir_fd=directory_fd)
            self.seen = True
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != self.uid or info.st_mode & 0o022:
                raise ValueError('host sample has an unexpected owner, type or writable permissions')
            if not 0 < info.st_size <= MAX_BYTES:
                raise ValueError('host sample size is invalid')
            raw = os.read(fd, MAX_BYTES + 1)
            if len(raw) != info.st_size:
                raise ValueError('host sample changed during reading')
            body = json.loads(raw)
            if not isinstance(body, dict) or type(body.get('schema')) is not int or body['schema'] != 1:
                raise ValueError('unsupported host sample schema')
            if (body.get('pci_bus') != self.pci or body.get('vendor_id') != self.vendor_id
                    or body.get('device_id') != self.device_id):
                raise ValueError('host sample does not match this PCI device')
            if not self.boot or body.get('boot_id') != self.boot:
                raise ValueError('host sample belongs to a different or unverifiable boot')
            stamp = body.get('taken')
            if (type(stamp) not in (int, float) or abs(stamp) >= 2**64 or not math.isfinite(stamp)):
                raise ValueError('invalid host sample timestamp')
            age = self.clock() - stamp
            if age < -1 or age > MAX_AGE:
                raise ValueError('host sample is stale or future-dated; restart the host reader')
            if body.get('state') != 'running':
                raise ValueError('host reader stopped; restart it to read protected counters')
            device = body.get('device')
            if not isinstance(device, dict) or device.get('source') != 'Intel sysfs' or device.get('pci_bus') != self.pci:
                raise ValueError('host reader has no matching Intel device telemetry')
            for key in NUMERIC:
                value = device.get(key)
                if value is None:
                    continue
                if type(value) not in (int, float) or abs(value) >= 2**64 or not math.isfinite(value):
                    raise ValueError(f'invalid host value for {key}')
                if key in INTEGER and (type(value) is not int or value >= 2**64):
                    raise ValueError(f'invalid host integer for {key}')
                if value < (-273.15 if key in ('temp', 'vram_temp') else 0):
                    raise ValueError(f'negative host value for {key}')
                if key in ('util_gpu', 'video', 'fan') and value > 100:
                    raise ValueError(f'invalid host percentage for {key}')
            if device.get('pcie_gen') is not None and device['pcie_gen'] not in range(1, 8):
                raise ValueError('invalid host PCIe generation')
            if device.get('pcie_width') is not None and device['pcie_width'] not in (1, 2, 4, 8, 12, 16, 32):
                raise ValueError('invalid host PCIe width')
            if (device.get('mem_total') is not None and device.get('mem_used') is not None
                    and device['mem_used'] > device['mem_total']):
                raise ValueError('host used memory exceeds its reported total')
            for key in ('metric_sources', 'metric_status'):
                value = device.get(key, {})
                if (not isinstance(value, dict) or len(value) > 100
                        or any(not isinstance(k, str) or not isinstance(v, str) or len(v) > 2048
                               for k, v in value.items())):
                    raise ValueError(f'invalid host {key}')
            for key in ('util_kind', 'video_kind', 'power_kind'):
                value = device.get(key, '')
                if not isinstance(value, str) or len(value) > 256:
                    raise ValueError(f'invalid host {key}')
            throttle = device.get('throttle')
            if throttle is not None and (not isinstance(throttle, list) or len(throttle) > 64
                    or any(not isinstance(flag, str) or len(flag) > 128 for flag in throttle)):
                raise ValueError('invalid host throttle state')
            return device, ''
        except FileNotFoundError:
            return None, 'host reader output disappeared' if self.seen else ''
        except (OSError, ValueError, TypeError, UnicodeError, OverflowError, RecursionError) as error:
            return None, str(error)
        finally:
            if fd is not None:
                os.close(fd)
            if directory_fd is not None:
                os.close(directory_fd)

    def apply(self, card, device) -> None:
        host, reason = self.read()
        if host is None:
            if reason:
                device.metric_status['host_feed'] = reason
            return
        sources = host.get('metric_sources', {})
        def put(key, value):
            setattr(device, key, value)
            if value is not None:
                device.metric_sources[key] = sources.get(key, 'Intel device counter') + ' via host reader'
                device.metric_status.pop(key, None)
            else:
                device.metric_sources.pop(key, None)

        for field in SCALARS:
            if getattr(device, field) is None and host.get(field) is not None:
                put(field, host[field])
                if field == 'power_w':
                    device.power_kind = host.get('power_kind', '')
        for field, kind in (('util_gpu', 'util_kind'), ('video', 'video_kind')):
            if host.get(field) is not None and (getattr(device, field) is None or
                    (getattr(device, kind) and not host.get(kind))):
                put(field, host[field])
                setattr(device, kind, host.get(kind, ''))
        for family in (MEMORY, PCIE, LINK):
            if (not all(getattr(device, field) is not None for field in family[:2])
                    and all(host.get(field) is not None for field in family[:2])):
                for field in family:
                    put(field, host.get(field))
        if (device.mem_read_bps is None and device.mem_write_bps is None
                and device.mem_bandwidth_max_bps is None and host.get('mem_bandwidth_max_bps') is not None):
            put('mem_bandwidth_max_bps', host['mem_bandwidth_max_bps'])
        if device.throttle is None and host.get('throttle') is not None:
            put('throttle', host['throttle'])

        host_total = host.get('mem_total')
        host_query = host_total is not None and host_total > 0 and sources.get('mem_total') == XE_TOTAL
        host_used = host.get('mem_used')
        use_host_used = (host_query and host_used is not None and sources.get('mem_used') == XE_USED
                         and device.metric_sources.get('mem_used') != XE_USED)
        use_host_total = host_query and (device.mem_total is None or device.metric_sources.get('mem_total') != XE_TOTAL)
        if use_host_used or use_host_total:
            put('mem_total', host_total)
            card.integrated = False
            if use_host_used:
                put('mem_used', host_used)
                device.memory_note = ('Host Xe query: driver-reported usable VRAM; used VRAM is the '
                                      "privileged driver's device-wide estimate from the host reader.")
            else:
                device.memory_note = ('Host Xe query: driver-reported usable VRAM; used VRAM remains the visible '
                                      "clients' resident sum, which can overlap and omit inaccessible clients.")
            device.metric_status.pop('memory_query', None)
        for field, reason in host.get('metric_status', {}).items():
            if field in NUMERIC and getattr(device, field) is None:
                device.metric_status[field] = 'Host reader: ' + reason
        device.metric_status.pop('host_feed', None)
        if device.mem_read_bps is not None:
            device.metric_status['util_mem'] = 'busy-time percentage unavailable; see measured memory bandwidth'

    def close(self):
        self.seen = False


class FeedWriter:
    """Atomic device-only publication into an owner-controlled directory."""
    def __init__(self, pci, vendor_id, device_id, *, directory=DIRECTORY):
        import fcntl

        self.pci, self.vendor_id, self.device_id = pci.lower(), vendor_id, device_id
        self.name, self.boot = filename(pci), boot_id()
        if not self.boot:
            raise RuntimeError('cannot verify the host boot identity')
        directory = Path(directory)
        directory.mkdir(mode=0o755, exist_ok=True)
        self.directory_fd = _secure_directory(directory, os.geteuid())
        os.fchmod(self.directory_fd, 0o755)  # public device counters must remain readable under a private umask
        self.lock_fd = None
        try:
            self.lock_fd = os.open(self.name + '.lock', os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0),
                                   0o600, dir_fd=self.directory_fd)
            info = os.fstat(self.lock_fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise ValueError('host reader lock has unsafe ownership or permissions')
            try:
                fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ReaderRunning('a host reader is already publishing this PCI device') from error
        except BaseException:
            self.close()
            raise

    def publish(self, device: dict | None, *, taken=None, state='running'):
        body = {'schema': 1, 'pci_bus': self.pci, 'vendor_id': self.vendor_id, 'device_id': self.device_id,
                'boot_id': self.boot, 'taken': time.time() if taken is None else taken, 'state': state,
                'device': device}
        raw = (json.dumps(body, allow_nan=False, separators=(',', ':')) + '\n').encode()
        if len(raw) > MAX_BYTES:
            raise ValueError('host device sample exceeds the handoff limit')
        temp_name = f'.{self.name}.{os.getpid()}.{time.monotonic_ns()}.tmp'
        fd = None
        try:
            fd = os.open(temp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0),
                         0o600, dir_fd=self.directory_fd)
            with os.fdopen(fd, 'wb') as stream:
                fd = None
                stream.write(raw)
                stream.flush()
                os.fchmod(stream.fileno(), 0o644)
            os.replace(temp_name, self.name, src_dir_fd=self.directory_fd, dst_dir_fd=self.directory_fd)
        finally:
            if fd is not None:
                os.close(fd)
            try:
                os.unlink(temp_name, dir_fd=self.directory_fd)
            except FileNotFoundError:
                pass

    def close(self):
        if self.lock_fd is not None:
            os.close(self.lock_fd)
            self.lock_fd = None
        if self.directory_fd is not None:
            os.close(self.directory_fd)
            self.directory_fd = None
