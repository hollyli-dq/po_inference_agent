#!/usr/bin/env python3
"""
Keep the machine awake until interrupted or for a fixed duration.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import time


def _run_caffeinate(duration: int | None) -> int:
    cmd = ["caffeinate", "-dimsu"]
    if duration is not None:
        cmd += ["-t", str(duration)]
    proc = subprocess.Popen(cmd)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return proc.wait()


def _run_windows(duration: int | None) -> int:
    try:
        import ctypes
    except Exception:
        print("Windows keep-awake requires ctypes (standard library).", file=sys.stderr)
        return 1

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    ctypes.windll.kernel32.SetThreadExecutionState(flags)
    try:
        if duration is None:
            while True:
                time.sleep(1)
        else:
            time.sleep(duration)
    except KeyboardInterrupt:
        return 0
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    return 0


def _run_linux(duration: int | None) -> int:
    if shutil.which("systemd-inhibit"):
        sleep_seconds = duration if duration is not None else 10**9
        cmd = [
            "systemd-inhibit",
            "--what=idle:sleep",
            "--why=keep_awake.py",
            "sleep",
            str(sleep_seconds),
        ]
        proc = subprocess.Popen(cmd)
        try:
            return proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            return proc.wait()

    print(
        "Linux keep-awake requires systemd-inhibit (systemd). "
        "Install it or run on macOS/Windows.",
        file=sys.stderr,
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Keep the machine awake.")
    parser.add_argument(
        "-t",
        "--seconds",
        type=int,
        default=None,
        help="Duration in seconds (omit to run until interrupted).",
    )
    args = parser.parse_args()

    system = platform.system()
    if system == "Darwin":
        return _run_caffeinate(args.seconds)
    if system == "Windows":
        return _run_windows(args.seconds)
    if system == "Linux":
        return _run_linux(args.seconds)

    print(f"Unsupported platform: {system}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
