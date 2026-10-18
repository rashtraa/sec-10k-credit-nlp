"""
Prove where everything will be written before starting a long job.

Run this whenever you are unsure which environment is live. It reports the
interpreter, every cache location, and free disk space, and it FAILS if
anything that will hold real data points at C:.

    python src\\env_check.py
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

TARGET_DRIVE = os.environ.get("EDGAR_TARGET_DRIVE", "X").upper().rstrip(":")


def drive_of(p) -> str:
    try:
        return Path(p).resolve().drive.upper().rstrip(":") or "?"
    except Exception:
        return "?"


def free_gb(p) -> float:
    try:
        probe = Path(p).resolve()
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        return shutil.disk_usage(probe).free / 1e9
    except Exception:
        return float("nan")


def row(label, value, *, must_be_target=False):
    d = drive_of(value)
    if must_be_target:
        good = (d == TARGET_DRIVE)
        mark = "ok  " if good else "FAIL"
    else:
        good = True
        mark = "    "
    print(f"  {mark} {label:<22} {value}")
    return good


def main() -> int:
    print(f"\nTarget drive: {TARGET_DRIVE}:\n")

    print("interpreter")
    venv = os.environ.get("VIRTUAL_ENV", "(none — no venv active)")
    ok = True
    ok &= row("python.exe", sys.executable, must_be_target=True)
    ok &= row("VIRTUAL_ENV", venv, must_be_target=True)
    row("version", f"{sys.version_info.major}.{sys.version_info.minor}."
                   f"{sys.version_info.micro}")

    print("\nproject output")
    try:
        import config
        ok &= row("data root", config.DATA, must_be_target=True)
        ok &= row("filings", config.FILINGS, must_be_target=True)
    except Exception as e:
        print(f"  FAIL could not import config: {e}")
        ok = False

    print("\ncaches that default to C:")
    for label, var, needed_at in [
        ("pip wheels", "PIP_CACHE_DIR", "install time"),
        ("huggingface", "HF_HOME", "stage 3 (~1 GB)"),
        ("torch", "TORCH_HOME", "stage 3"),
        ("temp", "TMP", "everywhere"),
    ]:
        val = os.environ.get(var)
        if not val:
            print(f"  FAIL {label:<22} NOT SET — will write to C:  "
                  f"[needed at {needed_at}]")
            ok = False
        else:
            ok &= row(label, val, must_be_target=True)

    print("\ndisk space")
    for d in ("C:\\", f"{TARGET_DRIVE}:\\"):
        g = free_gb(d)
        note = ""
        if d.startswith(TARGET_DRIVE) and g == g and g < 5:
            note = "  <-- tight; project needs ~3 GB"
        print(f"       {d:<22} {g:,.1f} GB free{note}")

    print()
    if ok:
        print("PASS — nothing writes to C:. Safe to start a long job.\n")
        return 0
    print("FAIL — see the lines marked FAIL above.")
    print("Fix: open a NEW PowerShell window, then:")
    print("       cd X:\\Projects\\edgar-credit-signal")
    print("       .\\activate.ps1\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
