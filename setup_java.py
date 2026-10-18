"""
Install a Java 17 runtime into this project — no admin rights, no PATH edits.

WHY THIS EXISTS
---------------
PySpark needs a JVM, and Spark 3.5 needs Java 8, 11 or 17 specifically. The
usual route is an installer that wants administrator rights and edits system
environment variables, which on a managed machine is either impossible or
inadvisable.

This unpacks a Temurin 17 JDK into `.java/` inside the project instead.
`spark_features.py` looks there automatically, so nothing needs JAVA_HOME set
and nothing behaves differently in a new terminal — which matters, because
"works in one window and not the next" is its own category of wasted evening.

Run:  python setup_java.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent / ".java"
VERSION = "17"


def existing() -> Path | None:
    if not TARGET.exists():
        return None
    for exe in sorted(TARGET.glob("*/bin/java*")):
        if exe.suffix in ("", ".exe"):
            return exe.parent.parent
    return None


def main() -> None:
    already = existing()
    if already:
        print(f"Java already present at {already}")
        print("Nothing to do — spark_features.py will find it.")
        return

    try:
        import jdk
    except ImportError:
        print("Installing the downloader first:\n")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "install-jdk"],
                       check=True)
        import jdk

    TARGET.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Temurin {VERSION} (~180 MB) into {TARGET} ...")
    try:
        path = jdk.install(VERSION, jre=False, path=str(TARGET))
    except Exception as e:
        print(f"\nDownload failed: {e}\n")
        print("Fall back to doing it by hand — no admin rights needed either:")
        print("  1. https://adoptium.net/temurin/releases/?version=17")
        print("     choose Windows x64, package type **.zip** (not .msi)")
        print(f"  2. extract it so you end up with {TARGET}\\jdk-17...\\bin\\java.exe")
        print("  3. re-run this script to confirm")
        raise SystemExit(1)

    print(f"\nInstalled to {path}")
    found = existing()
    if not found:
        raise SystemExit("Extracted, but no bin/java found — check the layout above.")

    print(f"JAVA_HOME will resolve to {found}")
    print("\nNothing else to set. Verify with:\n"
          "    python spark_features.py --check")
    print("\nThis lives inside the project, so `.gitignore` should exclude "
          "`.java/` — it is 300 MB of binaries.")


if __name__ == "__main__":
    main()
