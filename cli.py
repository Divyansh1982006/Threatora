#!/usr/bin/env python3
"""Top-level CLI launcher for Threatora."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# On Windows, if running under Python 3.14+ (which may encounter WDAC/AppLocker binary blocks on C-extensions),
# attempt transparent re-exec under Python 3.13 if available.
if sys.platform == "win32" and sys.version_info >= (3, 14) and not os.environ.get("THREATORA_NO_REEXEC"):
    py_launcher = shutil.which("py")
    if py_launcher:
        try:
            chk = subprocess.run(
                [py_launcher, "-3.13", "-c", "import sys; sys.exit(0)"],
                capture_output=True,
                timeout=5,
            )
            if chk.returncode == 0:
                os.environ["THREATORA_NO_REEXEC"] = "1"
                sys.exit(subprocess.call([py_launcher, "-3.13", str(Path(__file__).resolve())] + sys.argv[1:]))
        except Exception:
            pass

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.cli import main

if __name__ == "__main__":
    main()

