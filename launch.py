from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    app_path = repo_root / "ui" / "app.py"

    if not app_path.exists():
        raise FileNotFoundError(f"Streamlit app not found: {app_path}")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
    ]

    # Streamlit wants to run as the main process.
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
