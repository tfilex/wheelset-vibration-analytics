"""Backward-compatible wrapper for the packaged offline diagnostics script."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from console_diagnostics.run import main  # noqa: E402


if __name__ == "__main__":
    main()
