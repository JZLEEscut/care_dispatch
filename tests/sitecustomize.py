"""Ensure `import src...` works when a test file is run directly.

Example supported command on Windows:
    python C:/path/to/care_dispatch/tests/test_action_mask.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
