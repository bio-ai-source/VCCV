from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "src"
VENDOR_ROOT = PACKAGE_ROOT / "vendor"
for path in (SOURCE_ROOT, VENDOR_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
