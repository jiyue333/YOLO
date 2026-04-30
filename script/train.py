"""训练入口：加载 default.yaml，并以更友好的方式调用本地 Ultralytics 源码。"""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
	sys.path.insert(0, str(SCRIPT_DIR))

from yolo_experiment import run_from_cli


if __name__ == "__main__":
	raise SystemExit(run_from_cli("train"))

