"""训练入口：加载 default.yaml，并以更友好的方式调用本地 Ultralytics 源码。"""

from __future__ import annotations

import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
ENV_FILE = WORKSPACE_ROOT / ".env"
ENV_PLACEHOLDERS = {"your_api_key_here", "<your_ultralytics_api_key>", "replace_me"}


def load_env_file(env_file: Path = ENV_FILE) -> None:
	"""Load simple KEY=VALUE pairs from a local .env file without overriding the current shell."""
	if not env_file.exists():
		return

	for raw_line in env_file.read_text(encoding="utf-8").splitlines():
		line = raw_line.strip()
		if not line or line.startswith("#") or "=" not in line:
			continue

		key, value = line.split("=", 1)
		key = key.strip()
		value = value.strip().strip('"').strip("'")
		if not key or not value or key in os.environ or value.lower() in ENV_PLACEHOLDERS:
			continue

		os.environ[key] = value


load_env_file()

if str(SCRIPT_DIR) not in sys.path:
	sys.path.insert(0, str(SCRIPT_DIR))

from yolo_experiment import run_from_cli


if __name__ == "__main__":
	raise SystemExit(run_from_cli("train"))

