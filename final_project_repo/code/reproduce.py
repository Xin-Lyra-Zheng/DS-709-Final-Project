"""Run the analyses from the frozen inputs."""

from pathlib import Path
import os
import subprocess
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]

with ZipFile(ROOT / "data/inputs.zip") as archive:
    archive.extractall(ROOT)

env = dict(os.environ, OPENBLAS_NUM_THREADS="1")
for script in ("audit.py", "run_retention_eg.py", "run_retention_sensitivity.py"):
    subprocess.run([sys.executable, ROOT / "code" / script], check=True, env=env)
