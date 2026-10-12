"""Run the analyses from the frozen inputs."""

from pathlib import Path
import argparse
import os
import subprocess
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--from-raw", action="store_true")
args = parser.parse_args()

with ZipFile(ROOT / "data/inputs.zip") as archive:
    archive.extractall(ROOT)

env = dict(os.environ, OPENBLAS_NUM_THREADS="1")
if args.from_raw:
    data = ROOT / "data/refit"
    env["FAIRREPAIR_INPUT"] = str(data)
    env["FAIRREPAIR_OUTPUT"] = str(ROOT / "results/refit")
    subprocess.run([sys.executable, ROOT / "code/train_models.py", "--check"],
                   check=True, env=env)
    subprocess.run([sys.executable, ROOT / "code/run_baselines.py",
                    "--input-dir", str(data), "--output-dir", str(data / "baselines"),
                    "--check"], check=True, env=env)
for script in ("audit.py", "run_retention_eg.py", "run_retention_sensitivity.py"):
    subprocess.run([sys.executable, ROOT / "code" / script], check=True, env=env)
