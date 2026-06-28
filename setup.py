#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a project-local virtual environment and install RecSys dependencies.

Usage:
    python setup.py

All packages are installed into ./.venv via that environment's pip, leaving the
system Python and conda environments untouched.
"""

import argparse
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"

TORCH_PACKAGES = [
    "torch>=2.2",
]

PROJECT_PACKAGES = [
    "numpy>=1.24",
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "python-multipart>=0.0.9",
    "matplotlib>=3.7",
]


def venv_python():
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def run(command):
    print("+ " + " ".join(str(part) for part in command), flush=True)
    subprocess.check_call([str(part) for part in command], cwd=PROJECT_ROOT)


def create_venv(recreate=False):
    python_path = venv_python()
    if recreate and VENV_DIR.exists():
        shutil.rmtree(VENV_DIR)

    if not python_path.exists():
        print("Creating virtual environment: %s" % VENV_DIR, flush=True)
        builder = venv.EnvBuilder(with_pip=True, clear=False)
        builder.create(VENV_DIR)
    else:
        print("Using existing virtual environment: %s" % VENV_DIR, flush=True)

    return python_path


def install_packages(python_path, torch_index_url=None):
    run([python_path, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([python_path, "-m", "pip", "install", *PROJECT_PACKAGES])

    torch_command = [python_path, "-m", "pip", "install", *TORCH_PACKAGES]
    if torch_index_url:
        torch_command.extend(["--index-url", torch_index_url])
    run(torch_command)


def print_next_steps(python_path):
    if os.name == "nt":
        activate = r".venv\Scripts\activate"
        uvicorn = r".venv\Scripts\uvicorn"
    else:
        activate = "source .venv/bin/activate"
        uvicorn = ".venv/bin/uvicorn"

    print("\nSetup complete.", flush=True)
    print("Activate environment:", flush=True)
    print("  %s" % activate, flush=True)
    print("Run a model:", flush=True)
    print("  %s main.py --model dcn --config RecSys/config/dcn.yaml" % python_path, flush=True)
    print("Run metric server:", flush=True)
    print("  %s metric_server.main:app --reload --host 0.0.0.0 --port 8000" % uvicorn, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Bootstrap RecSys dependencies into ./.venv")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Remove the existing .venv before creating a fresh one.",
    )
    parser.add_argument(
        "--torch-index-url",
        default=os.environ.get("TORCH_INDEX_URL"),
        help="Optional PyTorch wheel index URL, e.g. https://download.pytorch.org/whl/cu121",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    python_path = create_venv(recreate=args.recreate)
    install_packages(python_path, torch_index_url=args.torch_index_url)
    print_next_steps(python_path)


if __name__ == "__main__":
    main()
