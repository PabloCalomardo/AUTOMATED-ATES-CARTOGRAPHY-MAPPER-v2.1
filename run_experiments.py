#!/usr/bin/env python3
"""Run commands listed directly in this file.

Edit COMMANDS below and paste the exact main.py invocations you want to run.
Each command gets its own log file under outputs/experiments/<run_id>/ and the
script writes a CSV manifest with the full batch plan and exit codes.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable


# Put your commands here, one labeled run per entry.
# Use the label for the result folder name.
# Example:
# COMMANDS = [
# 	("bow_no_limit_oh_025", "python main.py --dem inputs/DEM_BOW_SUMMIT.tif --forest inputs/FOREST_BOW_SUMMIT.tif --overhead-cellcount-weight 0.25"),
# 	("connaught_with_limit_oh_max", "python main.py --dem inputs/DEM_ATES_CONNAUGHT.tif --forest inputs/FOREST_ATES_CONNAUGHT.tif --study-area inputs/DELIMITACIO_CONNAUGHT.shp --overhead-cellcount-weight 2"),
# ]
COMMANDS: list[tuple[str, str]] = [
	# BOW SUMMIT
	("bow_no_limit_oh_0", "python main.py --dem inputs/DEM_BOW_SUMMIT.tif --forest inputs/FOREST_BOW_SUMMIT.tif --overhead-cellcount-weight 0"),
	("bow_no_limit_oh_0p5", "python main.py --dem inputs/DEM_BOW_SUMMIT.tif --forest inputs/FOREST_BOW_SUMMIT.tif --overhead-cellcount-weight 0.5"),
	("bow_no_limit_oh_1", "python main.py --dem inputs/DEM_BOW_SUMMIT.tif --forest inputs/FOREST_BOW_SUMMIT.tif --overhead-cellcount-weight 1"),
	("bow_no_limit_oh_max", "python main.py --dem inputs/DEM_BOW_SUMMIT.tif --forest inputs/FOREST_BOW_SUMMIT.tif --overhead-cellcount-weight 2"),
	("bow_limit_oh_0", "python main.py --dem inputs/DEM_BOW_SUMMIT.tif --forest inputs/FOREST_BOW_SUMMIT.tif --study-area inputs/DELIMITACIONS/DELIMITACIO_BOW.shp --overhead-cellcount-weight 0"),
	("bow_limit_oh_0p5", "python main.py --dem inputs/DEM_BOW_SUMMIT.tif --forest inputs/FOREST_BOW_SUMMIT.tif --study-area inputs/DELIMITACIONS/DELIMITACIO_BOW.shp --overhead-cellcount-weight 0.5"),
	("bow_limit_oh_1", "python main.py --dem inputs/DEM_BOW_SUMMIT.tif --forest inputs/FOREST_BOW_SUMMIT.tif --study-area inputs/DELIMITACIONS/DELIMITACIO_BOW.shp --overhead-cellcount-weight 1"),
	("bow_limit_oh_max", "python main.py --dem inputs/DEM_BOW_SUMMIT.tif --forest inputs/FOREST_BOW_SUMMIT.tif --study-area inputs/DELIMITACIONS/DELIMITACIO_BOW.shp --overhead-cellcount-weight 2"),
	# ATES CONNAUGHT
	("connaught_no_limit_oh_0", "python main.py --dem inputs/DEM_ATES_CONNAUGHT.tif --forest inputs/FOREST_ATES_CONNAUGHT.tif --overhead-cellcount-weight 0"),
	("connaught_no_limit_oh_0p5", "python main.py --dem inputs/DEM_ATES_CONNAUGHT.tif --forest inputs/FOREST_ATES_CONNAUGHT.tif --overhead-cellcount-weight 0.5"),
	("connaught_no_limit_oh_1", "python main.py --dem inputs/DEM_ATES_CONNAUGHT.tif --forest inputs/FOREST_ATES_CONNAUGHT.tif --overhead-cellcount-weight 1"),
	("connaught_no_limit_oh_max", "python main.py --dem inputs/DEM_ATES_CONNAUGHT.tif --forest inputs/FOREST_ATES_CONNAUGHT.tif --overhead-cellcount-weight 2"),
	("connaught_limit_oh_0", "python main.py --dem inputs/DEM_ATES_CONNAUGHT.tif --forest inputs/FOREST_ATES_CONNAUGHT.tif --study-area inputs/DELIMITACIONS/DELIMITACIO_CONNAUGHT.shp --overhead-cellcount-weight 0"),
	("connaught_limit_oh_0p5", "python main.py --dem inputs/DEM_ATES_CONNAUGHT.tif --forest inputs/FOREST_ATES_CONNAUGHT.tif --study-area inputs/DELIMITACIONS/DELIMITACIO_CONNAUGHT.shp --overhead-cellcount-weight 0.5"),
	("connaught_limit_oh_1", "python main.py --dem inputs/DEM_ATES_CONNAUGHT.tif --forest inputs/FOREST_ATES_CONNAUGHT.tif --study-area inputs/DELIMITACIONS/DELIMITACIO_CONNAUGHT.shp --overhead-cellcount-weight 1"),
	("connaught_limit_oh_max", "python main.py --dem inputs/DEM_ATES_CONNAUGHT.tif --forest inputs/FOREST_ATES_CONNAUGHT.tif --study-area inputs/DELIMITACIONS/DELIMITACIO_CONNAUGHT.shp --overhead-cellcount-weight 2"),
]


def _repo_root() -> Path:
	return Path(__file__).resolve().parent


def _venv_python(root: Path) -> Path:
	python_exe = root / ".venv312" / "Scripts" / "python.exe"
	if not python_exe.exists():
		raise FileNotFoundError(f"Virtual environment interpreter not found: {python_exe}")
	return python_exe


def _subprocess_env() -> dict[str, str]:
	env = os.environ.copy()
	osgeo_bin = Path(r"C:\OSGeo4W\bin")
	if osgeo_bin.exists():
		env["PATH"] = str(osgeo_bin) + os.pathsep + env.get("PATH", "")
	return env


def _normalize_command_text(command_text: str) -> str:
	return " ".join(command_text.strip().split())


def _split_command(command_text: str) -> list[str]:
	try:
		parts = shlex.split(command_text, posix=False)
	except ValueError as exc:
		raise ValueError(f"Could not parse command: {command_text}") from exc
	if not parts:
		raise ValueError("Empty command is not allowed")
	return parts


def _has_outputs_dir(parts: Iterable[str]) -> bool:
	parts_list = list(parts)
	for index, part in enumerate(parts_list):
		if part == "--outputs-dir":
			return True
		if part.startswith("--outputs-dir="):
			return True
		if part in {"-o", "--output-dir"}:
			return True
		if part in {"-o", "--output-dir"} and index + 1 < len(parts_list):
			return True
	return False


def _sanitize_label(text: str, fallback: str) -> str:
	filtered = []
	for char in text.lower():
		if char.isalnum():
			filtered.append(char)
		elif char in {"-", "_"}:
			filtered.append(char)
		else:
			filtered.append("_")
	label = "".join(filtered).strip("_")
	return label or fallback


def _build_run_plan(commands: list[tuple[str, str]], output_root: Path, interpreter: Path) -> list[dict[str, object]]:
	runs: list[dict[str, object]] = []
	for index, (label_text, command_text) in enumerate(commands, start=1):
		normalized = _normalize_command_text(command_text)
		parts = _split_command(normalized)
		parts = _force_venv_interpreter(parts, interpreter)
		fallback_label = f"run_{index:02d}"
		label = _sanitize_label(label_text, fallback_label)
		run_id = f"{index:02d}_{label}"
		run_dir = output_root / run_id
		log_path = run_dir / "run.log"
		command_parts = parts

		if not _has_outputs_dir(parts):
			command_parts = command_parts + ["--outputs-dir", str(run_dir / "outputs")]

		runs.append(
			{
				"index": index,
				"run_id": run_id,
				"label": label,
				"label_text": label_text,
				"command_text": normalized,
				"effective_command_text": " ".join(command_parts),
				"command_parts": command_parts,
				"run_dir": run_dir,
				"log_path": log_path,
			}
		)
	return runs


def _force_venv_interpreter(command_parts: list[str], interpreter: Path) -> list[str]:
	if not command_parts:
		raise ValueError("Empty command is not allowed")
	first_token = Path(command_parts[0]).name.lower()
	if first_token in {"python", "python.exe", "py", "py.exe"}:
		return [str(interpreter), *command_parts[1:]]
	return command_parts


def _write_manifest(manifest_path: Path, runs: list[dict[str, object]]) -> None:
	manifest_path.parent.mkdir(parents=True, exist_ok=True)
	with manifest_path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(
			handle,
			fieldnames=["index", "run_id", "label", "label_text", "command", "run_dir", "log_path", "exit_code"],
		)
		writer.writeheader()
		for run in runs:
			writer.writerow(
				{
					"index": run["index"],
					"run_id": run["run_id"],
					"label": run["label"],
					"label_text": run["label_text"],
					"command": run["effective_command_text"],
					"run_dir": str(run["run_dir"]),
					"log_path": str(run["log_path"]),
					"exit_code": "",
				}
			)


def main() -> int:
	root = _repo_root()
	interpreter = _venv_python(root)
	output_root = root / "outputs" / "experiments"
	manifest_path = output_root / "manifest.csv"

	parser = argparse.ArgumentParser(description="Run commands defined in the COMMANDS list.")
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Only write the manifest and print the planned runs; do not execute anything.",
	)
	args = parser.parse_args()

	commands = [(label.strip(), command.strip()) for label, command in COMMANDS if label.strip() and command.strip()]
	if not commands:
		parser.error("COMMANDS is empty. Edit run_16_experiments.py and add labeled commands there.")

	runs = _build_run_plan(commands, output_root, interpreter)
	_write_manifest(manifest_path, runs)

	print(f"Manifest written to: {manifest_path}")
	print(f"Total commands: {len(runs)}")

	if args.dry_run:
		for run in runs:
			print(f"[{run['index']}] {run['run_id']}")
			print(f"    {run['effective_command_text']}")
		return 0

	for run in runs:
		index = int(run["index"])
		run_id = str(run["run_id"])
		run_dir = Path(run["run_dir"])
		log_path = Path(run["log_path"])
		command_parts = list(run["command_parts"])

		run_dir.mkdir(parents=True, exist_ok=True)
		print(f"[{index}/{len(runs)}] Running {run_id}")
		print(f"          log: {log_path}")
		env = _subprocess_env()
		with log_path.open("w", encoding="utf-8") as log_file:
			result = subprocess.run(command_parts, stdout=log_file, stderr=subprocess.STDOUT, check=False, env=env)

		if result.returncode != 0:
			print(f"[{index}/{len(runs)}] Failed with exit code {result.returncode}")
			return result.returncode

		print(f"[{index}/{len(runs)}] Done")

	print("All commands finished successfully.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())