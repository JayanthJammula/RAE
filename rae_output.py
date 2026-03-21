"""Structured run output: metadata collection, per-case result tracking, JSON persistence."""

import json
import os
import subprocess
import sys
import time
import uuid


def generate_run_id():
    """Return a unique run identifier: YYYYMMDD_HHMMSS_<7-char uuid>."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:7]
    return f"{ts}_{short}"


def collect_run_metadata(args, model_id):
    """Capture reproducibility metadata for the current run."""
    git_hash = "unknown"
    git_dirty = False
    try:
        git_hash = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        ).stdout.strip() or "unknown"
        git_dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5
        ).stdout.strip())
    except Exception:
        pass

    torch_version = "unknown"
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        pass

    return {
        "run_id": None,  # set by RunResultCollector
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_hash": git_hash,
        "git_dirty": git_dirty,
        "model_key": args.model,
        "model_hf_id": model_id,
        "dataset": args.dataset,
        "args": {k: v for k, v in vars(args).items()},
        "python_version": sys.version,
        "torch_version": torch_version,
        "device": args.device,
    }


class RunResultCollector:
    """Collects per-case results and writes them to a JSON file."""

    def __init__(self, metadata, output_dir="results", run_id=None):
        self.run_id = run_id or generate_run_id()
        self.metadata = metadata
        self.metadata["run_id"] = self.run_id
        self.cases = []
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._path = os.path.join(output_dir, f"run_{self.run_id}.json")

    def add_case(self, case_result):
        """Append a single case result dict."""
        self.cases.append(case_result)

    def save_incremental(self):
        """Write current state to disk after every case (crash-safe)."""
        payload = {
            "metadata": self.metadata,
            "status": "in_progress",
            "cases": self.cases,
        }
        self._write(payload)

    def save(self, metrics):
        """Write final output with aggregate metrics. Returns file path."""
        total = metrics.get("total_ques", 1)
        payload = {
            "metadata": self.metadata,
            "status": "completed",
            "aggregate_metrics": {
                "total_cases": total,
                "raw_exact_match_acc": metrics["raw_exact_match_cor"] / total if total else 0,
                "raw_par_match_acc": metrics["raw_par_match_cor"] / total if total else 0,
                "prun_exact_match_acc": metrics["prun_exact_match_cor"] / total if total else 0,
                "prun_par_match_acc": metrics["prun_par_match_cor"] / total if total else 0,
                "raw_ans_acc": metrics["total_raw_cor"] / total if total else 0,
                "prun_ans_acc": metrics["total_prun_cor"] / total if total else 0,
            },
            "cases": self.cases,
        }
        self._write(payload)
        return self._path

    def _write(self, payload):
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
