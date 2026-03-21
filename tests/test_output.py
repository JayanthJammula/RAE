"""Tests for rae_output module."""

import json
import os
import pytest

from rae_output import generate_run_id, collect_run_metadata, RunResultCollector


class TestGenerateRunId:

    def test_format(self):
        rid = generate_run_id()
        parts = rid.split("_")
        assert len(parts) == 3
        assert len(parts[0]) == 8  # YYYYMMDD
        assert len(parts[1]) == 6  # HHMMSS
        assert len(parts[2]) == 7  # uuid chars

    def test_unique(self):
        ids = {generate_run_id() for _ in range(10)}
        assert len(ids) == 10


class TestRunResultCollector:

    def test_incremental_save(self, tmp_path, mock_args):
        metadata = {"run_id": None, "model_key": "gpt2", "args": vars(mock_args)}
        collector = RunResultCollector(metadata, output_dir=str(tmp_path))
        collector.add_case({"case_id": 1, "solved": True})
        collector.save_incremental()

        assert os.path.exists(collector._path)
        with open(collector._path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["status"] == "in_progress"
        assert len(data["cases"]) == 1

    def test_final_save(self, tmp_path, mock_args):
        metadata = {"run_id": None, "model_key": "gpt2", "args": vars(mock_args)}
        collector = RunResultCollector(metadata, output_dir=str(tmp_path))
        collector.add_case({"case_id": 1, "solved": True})

        metrics = {
            "total_ques": 1,
            "raw_exact_match_cor": 1,
            "raw_par_match_cor": 1,
            "prun_exact_match_cor": 1,
            "prun_par_match_cor": 1,
            "total_raw_cor": 1,
            "total_prun_cor": 1,
        }
        path = collector.save(metrics)

        assert os.path.exists(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["status"] == "completed"
        assert "aggregate_metrics" in data
        assert data["aggregate_metrics"]["prun_ans_acc"] == 1.0

    def test_custom_run_id(self, tmp_path, mock_args):
        metadata = {"run_id": None, "model_key": "gpt2", "args": vars(mock_args)}
        collector = RunResultCollector(metadata, output_dir=str(tmp_path), run_id="test_123")
        assert collector.run_id == "test_123"
        assert "test_123" in collector._path
