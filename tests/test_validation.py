"""Tests for rae_validation module."""

import pytest
import os
import pickle
import tempfile
from types import SimpleNamespace

from rae_validation import (
    validate_dataset_entry,
    validate_dataset,
    safe_load_pickle,
    validate_args,
    DataValidationError,
)


class TestValidateDatasetEntry:

    def test_valid_entry_passes(self, sample_dataset_entry):
        warnings = validate_dataset_entry(sample_dataset_entry, 0)
        assert warnings == []

    def test_missing_case_id_raises(self):
        entry = {"questions": ["Q1"], "new_answer": "A", "new_answer_alias": [], "orig": {"new_triples": [], "new_triples_labeled": []}}
        with pytest.raises(DataValidationError, match="missing required key 'case_id'"):
            validate_dataset_entry(entry, 0)

    def test_missing_questions_raises(self):
        entry = {"case_id": 1, "new_answer": "A", "new_answer_alias": [], "orig": {"new_triples": [], "new_triples_labeled": []}}
        with pytest.raises(DataValidationError, match="missing required key 'questions'"):
            validate_dataset_entry(entry, 0)

    def test_empty_questions_raises(self):
        entry = {"case_id": 1, "questions": [], "new_answer": "A", "new_answer_alias": [], "orig": {"new_triples": [], "new_triples_labeled": []}}
        with pytest.raises(DataValidationError, match="must be a non-empty list"):
            validate_dataset_entry(entry, 0)

    def test_missing_orig_key_raises(self):
        entry = {"case_id": 1, "questions": ["Q1"], "new_answer": "A", "new_answer_alias": []}
        with pytest.raises(DataValidationError, match="missing required key 'orig'"):
            validate_dataset_entry(entry, 0)

    def test_missing_orig_subkey_raises(self):
        entry = {"case_id": 1, "questions": ["Q1"], "new_answer": "A", "new_answer_alias": [], "orig": {"new_triples": []}}
        with pytest.raises(DataValidationError, match="'orig' missing required key 'new_triples_labeled'"):
            validate_dataset_entry(entry, 0)


class TestValidateDataset:

    def test_non_list_raises(self):
        with pytest.raises(DataValidationError, match="must be a JSON list"):
            validate_dataset({"key": "value"}, "test.json")

    def test_empty_raises(self):
        with pytest.raises(DataValidationError, match="is empty"):
            validate_dataset([], "test.json")

    def test_valid_dataset_passes(self, sample_dataset_entry):
        validate_dataset([sample_dataset_entry], "test.json")


class TestSafeLoadPickle:

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            safe_load_pickle("/nonexistent/path.pkl")

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "empty.pkl"
        p.write_bytes(b"")
        with pytest.raises(DataValidationError, match="empty"):
            safe_load_pickle(str(p))

    def test_corrupt_file_raises(self, tmp_path):
        p = tmp_path / "corrupt.pkl"
        p.write_bytes(b"not a pickle")
        with pytest.raises(DataValidationError, match="Failed to unpickle"):
            safe_load_pickle(str(p))

    def test_non_dict_raises(self, tmp_path):
        p = tmp_path / "list.pkl"
        with open(str(p), "wb") as f:
            pickle.dump([1, 2, 3], f)
        with pytest.raises(DataValidationError, match="Expected dict"):
            safe_load_pickle(str(p))

    def test_valid_pickle_loads(self, tmp_path):
        p = tmp_path / "valid.pkl"
        data = {"Q123": "some\tfact\tdata"}
        with open(str(p), "wb") as f:
            pickle.dump(data, f)
        result = safe_load_pickle(str(p))
        assert result == data


class TestValidateArgs:

    def test_valid_args_no_warnings(self, mock_args):
        warnings = validate_args(mock_args)
        assert warnings == []

    def test_negative_beam_width(self, mock_args):
        mock_args.beam_width = -1
        warnings = validate_args(mock_args)
        assert any("beam_width" in w for w in warnings)

    def test_conf_threshold_out_of_range(self, mock_args):
        mock_args.conf_threshold = 1.5
        warnings = validate_args(mock_args)
        assert any("conf_threshold" in w for w in warnings)

    def test_zero_temp(self, mock_args):
        mock_args.temp = 0
        warnings = validate_args(mock_args)
        assert any("temp" in w for w in warnings)

    def test_negative_seed(self, mock_args):
        mock_args.seed = -1
        warnings = validate_args(mock_args)
        assert any("seed" in w for w in warnings)
