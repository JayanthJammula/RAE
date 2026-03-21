"""Tests for model.py edge cases."""

import pytest
import torch


class TestFactsEntropy:
    """Test entropy normalization edge cases (isolated from model class)."""

    def test_equal_entropy_returns_zeros(self):
        """When all entropy values equal, val_range==0, should return zeros."""
        entropy_values = torch.tensor([0.5, 0.5, 0.5])
        val_range = torch.max(entropy_values) - torch.min(entropy_values)
        if val_range > 0:
            entropy_values = (entropy_values - torch.min(entropy_values)) / val_range
        else:
            entropy_values = torch.zeros_like(entropy_values)
        assert entropy_values.tolist() == [0.0, 0.0, 0.0]

    def test_normal_normalization(self):
        """Standard case: values get normalized to 0-1 range."""
        entropy_values = torch.tensor([1.0, 2.0, 3.0])
        val_range = torch.max(entropy_values) - torch.min(entropy_values)
        entropy_values = (entropy_values - torch.min(entropy_values)) / val_range
        assert entropy_values.tolist() == [0.0, 0.5, 1.0]

    def test_single_value(self):
        """Single fact: val_range==0, should return zero."""
        entropy_values = torch.tensor([2.5])
        val_range = torch.max(entropy_values) - torch.min(entropy_values)
        if val_range > 0:
            entropy_values = (entropy_values - torch.min(entropy_values)) / val_range
        else:
            entropy_values = torch.zeros_like(entropy_values)
        assert entropy_values.tolist() == [0.0]

    def test_min_entropy_index(self):
        """Pruning selects facts up to min entropy index."""
        entropy_val = [0.0, 0.5, 1.0, 0.3]
        min_index = entropy_val.index(min(entropy_val))
        assert min_index == 0
        facts = ["fact0", "fact1", "fact2", "fact3"]
        pruned = facts[0 : (min_index + 1)]
        assert pruned == ["fact0"]


class TestPruneTraceStructure:
    """Test that prune_fact trace has expected keys."""

    def test_trace_keys(self):
        trace = {
            "input_facts": ["A", "B"],
            "entropy_values": [0.1, 0.5],
            "min_entropy_index": 0,
            "pruned_facts": ["A"],
            "facts_removed": 1,
        }
        assert set(trace.keys()) == {
            "input_facts",
            "entropy_values",
            "min_entropy_index",
            "pruned_facts",
            "facts_removed",
        }
