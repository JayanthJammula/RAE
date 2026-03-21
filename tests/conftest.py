"""Shared fixtures for RAE pytest suite."""

import pytest
from types import SimpleNamespace


@pytest.fixture
def sample_dataset_entry():
    """A minimal valid dataset entry for testing."""
    return {
        "case_id": 1,
        "questions": [
            "Who is the head of state of the country Ellie Kemper is a citizen of?",
            "What country is Ellie Kemper from and who leads it?",
            "Ellie Kemper's nationality's head of state is?",
        ],
        "answer": "Joe Biden",
        "new_answer": "Kolinda Grabar-Kitarovic",
        "answer_alias": ["Biden"],
        "new_answer_alias": ["Grabar-Kitarovic"],
        "orig": {
            "new_triples": [
                ["Q1234", "P27", "Q5678"],
                ["Q5678", "P35", "Q9999"],
            ],
            "new_triples_labeled": [
                ["Ellie Kemper", "P27", "Croatia"],
                ["Croatia", "P35", "Kolinda Grabar-Kitarovic"],
            ],
            "edit_triples": [
                ["Q1234", "P27", "Q5678"],
            ],
        },
    }


@pytest.fixture
def mock_args():
    """A mock args namespace matching main.py's argparse output."""
    return SimpleNamespace(
        model="gpt2",
        dataset="slices/MQuAKE-CF_slice_10",
        relation_path="data/relation.json",
        NatureL=True,
        template=True,
        template_number=5,
        entropy_template_number=5,
        starting_line=0,
        mode="beam",
        correctConflict=True,
        seed=42,
        device="cpu",
        conf_threshold=0.7,
        max_retrieval_rounds=3,
        loss="prob_div",
        beam_width=5,
        num_beams=1,
        max_new_tokens=50,
        temp=1.0,
        load_8bit=False,
    )
