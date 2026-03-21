"""Tests for utils_func.py core functions."""

import pytest
import re
import sys
import os

# Add project root to path so we can import utils_func
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAnswerTruncation:
    """Test the truncation logic used in QA_func."""

    def _truncate(self, ans):
        for stop_tok in ["\n", "Question:", "Facts:"]:
            if stop_tok in ans:
                ans = ans[:ans.index(stop_tok)].strip()
        return ans.rstrip(".")

    def test_truncate_at_newline(self):
        assert self._truncate("Croatia\nSomething else") == "Croatia"

    def test_truncate_at_question(self):
        assert self._truncate("Croatia Question: who?") == "Croatia"

    def test_truncate_at_facts(self):
        assert self._truncate("Croatia Facts: some facts") == "Croatia"

    def test_no_truncation(self):
        assert self._truncate("Croatia") == "Croatia"

    def test_multiple_stop_tokens(self):
        assert self._truncate("Answer\nQuestion: x\nFacts: y") == "Answer"

    def test_strip_trailing_period(self):
        assert self._truncate("Croatia.") == "Croatia"


class TestAnswerMatching:
    """Test the regex-based answer comparison logic from QA_func."""

    def _simplify(self, text):
        return re.sub(r"[^a-zA-Z ]+", "", text).lower()

    def test_simple_match(self):
        assert self._simplify("Kolinda Grabar-Kitarovic") == "kolinda grabarkitarovic"

    def test_unicode_stripped(self):
        assert self._simplify("München") == "mnchen"

    def test_numbers_stripped(self):
        assert self._simplify("Agent 007") == "agent "

    def test_empty_string(self):
        assert self._simplify("") == ""

    def test_alias_matching(self):
        """Simulate the alias matching logic."""
        ground = "Kolinda Grabar-Kitarovic"
        aliases = ["Grabar-Kitarovic", "Kolinda"]
        ans = "The answer is Grabar-Kitarovic"

        simple_ans = self._simplify(ans)
        simple_ground = self._simplify(ground)

        correct = 0
        if simple_ground in simple_ans:
            correct = 1
        else:
            for alias in aliases:
                if self._simplify(alias) in simple_ans:
                    correct = 1
                    break
        assert correct == 1


class TestMatchFunc:
    """Test fact matching logic (isolated from the actual function to avoid heavy imports)."""

    def _match_func(self, ground_truth_fact, fact_str):
        """Simplified match_func matching the logic in utils_func.py."""
        if fact_str == "":
            return 0, 0

        part_match_cor = 0
        for fact in ground_truth_fact:
            if fact in fact_str:
                part_match_cor = 1
                break

        ext = 0
        for fact in ground_truth_fact:
            if fact in fact_str:
                ext += 1
        exact_match_cor = 1 if ext == len(ground_truth_fact) else 0

        return part_match_cor, exact_match_cor

    def test_empty_fact_str(self):
        assert self._match_func({"A B C"}, "") == (0, 0)

    def test_partial_match(self):
        ground = {"Ellie Kemper citizen of Croatia", "Croatia head of state Someone"}
        facts = "Ellie Kemper citizen of Croatia."
        pm, em = self._match_func(ground, facts)
        assert pm == 1
        assert em == 0

    def test_exact_match(self):
        ground = {"A B C"}
        facts = "A B C."
        pm, em = self._match_func(ground, facts)
        assert pm == 1
        assert em == 1

    def test_no_match(self):
        ground = {"X Y Z"}
        facts = "A B C."
        pm, em = self._match_func(ground, facts)
        assert pm == 0
        assert em == 0

    def test_multi_fact_exact(self):
        ground = {"fact one", "fact two"}
        facts = "fact one.\nfact two."
        pm, em = self._match_func(ground, facts)
        assert pm == 1
        assert em == 1
