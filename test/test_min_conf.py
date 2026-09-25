"""The score floor that `min_conf` promised and nothing applied.

The model always ranks, so `select_answer` handed back a best option even for
a query the option list does not answer. `min_conf` has been in this plugin's
default config since the first commit and nothing read it.

The ranker is stubbed here. These tests are about the floor, not about the
model: a test that downloads ms-marco to assert an inequality would be slow,
would need the network, and would still not pin the numbers, because they move
with the model.
"""
import unittest
from unittest.mock import MagicMock, patch

from ovos_flashrank_solver import FlashRankMultipleChoiceSolver

OPTIONS = ["the capital of portugal is lisbon",
           "water boils at 100 degrees",
           "the sun is a star"]


def _solver(min_conf, scores):
    """A solver whose ranker returns `scores` against OPTIONS, in order."""
    with patch("ovos_flashrank_solver.Ranker") as ranker:
        ranker.return_value.rerank.return_value = [
            {"text": t, "score": s} for t, s in zip(OPTIONS, scores)
        ]
        s = FlashRankMultipleChoiceSolver({"min_conf": min_conf,
                                           "n_answer": 1,
                                           "model": "stub"})
    return s


class TestMinConf(unittest.TestCase):

    def test_no_floor_keeps_every_option(self):
        """The default. min_conf None is falsy, so nothing is dropped and the
        behaviour is what every release so far has had."""
        s = _solver(None, [0.64, 0.02, 0.001])
        self.assertEqual(len(s.rerank("q", list(OPTIONS))), 3)
        self.assertEqual(s.select_answer("q", list(OPTIONS)), OPTIONS[0])

    def test_floor_drops_the_options_below_it(self):
        s = _solver(0.5, [0.64, 0.02, 0.001])
        ranked = s.rerank("q", list(OPTIONS))
        self.assertEqual([r[1] for r in ranked], [OPTIONS[0]])

    def test_select_answer_is_none_when_nothing_clears_the_floor(self):
        """The case the floor exists for: the model ranked, and every score is
        below what the deployment will act on."""
        s = _solver(0.9, [0.64, 0.02, 0.001])
        self.assertIsNone(s.select_answer("q", list(OPTIONS)))

    def test_select_answer_without_a_floor_never_returns_none(self):
        """Control: the same near-zero vector, no floor, still answers."""
        s = _solver(None, [0.0012, 0.0008, 0.0003])
        self.assertEqual(s.select_answer("q", list(OPTIONS)), OPTIONS[0])

    def test_floor_applies_to_the_index_form_too(self):
        """return_index goes through the same filter, so a caller asking for an
        index cannot bypass the floor."""
        s = _solver(0.9, [0.64, 0.02, 0.001])
        self.assertEqual(s.rerank("q", list(OPTIONS), return_index=True), [])
        self.assertIsNone(s.select_answer("q", list(OPTIONS), return_index=True))

    def test_a_clearing_score_still_answers(self):
        s = _solver(0.9, [0.997, 0.004, 0.001])
        self.assertEqual(s.select_answer("q", list(OPTIONS)), OPTIONS[0])
