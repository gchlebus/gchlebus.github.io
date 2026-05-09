import unittest
from pathlib import Path

from benchmark import score_skill

ROOT = Path(__file__).parent


class BenchmarkTest(unittest.TestCase):
    def test_good_skill_scores_high(self):
        score = score_skill(ROOT / "fixtures" / "good_skill" / "SKILL.md")
        self.assertGreaterEqual(score.score, 90)
        self.assertEqual(score.grade, "A")

    def test_problematic_skill_scores_low(self):
        score = score_skill(ROOT / "fixtures" / "problematic_skill" / "SKILL.md")
        self.assertLess(score.score, 40)
        self.assertEqual(score.grade, "F")


if __name__ == "__main__":
    unittest.main()
