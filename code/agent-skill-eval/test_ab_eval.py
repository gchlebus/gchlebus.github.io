import unittest
from pathlib import Path

from ab_eval import evaluate_trial

ROOT = Path(__file__).parent


class ABEvalTest(unittest.TestCase):
    def test_with_skill_improves_receipt_trial(self):
        result = evaluate_trial(ROOT / "agent_trials" / "receipt_task.json", "deterministic", "unused")
        self.assertGreater(result.with_skill.score, result.baseline.score)
        self.assertGreater(result.delta, 0)

    def test_with_skill_improves_code_trial(self):
        result = evaluate_trial(ROOT / "agent_trials" / "code_task.json", "deterministic", "unused")
        self.assertGreater(result.with_skill.score, result.baseline.score)
        self.assertGreater(result.delta, 0)


if __name__ == "__main__":
    unittest.main()
