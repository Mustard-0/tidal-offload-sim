import unittest
from pathlib import Path
from types import SimpleNamespace

from followup import command_for, verify_convergence, verify_pilot


class FollowupTest(unittest.TestCase):
    def test_formal_main_needs_selection(self):
        args = SimpleNamespace(stage="formal-main", updates=300, device="cuda",
                               convergence_run=None, max_seconds=None, seed=2026, resume=False)
        with self.assertRaises(ValueError):
            command_for(args, Path("unused"))

    def test_formal_main_uses_explicit_full_tide(self):
        args = SimpleNamespace(stage="formal-main", updates=300, device="cuda",
                               convergence_run=Path("selection-run"), max_seconds=3600,
                               seed=2026, resume=False)
        command = command_for(args, Path("new-run"))
        self.assertIn("--formal-eval", command)
        self.assertIn("--selection", command)
        self.assertIn("--max-seconds", command)

    def test_incomplete_pilot_and_convergence_are_rejected(self):
        with self.assertRaises(FileNotFoundError):
            verify_pilot(Path("missing-pilot"))
        with self.assertRaises(FileNotFoundError):
            verify_convergence(Path("missing-convergence"))


if __name__ == "__main__":
    unittest.main()
