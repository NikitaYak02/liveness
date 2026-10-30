import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "adapters"))

from dp360 import windows  # noqa: E402


class WindowsTest(unittest.TestCase):
    def test_full_length_and_coverage(self):
        for n in (1, 7, 8, 16, 29, 61):
            for chunk in (8,):
                wins = windows(n, chunk)
                self.assertTrue(all(len(w) == chunk for w in wins))
                self.assertEqual(sorted({i for w in wins for i in w}), list(range(n)))

    def test_neighbours_overlap(self):
        wins = windows(20, 8)
        for a, b in zip(wins, wins[1:]):
            self.assertEqual(a[-1:], b[:1])


if __name__ == "__main__":
    unittest.main()
