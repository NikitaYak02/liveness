import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_pose_synth_tree import discover_jobs, has_images, job_done  # noqa: E402


class DiscoverTest(unittest.TestCase):
    def test_nested_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "a" / "seq1").mkdir(parents=True)
            (root / "a" / "seq2").mkdir()
            (root / "b").mkdir()
            (root / "a" / "seq1" / "0001.jpg").write_bytes(b"x")
            (root / "a" / "seq1" / "0002.png").write_bytes(b"x")
            (root / "a" / "seq2" / "frame.jpeg").write_bytes(b"x")
            (root / "b" / "readme.txt").write_text("no images")
            jobs = discover_jobs(root)
            self.assertEqual([j.relative_to(root).as_posix() for j in jobs], ["a/seq1", "a/seq2"])

    def test_job_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "3ddfa" / "frames").mkdir(parents=True)
            (out / "3ddfa" / "frames" / "0000.png").write_bytes(b"x")
            (out / "liveportrait").mkdir()
            (out / "liveportrait" / "error.txt").write_text("boom")
            (out / "compare.mp4").write_bytes(b"x")
            self.assertTrue(job_done(out, ["3ddfa", "liveportrait"]))
            self.assertFalse(job_done(out, ["3ddfa", "facelift"]))


if __name__ == "__main__":
    unittest.main()
