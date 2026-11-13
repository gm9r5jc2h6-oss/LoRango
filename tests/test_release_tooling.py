from __future__ import annotations

import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        f"lorango_test_{name}", ROOT / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseToolingTests(unittest.TestCase):
    def test_release_builders_use_only_tracked_files(self) -> None:
        for name in ("build_release", "build_anonymous_review"):
            module = load_script(name)
            with self.subTest(builder=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                subprocess.run(["git", "init", "-q"], cwd=root, check=True)
                (root / "tracked.txt").write_text("tracked\n", encoding="utf-8")
                (root / "untracked.txt").write_text("untracked\n", encoding="utf-8")
                subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
                module.ROOT = root
                self.assertEqual(
                    [path.relative_to(root) for path in module.source_paths()],
                    [Path("tracked.txt")],
                )

    @unittest.skipUnless(hasattr(os, "symlink"), "symbolic links are unavailable")
    def test_reassembler_rejects_symlink_chunks(self) -> None:
        module = load_script("reassemble_large_assets")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            target = root / "target.bin"
            target.write_bytes(b"payload")
            (root / "chunk.part-000").symlink_to(target)
            module.ROOT = root
            with self.assertRaisesRegex(ValueError, "Symbolic|symbolic"):
                module.safe_path("chunk.part-000")

    def test_reassembler_rejects_escaping_paths(self) -> None:
        module = load_script("reassemble_large_assets")
        with tempfile.TemporaryDirectory() as directory:
            module.ROOT = Path(directory).resolve()
            with self.assertRaises(ValueError):
                module.safe_path("../outside.bin")
            with self.assertRaises(ValueError):
                module.safe_path("/absolute.bin")


if __name__ == "__main__":
    unittest.main()
