from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from lorango_sdxl.carrier import cover_rank_profile, expected_carrier_keys, select_carriers
from lorango_sdxl.checkpoint import load_checkpoint
from lorango_sdxl.config import load_data


ROOT = Path(__file__).resolve().parents[1]


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scaffold = load_checkpoint(ROOT / "assets" / "scaffold_init.safetensors")

    def test_example_data(self) -> None:
        data = load_data(ROOT / "examples" / "train_data.example.json")
        self.assertEqual(len(data["train"]), 8)
        self.assertEqual(len(data["calibration"]), 2)

    def test_duplicate_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(
                json.dumps(
                    {
                        "train": [["x", "base", "target"]],
                        "calibration": [["x", "base2", "target2"]],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_data(path)

    def test_duplicate_prompts_across_splits_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(
                json.dumps(
                    {
                        "train": [["train", "same prompt", "target one"]],
                        "calibration": [["cal", " SAME   PROMPT ", "target two"]],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "must be disjoint"):
                load_data(path)

    def test_four_groups_are_disjoint(self) -> None:
        channels_by_block: dict[str, set[int]] = {}
        for group in range(4):
            config = {
                "carrier_catalog": str(ROOT / "assets" / "carrier_catalog.json"),
                "carrier": {"group_index": group},
            }
            selected = select_carriers(config, self.scaffold)
            self.assertEqual(len(selected), 5)
            self.assertEqual(len(expected_carrier_keys(selected)), 20)
            for row in selected:
                used = channels_by_block.setdefault(row["block"], set())
                current = set(row["channels"])
                self.assertFalse(used.intersection(current))
                used.update(current)

    def test_frozen_assets_match_manifest(self) -> None:
        manifest = json.loads((ROOT / "assets" / "ASSET_MANIFEST.json").read_text())
        for name, spec in manifest["assets"].items():
            digest = hashlib.sha256((ROOT / "assets" / name).read_bytes()).hexdigest()
            self.assertEqual(digest, spec["sha256"])

    def test_reference_cover_scaling_is_explicit(self) -> None:
        signature = cover_rank_profile(
            ROOT / "reference" / "inputs" / "utility_s.safetensors",
            [5] * 5 + [4] * 5,
        )
        payload = cover_rank_profile(
            ROOT / "reference" / "inputs" / "utility_p.safetensors",
            [4] * 10,
        )
        self.assertEqual(signature["inferred_loader_alpha"], 4)
        self.assertEqual(payload["inferred_loader_alpha"], 4)

    def test_runtime_contains_no_pickle_torch_serialization(self) -> None:
        sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "lorango_sdxl").glob("*.py")
        )
        self.assertNotIn("torch.load(", sources)
        self.assertNotIn("torch.save(", sources)

    def test_required_runtime_dependency_is_pinned(self) -> None:
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("peft==0.12.0", requirements.splitlines())

    def test_reference_model_manifest_covers_weights(self) -> None:
        manifest = json.loads(
            (ROOT / "reference" / "MODEL_MANIFEST.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["weight_files_count"], 4)
        self.assertEqual(manifest["weight_bytes"], 6938011430)
        self.assertEqual(len(manifest["weights_digest"]), 64)
        self.assertEqual(manifest["metadata_files_count"], 14)
        self.assertEqual(len(manifest["metadata_files"]), 14)
        self.assertEqual(len(manifest["metadata_digest"]), 64)


if __name__ == "__main__":
    unittest.main()
