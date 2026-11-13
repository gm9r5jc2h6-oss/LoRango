from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from xarch.audit import audit_record
from xarch.candidates import generate_candidates
from xarch.config import load_architecture, load_policy
from xarch.metrics import four_condition_metrics
from xarch.scoring import rank_candidates, select_candidates
from xarch.util import load_json, load_jsonl


ROOT = Path(__file__).resolve().parent.parent


class PipelineTests(unittest.TestCase):
    def test_four_condition_metrics(self) -> None:
        metrics = four_condition_metrics(
            s_clean=[0.0, 0.0],
            p_clean=[0.0, 0.0],
            sp_clean=[0.0, 0.0],
            s_output=[0.1, 0.0],
            p_output=[0.0, 0.1],
            sp_output=[1.1, 1.1],
            target=[1.0, 1.0],
        )
        self.assertAlmostEqual(metrics["interaction_cosine"], 1.0)
        self.assertAlmostEqual(metrics["s_drift_ratio"], 0.005)
        self.assertAlmostEqual(metrics["p_drift_ratio"], 0.005)

    def test_demo_selection_is_deterministic(self) -> None:
        architecture = load_architecture(ROOT / "architectures" / "synthetic_demo.json")
        policy = load_policy(ROOT / "policies" / "controlled_v1.json")
        candidates = generate_candidates(
            architecture, load_json(ROOT / "examples" / "synthetic_inventory.json")
        )
        self.assertEqual(candidates["candidate_count"], 4)
        ranked = rank_candidates(
            candidates,
            load_jsonl(ROOT / "examples" / "synthetic_probe_metrics.jsonl"),
            policy,
        )
        selected = select_candidates(ranked, policy)
        self.assertEqual(selected["status"], "selected")
        self.assertEqual(selected["selected"][0]["candidate_id"], "joint_attention_b2_to_b5")
        self.assertEqual(selected["selected"][0]["rank"], 4)

    def test_negative_result_is_valid(self) -> None:
        policy = load_policy(ROOT / "policies" / "controlled_v1.json")
        selected = select_candidates({"rows": []}, policy)
        self.assertEqual(selected["status"], "no_supported_construction_found")

    def test_minimum_audit_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            signature = root / "signature.safetensors"
            payload = root / "payload.safetensors"
            signature.write_bytes(b"controlled-signature")
            payload.write_bytes(b"controlled-payload")
            sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            conditions = {
                name: {"completed": True, "finite": True, "shape": [1, 4, 8], "metrics": {}}
                for name in ["Base", "S", "P", "SP"]
            }
            record = {
                "training": {"configured_updates": 2, "completed_updates": 2, "finite": True},
                "files": {
                    "signature": {"path": str(signature), "sha256": sha(signature), "loader_registered": True},
                    "payload": {"path": str(payload), "sha256": sha(payload), "loader_registered": True},
                },
                "hooks_after_load": 0,
                "conditions": conditions,
            }
            self.assertEqual(audit_record(record)["status"], "PASS")


if __name__ == "__main__":
    unittest.main()

