from __future__ import annotations

import json
import importlib.util
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_verifier():
    spec = importlib.util.spec_from_file_location(
        "lorango_verify_run", ROOT / "scripts" / "verify_run.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StrictRunVerifierTests(unittest.TestCase):
    def test_forged_run_cannot_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            documents = {
                "INPUT_CONFIG.json": {"gates": {}},
                "INPUT_DATA.json": {"train": [], "calibration": []},
                "INPUT_MODEL_MANIFEST.json": {"status": "NOT_DECLARED"},
                "PROTOCOL.json": {
                    "version": "1.4.0",
                    "config": {"gates": {}},
                    "data": {"train": [], "calibration": []},
                },
                "RUN_METADATA.json": {
                    "toolkit_version": "1.4.0",
                    "mode": "train",
                    "inputs": {"base_model": {}},
                },
                "training_trace.json": [],
                "checkpoint_trained.json": {
                    "format": "lorango-checkpoint-v1",
                    "tensor_file": "checkpoint_trained.safetensors",
                    "tensor_sha256": "0" * 64,
                    "layers": [],
                },
                "CALIBRATION.json": {"rows": [], "mean_cosine": -1},
                "POST_EXPORT_CALIBRATION.json": {"rows": []},
                "COMPILE.json": {"files": {}, "layers": []},
                "MANIFEST.json": {
                    "toolkit_version": "1.4.0",
                    "status": "PASS",
                    "ordinary_static_lora": True,
                    "post_export_gates": {"forged": True},
                    "compilation": {"files": {}, "layers": []},
                },
                "EVIDENCE.json": {
                    "format": "lorango-run-evidence-v2",
                    "toolkit_version": "1.4.0",
                    "files": {},
                    "files_digest": hashlib.sha256(b"{}").hexdigest(),
                },
            }
            for name, value in documents.items():
                (run / name).write_text(json.dumps(value), encoding="utf-8")
            for name in (
                "checkpoint_trained.safetensors",
                "signature.safetensors",
                "payload.safetensors",
                "modified_s.safetensors",
                "modified_p.safetensors",
            ):
                (run / name).write_text("forged", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "verify_run.py"),
                    "--run-dir",
                    str(run),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            result = json.loads(completed.stdout)
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(result["failures"])

    def test_evidence_aggregate_detects_tampering(self) -> None:
        verifier = load_verifier()
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            artifact = run / "record.json"
            artifact.write_text('{"value": 1}\n', encoding="utf-8")
            records = {
                artifact.name: {
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    "bytes": artifact.stat().st_size,
                }
            }
            (run / "EVIDENCE.json").write_text(
                json.dumps(
                    {
                        "format": "lorango-run-evidence-v2",
                        "toolkit_version": "1.4.0",
                        "files": records,
                        "files_digest": verifier.canonical_hash(records),
                    }
                ),
                encoding="utf-8",
            )
            failures = []
            verifier.verify_evidence(run, failures)
            self.assertEqual(failures, [])
            artifact.write_text('{"value": 2}\n', encoding="utf-8")
            failures = []
            verifier.verify_evidence(run, failures)
            self.assertTrue(any("hash/size mismatch" in item for item in failures))

    def test_model_manifest_snapshot_binding_detects_tampering(self) -> None:
        verifier = load_verifier()
        snapshot = json.loads(
            (ROOT / "reference" / "MODEL_MANIFEST.json").read_text(encoding="utf-8")
        )
        weight_files = {
            row["relative_path"]: {"sha256": row["sha256"], "bytes": row["bytes"]}
            for row in snapshot["weight_files"]
        }
        metadata_files = {
            row["relative_path"]: {"sha256": row["sha256"], "bytes": row["bytes"]}
            for row in snapshot["metadata_files"]
        }
        manifest_path = "/frozen/MODEL_MANIFEST.json"
        metadata = {
            "inputs": {
                "base_model": {
                    "weight_files": weight_files,
                    "weights_digest": verifier.canonical_hash(weight_files),
                    "metadata_files": metadata_files,
                    "metadata_digest": verifier.canonical_hash(metadata_files),
                },
                "model_manifest": {
                    "path": manifest_path,
                    "binding_status": "MATCH",
                    "canonical_sha256": verifier.canonical_hash(snapshot),
                    "weights_digest": snapshot["weights_digest"],
                    "metadata_digest": snapshot["metadata_digest"],
                },
            }
        }
        failures = []
        verifier.verify_model_manifest_binding(
            {"model_manifest": manifest_path}, metadata, snapshot, failures
        )
        self.assertEqual(failures, [])
        snapshot["weight_files"][0]["bytes"] += 1
        failures = []
        verifier.verify_model_manifest_binding(
            {"model_manifest": manifest_path}, metadata, snapshot, failures
        )
        self.assertTrue(failures)


if __name__ == "__main__":
    unittest.main()
