"""Pre-run lock refuses changed inputs and never upgrades a pilot by itself."""

import json
import tempfile
import unittest
from pathlib import Path

from shared_evidence.freeze import build_manifest, verify_manifest


class FreezeTests(unittest.TestCase):
    def test_manifest_binds_current_source_and_config(self) -> None:
        root = Path(__file__).resolve().parent.parent
        config = root / "configs/production-primary-v1.json"
        protocol = root / "docs/research_protocol.md"
        manifest = build_manifest(config, protocol, bootstrap_resamples=10_000)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(len(verify_manifest(path, config)), 64)
            manifest["source_sha256"]["shared_evidence/model.py"] = "0" * 64
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs"):
                verify_manifest(path, config)

    def test_development_config_cannot_be_frozen(self) -> None:
        root = Path(__file__).resolve().parent.parent
        with self.assertRaisesRegex(ValueError, "production"):
            build_manifest(
                root / "configs/development-pilot-v0.json",
                root / "docs/research_protocol.md",
            )


if __name__ == "__main__":
    unittest.main()
