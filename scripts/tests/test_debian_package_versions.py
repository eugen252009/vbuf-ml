from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
VERSION_PATH = ROOT / "packaging/debian/version.py"
spec = importlib.util.spec_from_file_location("vbuf_debian_version", VERSION_PATH)
assert spec and spec.loader
version = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = version
spec.loader.exec_module(version)


class DebianVersionMappingTests(unittest.TestCase):
    def test_canonical_tag_mappings(self) -> None:
        cases = {
            "alpha-0.1.0": ("alpha", "alpha", "0.1.0~alpha-1"),
            "alpha-0.1.0.2": ("alpha", "alpha", "0.1.0~alpha.2-1"),
            "alpha-0.1.0.10": ("alpha", "alpha", "0.1.0~alpha.10-1"),
            "beta-0.1.0": ("beta", "beta", "0.1.0~beta-1"),
            "beta-0.1.0.2": ("beta", "beta", "0.1.0~beta.2-1"),
            "release-0.1.0": ("release", "release", "0.1.0-1"),
            "release-0.1.0-r2": ("release", "release", "0.1.0-2"),
            "release-0.1.0-r10": ("release", "release", "0.1.0-10"),
        }
        for tag, expected in cases.items():
            with self.subTest(tag=tag):
                parsed = version.parse_release_tag(tag)
                self.assertEqual(
                    (parsed.channel, parsed.apt_suite, parsed.debian_version), expected
                )
                self.assertEqual(parsed.tag, tag)

    def test_noncanonical_and_unrelated_tags_are_not_release_aliases(self) -> None:
        rejected = (
            "v0.1.0",
            "alpha-v0.1.0",
            "alpha-00.1.0",
            "alpha-0.01.0",
            "alpha-0.1.00",
            "alpha-0.1.0.0",
            "alpha-0.1.0.1",
            "alpha-0.1.0.02",
            "beta-0.1.0.1",
            "release-0.1.0-r0",
            "release-0.1.0-r1",
            "release-0.1.0-r02",
            "release-0.1.0.2",
            "rc-0.1.0",
            "model-is-working",
        )
        for tag in rejected:
            with self.subTest(tag=tag):
                with self.assertRaises(ValueError):
                    version.parse_release_tag(tag)

    def test_only_managed_prefixes_are_strictly_parsed(self) -> None:
        self.assertTrue(version.is_release_namespace("alpha-not-a-version"))
        self.assertTrue(version.is_release_namespace("release-0.1.0"))
        self.assertFalse(version.is_release_namespace("model-is-working"))
        self.assertFalse(version.is_release_namespace("vbuf-ml-0.1.0-direct-view-baseline"))

    @unittest.skipUnless(shutil.which("dpkg"), "dpkg is required to test Debian ordering")
    def test_debian_versions_order_by_channel_and_revision(self) -> None:
        ordered = [
            "0.1.0~alpha-1",
            "0.1.0~alpha.2-1",
            "0.1.0~beta-1",
            "0.1.0~beta.2-1",
            "0.1.0-1",
            "0.1.0-2",
        ]
        for lower, higher in zip(ordered, ordered[1:]):
            with self.subTest(lower=lower, higher=higher):
                subprocess.run(
                    ["dpkg", "--compare-versions", lower, "lt", higher], check=True
                )

    def test_build_inputs_are_explicit_and_exclude_research_payloads(self) -> None:
        build_path = ROOT / "packaging/debian/build.py"
        text = build_path.read_text(encoding="utf-8")
        self.assertIn('"integrations/ggml"', text)
        self.assertIn('"rust/Cargo.lock"', text)
        self.assertIn('"rust/vbuf-ml"', text)
        self.assertNotIn('"research"', text)
        self.assertNotIn('"research-models"', text)


if __name__ == "__main__":
    unittest.main()
