"""Tests for tools/build_triplet_manifest.py."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOL_PATH = _REPO_ROOT / "tools" / "build_triplet_manifest.py"
_SPEC = importlib.util.spec_from_file_location("build_triplet_manifest", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TOOL
_SPEC.loader.exec_module(_TOOL)


class BuildTripletManifestTest(unittest.TestCase):
    def _make_roots(self, root: Path):
        roots = {name: root / name for name in ("hr", "hq", "lr")}
        for path in roots.values():
            path.mkdir()
        return roots

    @staticmethod
    def _touch(path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"media")

    def test_relative_stem_matches_nested_video_triplets(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = self._make_roots(Path(tmp))
            for sample in ("scene_a/clip_001.mp4", "scene_b/clip_002.mkv"):
                for dataset_root in roots.values():
                    self._touch(dataset_root / sample)

            records, summary = _TOOL.build_manifest(
                hr_root=roots["hr"],
                hq_root=roots["hq"],
                lr_root=roots["lr"],
                val_fraction=0.0,
                strict=True,
            )

            self.assertEqual(
                [record.sample_id for record in records],
                ["scene_a/clip_001", "scene_b/clip_002"],
            )
            self.assertTrue(all(record.media_type == "video" for record in records))
            self.assertEqual(summary["triplet_count"], 2)
            self.assertEqual(summary["media_mode"], "video")

    def test_frame_sequences_are_grouped_by_six_digit_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = self._make_roots(Path(tmp))
            for clip in ("video1_comp2", "video3_comp4"):
                for frame in range(3):
                    for dataset_root in roots.values():
                        self._touch(dataset_root / f"{clip}_{frame:06d}.png")

            records, summary = _TOOL.build_manifest(
                hr_root=roots["hr"],
                hq_root=roots["hq"],
                lr_root=roots["lr"],
                val_fraction=0.0,
                strict=True,
            )

            self.assertEqual(
                [record.sample_id for record in records],
                ["video1_comp2", "video3_comp4"],
            )
            first = records[0]
            self.assertEqual(first.media_type, "frames")
            self.assertEqual(first.frame_start, 0)
            self.assertEqual(first.frame_end, 2)
            self.assertEqual(first.frame_count, 3)
            self.assertEqual(first.frame_digits, 6)
            self.assertTrue(first.hr.endswith("video1_comp2_{frame:06d}.png"))
            self.assertEqual(summary["media_mode"], "frames")

    def test_text_suffix_is_preserved_in_frame_pattern(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = self._make_roots(Path(tmp))
            for frame in range(2):
                for dataset_root in roots.values():
                    self._touch(dataset_root / f"video1_comp2_{frame:06d}_text.png")

            regex = r"^(?P<clip>.+)_(?P<frame>\d{6})_text$"
            records, summary = _TOOL.build_manifest(
                hr_root=roots["hr"],
                hq_root=roots["hq"],
                lr_root=roots["lr"],
                media_mode="frames",
                frame_regex=regex,
                val_fraction=0.0,
                strict=True,
            )

            self.assertEqual(records[0].sample_id, "video1_comp2")
            self.assertTrue(
                records[0].lr.endswith("video1_comp2_{frame:06d}_text.png")
            )
            self.assertEqual(summary["frame_regexes"]["lr"], regex)

    def test_per_root_regex_maps_plain_targets_to_text_lr(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = self._make_roots(Path(tmp))
            for frame in range(3):
                self._touch(roots["hr"] / f"video1_comp2_{frame:06d}.png")
                self._touch(roots["hq"] / f"video1_comp2_{frame:06d}.png")
                self._touch(roots["lr"] / f"video1_comp2_{frame:06d}_text.png")

            text_regex = r"^(?P<clip>.+)_(?P<frame>\d{6})_text$"
            records, summary = _TOOL.build_manifest(
                hr_root=roots["hr"],
                hq_root=roots["hq"],
                lr_root=roots["lr"],
                media_mode="frames",
                lr_frame_regex=text_regex,
                val_fraction=0.0,
                strict=True,
            )

            record = records[0]
            self.assertTrue(record.hr.endswith("video1_comp2_{frame:06d}.png"))
            self.assertTrue(record.hq.endswith("video1_comp2_{frame:06d}.png"))
            self.assertTrue(record.lr.endswith("video1_comp2_{frame:06d}_text.png"))
            self.assertEqual(summary["frame_regexes"]["lr"], text_regex)

    def test_nested_frame_sequences_keep_relative_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = self._make_roots(Path(tmp))
            for dataset_root in roots.values():
                self._touch(dataset_root / "scene_a/video1_comp2_000001.png")
                self._touch(dataset_root / "scene_a/video1_comp2_000002.png")

            records, _ = _TOOL.build_manifest(
                hr_root=roots["hr"],
                hq_root=roots["hq"],
                lr_root=roots["lr"],
                media_mode="frames",
                val_fraction=0.0,
                strict=True,
            )
            self.assertEqual(records[0].sample_id, "scene_a/video1_comp2")
            self.assertEqual(records[0].frame_start, 1)

    def test_strict_rejects_mismatched_frame_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = self._make_roots(Path(tmp))
            for frame in (0, 1, 2):
                self._touch(roots["hr"] / f"clip_{frame:06d}.png")
                self._touch(roots["hq"] / f"clip_{frame:06d}.png")
            for frame in (0, 2):
                self._touch(roots["lr"] / f"clip_{frame:06d}.png")

            with self.assertRaisesRegex(ValueError, "Frame sequence alignment failed"):
                _TOOL.build_manifest(
                    hr_root=roots["hr"],
                    hq_root=roots["hq"],
                    lr_root=roots["lr"],
                    media_mode="frames",
                    strict=True,
                )

    def test_non_strict_skips_non_contiguous_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = self._make_roots(Path(tmp))
            for dataset_root in roots.values():
                for frame in (0, 1):
                    self._touch(dataset_root / f"good_{frame:06d}.png")
                for frame in (0, 2):
                    self._touch(dataset_root / f"bad_{frame:06d}.png")

            records, summary = _TOOL.build_manifest(
                hr_root=roots["hr"],
                hq_root=roots["hq"],
                lr_root=roots["lr"],
                media_mode="frames",
                val_fraction=0.0,
                strict=False,
            )
            self.assertEqual([record.sample_id for record in records], ["good"])
            self.assertEqual(summary["invalid_sequence_count"], 1)

    def test_auto_rejects_different_media_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = self._make_roots(Path(tmp))
            self._touch(roots["hr"] / "clip.mp4")
            self._touch(roots["hq"] / "clip_000000.png")
            self._touch(roots["lr"] / "clip_000000.png")
            with self.assertRaisesRegex(ValueError, "different media modes"):
                _TOOL.build_manifest(
                    hr_root=roots["hr"],
                    hq_root=roots["hq"],
                    lr_root=roots["lr"],
                )

    def test_non_strict_video_keeps_only_complete_triplets(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = self._make_roots(Path(tmp))
            for dataset_root in roots.values():
                self._touch(dataset_root / "complete.mp4")
            self._touch(roots["hr"] / "hr_only.mp4")

            records, summary = _TOOL.build_manifest(
                hr_root=roots["hr"],
                hq_root=roots["hq"],
                lr_root=roots["lr"],
                val_fraction=0.0,
                strict=False,
            )

            self.assertEqual([record.sample_id for record in records], ["complete"])
            self.assertEqual(summary["missing_counts"]["hq"], 1)
            self.assertEqual(summary["missing_counts"]["lr"], 1)

    def test_basename_mode_rejects_duplicate_video_stems(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = self._make_roots(Path(tmp))
            self._touch(roots["hr"] / "a/clip.mp4")
            self._touch(roots["hr"] / "b/clip.mp4")
            self._touch(roots["hq"] / "clip.mp4")
            self._touch(roots["lr"] / "clip.mp4")

            with self.assertRaisesRegex(ValueError, "Duplicate match keys"):
                _TOOL.build_manifest(
                    hr_root=roots["hr"],
                    hq_root=roots["hq"],
                    lr_root=roots["lr"],
                    match_mode="basename_stem",
                )

    def test_split_is_deterministic_and_manifest_is_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roots = self._make_roots(root)
            for index in range(100):
                for dataset_root in roots.values():
                    self._touch(dataset_root / f"clip_{index:03d}.mp4")

            kwargs = dict(
                hr_root=roots["hr"],
                hq_root=roots["hq"],
                lr_root=roots["lr"],
                seed=17,
                val_fraction=0.2,
                test_fraction=0.1,
                strict=True,
            )
            first, first_summary = _TOOL.build_manifest(**kwargs)
            second, _ = _TOOL.build_manifest(**kwargs)
            self.assertEqual(first, second)
            self.assertEqual(sum(first_summary["split_counts"].values()), 100)

            output = root / "manifests/triplets.jsonl"
            _TOOL.write_manifest(first, first_summary, output)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(rows), 100)
            self.assertEqual(rows[0]["media_type"], "video")
            self.assertNotIn("frame_start", rows[0])
            self.assertTrue(output.with_suffix(".jsonl.summary.json").is_file())

    def test_help_runs_as_script(self):
        result = subprocess.run(
            [sys.executable, str(_TOOL_PATH), "--help"],
            cwd=_REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--media-mode", result.stdout)
        self.assertIn("--frame-regex", result.stdout)
        self.assertIn("--lr-frame-regex", result.stdout)


if __name__ == "__main__":
    unittest.main()
