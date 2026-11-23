"""video-studio 核心模块单元与端到端合成测试。"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
from PIL import Image

from studio.compliance import check_project_compliance
from studio.compose import compose_project, swap_hosts
from studio.config import ffmpeg_bin
from studio.h3 import build_h3_graph, build_h3_prompt, prepare_h3_graphs, submit_h3
from studio.project import init_project, load_json, save_json
from studio.qc import run_final_qc
from studio.subtitles import align_chapter_cues, generate_subtitles, split_phrases
from studio.sync_gate import audio_stream_sha256, compute_candidate_shifts, is_sync_accepted
from studio.tts import build_audio_timeline


def _write_tone_wav(path: Path, duration_sec: float, freq: float = 440.0, sr: int = 24000) -> None:
    n = int(round(duration_sec * sr))
    t = np.arange(n, dtype=np.float64) / sr
    samples = (np.sin(2.0 * math.pi * freq * t) * 12000).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())


def _write_solid_mp4(path: Path, duration_sec: float, color: str, size: str = "960x960", fps: int = 24) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg_bin(),
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s={size}:r={fps}:d={duration_sec:.3f}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=330:sample_rate=24000:duration={duration_sec:.3f}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
    )


def _sample_frame_rgb(video: Path, sec: float) -> np.ndarray:
    raw = subprocess.check_output([
        ffmpeg_bin(),
        "-nostdin",
        "-v",
        "error",
        "-ss",
        f"{sec:.3f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ])
    return np.frombuffer(raw, dtype=np.uint8).reshape(1080, 1920, 3)


class ComplianceTests(unittest.TestCase):
    def test_valid_software_explainer_warns_only_on_unconfirmed_portrait(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pf = init_project(Path(td) / "demo", title="测试片", use_case="U8", industry="I20")
            m = load_json(pf)
            res = check_project_compliance(m)
            self.assertEqual(res["status"], "warn")
            self.assertEqual([w["code"] for w in res["warnings"]], ["PORTRAIT_AUTH_UNCONFIRMED"])
            m["presenter"]["portraitAuthorized"] = True
            res2 = check_project_compliance(m)
            self.assertEqual(res2["status"], "pass")
            self.assertGreaterEqual(len(res2["humanReviewChecklist"]), 5)

    def test_blocked_industry_and_restricted_pair_and_medical_term(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pf = init_project(Path(td) / "med", title="医疗测试", use_case="U1", industry="I15")
            m = load_json(pf)
            res = check_project_compliance(m)
            self.assertEqual(res["status"], "blocked")
            codes = {e["code"] for e in res["errors"]}
            self.assertIn("INDUSTRY_BLOCKED", codes)
            self.assertIn("RESTRICTED_PAIR", codes)

            m["industry"] = "I7"
            m["useCase"] = "U7"
            m["chapters"][0]["narration"] = "这款训练课包治百病，99%有效。"
            m["brief"]["facts"] = ["课程时长 30 天"]
            m["presenter"]["aiLabel"] = ""
            res2 = check_project_compliance(m)
            codes2 = {e["code"] for e in res2["errors"]}
            warn_codes = {w["code"] for w in res2["warnings"]}
            self.assertIn("RESTRICTED_PAIR", codes2)
            self.assertIn("MEDICAL_CLAIM_OUTSIDE_MEDICAL", codes2)
            self.assertIn("MISSING_AI_LABEL", codes2)
            self.assertIn("UNVERIFIED_NUMBERS", warn_codes)


class SubtitleTests(unittest.TestCase):
    def test_split_and_asr_clamp_inside_duration(self) -> None:
        text = "第一步是定尺寸。光是这座殿有多宽，就能查到好几个数。我们采用三维扫描反推的设计值，十一点五五米。"
        phrases = split_phrases(text, max_chars=24)
        self.assertGreaterEqual(len(phrases), 3)
        # Simulate ASR token exceeding wav duration by 2 seconds (known timing bug).
        asr = [
            {
                "tokens": [
                    {"text": "第一步是定尺寸", "offsets": {"from": 0, "to": 1800}},
                    {"text": "十一点五五米", "offsets": {"from": 4800, "to": 8000}},
                ]
            }
        ]
        cues, mode = align_chapter_cues(narration=text, start_sec=10.0, duration_sec=6.0, asr_transcription=asr)
        self.assertEqual(mode, "asr-anchored")
        for cue in cues:
            self.assertGreaterEqual(cue["start"], 10.0)
            self.assertLessEqual(cue["end"], 16.0)
            self.assertGreater(cue["end"], cue["start"])


class H3GraphTests(unittest.TestCase):
    def test_prompt_and_graph_and_pilot_guard(self) -> None:
        prompt = build_h3_prompt(
            subject_description="a Chinese tech presenter",
            dialogue="先看结果。左边看口型，右边看手势。",
            gestures=[{"start": 1.2, "end": 2.4, "phrase": "左边看口型", "action": "offer one open palm left"}],
            dual_reference=True,
        )
        self.assertIn("<Picture 1> and <Picture 2>", prompt)
        self.assertIn("“左边看口型” (1.20-2.40 s)", prompt)
        self.assertIn("<d>[Chinese] 先看结果。左边看口型，右边看手势。 </d>", prompt)

        graph = build_h3_graph(
            remote_waist="proj/waist.png",
            remote_portrait="proj/portrait.png",
            remote_audio="proj/voice-01.wav",
            prompt_text=prompt,
            seconds=7.5,
            seed=7822077,
            filename_prefix="proj/shot_01",
            client_id="proj-01",
        )
        self.assertEqual(graph["prompt"]["3"]["inputs"]["audio_mode"], "数字人（锁定音频）")
        self.assertEqual(graph["prompt"]["91"]["inputs"]["refine_scale"], 1.5)
        media = json.loads(graph["prompt"]["11"]["inputs"]["inline_media_state"])
        self.assertEqual(len(media["images"]), 2)
        self.assertEqual(media["audios"][0]["filename"], "proj/voice-01.wav")

        with tempfile.TemporaryDirectory() as td:
            proj = Path(td) / "p1"
            pf = init_project(proj, title="多章测试")
            m = load_json(pf)
            m["chapters"].append({**m["chapters"][0], "id": "02", "title": "第二章"})
            save_json(pf, m)
            written = prepare_h3_graphs(proj, upload_media=False)
            self.assertEqual(len(written), 2)
            with self.assertRaisesRegex(RuntimeError, "样片先行"):
                submit_h3(proj, take=1, pilot_only=False)


class SyncGateMathTests(unittest.TestCase):
    def test_stable_offsets_produce_expected_24fps_shift(self) -> None:
        raw = {
            "overall": {"picture_late_ms": 120, "confidence": 4.2},
            "windows": [
                {"at": 1.0, "picture_late_ms": 120, "confidence": 3.5},
                {"at": 2.2, "picture_late_ms": 80, "confidence": 3.8},
                {"at": 3.4, "picture_late_ms": 120, "confidence": 4.0},
            ],
        }
        shifts = compute_candidate_shifts(raw, fps=24, spread_max_ms=80)
        self.assertIn(3, shifts)

        early = {
            "overall": {"picture_late_ms": -240, "confidence": 3.5},
            "windows": [
                {"at": 1.0, "picture_late_ms": -280, "confidence": 3.1},
                {"at": 2.2, "picture_late_ms": -240, "confidence": 3.4},
            ],
        }
        self.assertIn(-6, compute_candidate_shifts(early, fps=24, spread_max_ms=80))

    def test_unstable_or_boundary_offsets_refuse_shift(self) -> None:
        unstable = {
            "overall": {"picture_late_ms": 40, "confidence": 2.0},
            "windows": [
                {"at": 1.0, "picture_late_ms": -80, "confidence": 2.0},
                {"at": 2.2, "picture_late_ms": 160, "confidence": 2.0},
            ],
        }
        self.assertEqual(compute_candidate_shifts(unstable), [])

        boundary = {
            "overall": {"picture_late_ms": 360, "confidence": 2.0},
            "windows": [{"at": 1.0, "picture_late_ms": 360, "confidence": 2.0}],
        }
        self.assertEqual(compute_candidate_shifts(boundary), [])

        self.assertTrue(is_sync_accepted({
            "overall": {"picture_late_ms": 0, "confidence": 3.2},
            "windows": [{"picture_late_ms": -40, "confidence": 2.1}, {"picture_late_ms": 0, "confidence": 2.9}],
        }))
        self.assertFalse(is_sync_accepted({
            "overall": {"picture_late_ms": 0, "confidence": 3.2},
            "windows": [{"picture_late_ms": 80, "confidence": 2.1}],
        }))


class EndToEndComposeAndSwapTests(unittest.TestCase):
    def test_full_pipeline_all_layouts_and_lossless_host_swap(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            proj = Path(td) / "demo-film"
            pf = init_project(proj, title="实测短片", subtitle="端到端验证")
            m = load_json(pf)
            m["cover"]["enabled"] = False
            m["chapters"] = [
                {
                    "id": "01",
                    "title": "分屏与大窗",
                    "sectionIndex": 0,
                    "narration": "先看左边蓝色素材和右边绿色数字人，随后切到宽画幅大窗展示。",
                    "onCameraText": "先看左边蓝色素材和右边绿色数字人。",
                    "onCameraSec": 2.0,
                    "heroLines": ["87 项参数", "端到端实测"],
                    "cuts": [
                        {"at": 0.0, "layout": "split", "source": "assets/blue.png"},
                        {"at": 1.0, "layout": "host", "source": ""},
                        {"at": 2.0, "layout": "wide", "source": "assets/orange.png"},
                    ],
                },
                {
                    "id": "02",
                    "title": "全屏镜头",
                    "sectionIndex": 1,
                    "narration": "最后用全屏画面收束全片。",
                    "onCameraText": "",
                    "onCameraSec": 0,
                    "cuts": [
                        {"at": 0.0, "layout": "full", "source": "assets/orange.png"},
                    ],
                },
            ]
            save_json(pf, m)

            Image.new("RGB", (1280, 720), "#1e64c8").save(proj / "assets" / "blue.png")
            Image.new("RGB", (1280, 720), "#d97828").save(proj / "assets" / "orange.png")
            _write_tone_wav(proj / "work" / "narration" / "01.wav", 3.0, freq=440.0)
            _write_tone_wav(proj / "work" / "narration" / "02.wav", 1.5, freq=520.0)
            _write_solid_mp4(proj / "work" / "h3" / "accepted" / "01.mp4", 2.2, color="0x20a848")

            tl = build_audio_timeline(proj, chapter_gap_sec=0.0)
            self.assertAlmostEqual(tl["totalDuration"], 4.5, places=1)
            self.assertTrue((proj / "work" / "h3" / "voice" / "voice-01.wav").is_file())

            sub = generate_subtitles(proj)
            self.assertGreaterEqual(sub["cueCount"], 2)

            comp = compose_project(proj)
            out_mp4 = Path(comp["output"])
            self.assertTrue(out_mp4.is_file())
            self.assertEqual(comp["frames"], comp["expectedFrames"])
            self.assertEqual(comp["hostWindows"], 2)

            # Content assertions on real extracted frames (not container-only checks):
            # 0.5s -> split: paper background top-left, blue media box, green host box.
            f_split = _sample_frame_rgb(out_mp4, 0.5)
            bg_rgb = f_split[15, 15]
            self.assertGreater(int(bg_rgb[0]), 220)
            self.assertGreater(int(bg_rgb[1]), 220)
            media_rgb = f_split[500, 500]  # inside split media (72..1160, 197..809)
            self.assertGreater(int(media_rgb[2]), 150)  # blue channel dominant
            self.assertLess(int(media_rgb[0]), 80)
            host_rgb = f_split[500, 1500]  # inside split host (1186..1848, 174..836)
            self.assertGreater(int(host_rgb[1]), 130)  # green channel dominant
            self.assertLess(int(host_rgb[0]), 80)

            # 1.5s -> host layout: large 804x804 green host box at (1008,137).
            f_host = _sample_frame_rgb(out_mp4, 1.5)
            self.assertGreater(int(f_host[500, 1400][1]), 130)

            # 3.8s -> full orange frame.
            f_full = _sample_frame_rgb(out_mp4, 3.8)
            self.assertGreater(int(f_full[500, 500][0]), 180)
            self.assertLess(int(f_full[500, 500][2]), 80)

            # Swap host from green -> red and assert audio hash matches byte-for-byte.
            alt_dir = proj / "work" / "alt-hosts"
            _write_solid_mp4(alt_dir / "01.mp4", 2.2, color="0xc82828")
            swapped_mp4 = proj / "outputs" / "swapped.mp4"
            swap_res = swap_hosts(
                base_video=out_mp4,
                windows_json=proj / "work" / "host-windows.json",
                hosts_dir=alt_dir,
                output_video=swapped_mp4,
                fps=30,
            )
            self.assertTrue(swap_res["audioHashMatchedBase"])
            self.assertEqual(audio_stream_sha256(swapped_mp4), audio_stream_sha256(out_mp4))
            f_swapped = _sample_frame_rgb(swapped_mp4, 0.5)
            # Media box still blue, host box now red!
            self.assertGreater(int(f_swapped[500, 500][2]), 150)
            self.assertGreater(int(f_swapped[500, 1500][0]), 150)
            self.assertLess(int(f_swapped[500, 1500][1]), 80)

            # QC gate check (windows < 2.0s skipped for SyncNet, container + LUFS + audio hash + 12-frame grid verified).
            qc_res = run_final_qc(
                swapped_mp4,
                report_dir=proj / "outputs" / "qc",
                windows_json=proj / "work" / "host-windows.json",
                base_audio_video=out_mp4,
            )
            self.assertTrue(qc_res["passed"], qc_res["failures"])
            self.assertTrue(qc_res["audioMatchedBase"])
            self.assertTrue(Path(qc_res["visualGrid"]).is_file())


if __name__ == "__main__":
    unittest.main()
