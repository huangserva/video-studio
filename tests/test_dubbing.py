"""逐句配音、局部重配、事实与念法检查、封面与新响度的测试（CosyVoice 和 whisper 用假实现替代）。"""

from __future__ import annotations

import itertools
import math
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from studio.compliance import check_project_compliance
from studio.compose import compose_project, resolve_cut_times
from studio.config import ffmpeg_bin
from studio.project import init_project, load_json, save_json
from studio.qc import run_final_qc
from studio.subtitles import generate_subtitles
from studio.tts import (
    choose_take,
    h3_output_sec,
    level_sentence,
    load_int16,
    pair_sentences,
    redub,
    run_tts,
    split_sentences,
    suggest_on_camera,
    trim_silence,
    voiced_rms_db,
)

SR = 24000


class FakeVoice:
    """假 CosyVoice：按字数出一段带头尾静音的音，每次调用长度略不同；同时充当假 whisper。"""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.text_of: dict[str, str] = {}
        self._jitter = itertools.cycle([0.10, 0.0, 0.05])

    def synth(self, *, endpoint: str, voice_id: str, text: str, speed: float, output: Path, timeout: int = 240) -> Path:
        self.calls.append(text)
        speech = 0.12 * len(text) + next(self._jitter)
        n_sil = int(0.2 * SR)
        n = int(speech * SR)
        t = np.arange(n) / SR
        freq = 180 + 7 * (len(self.calls) % 11)
        tone = (np.sin(2 * math.pi * freq * t) * (0.25 + 0.15 * np.sin(2 * math.pi * 3 * t))) * 32767 * 0.3
        pcm = np.concatenate([np.zeros(n_sil), tone, np.zeros(n_sil)]).astype("<i2")
        output.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(pcm.tobytes())
        self.text_of[str(output).replace(".raw.wav", ".wav")] = text
        return output

    def asr(self, wavs: list[Path], **_: object) -> dict[str, object]:
        out = {}
        for w in wavs:
            text = self.text_of[str(w)]
            with wave.open(str(w), "rb") as f:
                dur_ms = int(f.getnframes() / f.getframerate() * 1000)
            step = dur_ms / max(1, len(text))
            out[str(w)] = [{
                "text": text,
                "offsets": {"from": 0, "to": dur_ms},
                "tokens": [
                    {"text": ch, "offsets": {"from": int(i * step), "to": int((i + 1) * step)}}
                    for i, ch in enumerate(text)
                ],
            }]
        return out


def _demo_project(root: Path, *, cover: bool = False) -> Path:
    pf = init_project(root, title="逐句测试", subtitle="局部重配")
    m = load_json(pf)
    m["cover"]["enabled"] = cover
    m["voice"]["endpoint"] = "http://127.0.0.1:9"
    m["voice"]["voiceId"] = "fake"
    m["presenter"]["portraitAuthorized"] = True
    m["chapters"] = [
        {
            "id": "01",
            "title": "第一章",
            "sectionIndex": 0,
            "narration": "第一句先说结论。第二句讲方法。第三句给数字。第四句收尾。",
            "ttsText": "第一句先说结论。第二句讲方法。第三句给数字。第四句收尾。",
            "onCameraText": "第一句先说结论。",
            "onCameraSec": 1.4,
            "cuts": [
                {"at": 0.0, "layout": "split", "source": "assets/blue.png"},
                {"at": "onCameraEnd", "layout": "wide", "source": "assets/orange.png"},
                {"atSentence": 4, "layout": "full", "source": "assets/orange.png"},
            ],
        },
        {
            "id": "02",
            "title": "第二章",
            "sectionIndex": 1,
            "narration": "第二章只有一句。",
            "onCameraSec": 0,
            "cuts": [{"at": 0.0, "layout": "full", "source": "assets/orange.png"}],
        },
    ]
    save_json(pf, m)
    Image.new("RGB", (1280, 720), "#1e64c8").save(root / "assets" / "blue.png")
    Image.new("RGB", (1280, 720), "#d97828").save(root / "assets" / "orange.png")
    return root


def _write_solid_mp4(path: Path, duration_sec: float, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg_bin(), "-nostdin", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"color=c={color}:s=960x960:r=24:d={duration_sec:.3f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )


def _frame(video: Path, sec: float) -> np.ndarray:
    raw = subprocess.check_output([
        ffmpeg_bin(), "-nostdin", "-v", "error", "-ss", f"{sec:.3f}", "-i", str(video),
        "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
    ])
    return np.frombuffer(raw, dtype=np.uint8).reshape(1080, 1920, 3)


class SentenceUnitTests(unittest.TestCase):
    def test_split_and_pair(self) -> None:
        self.assertEqual(split_sentences("甲。乙！丙？丁"), ["甲。", "乙！", "丙？", "丁"])
        self.assertEqual(split_sentences("他说：“好。”然后走了。"), ["他说：“好。”", "然后走了。"])
        pairs = pair_sentences({"id": "01", "narration": "有 25 万。行。", "ttsText": "有二十五万。行。"})
        self.assertEqual(pairs[0]["tts"], "有二十五万。")
        with self.assertRaises(ValueError):
            pair_sentences({"id": "01", "narration": "一。二。", "ttsText": "一二。"})

    def test_choose_take_prefers_clean_then_short(self) -> None:
        takes = [
            {"take": 1, "durationSec": 2.0, "asrScore": 0.97},
            {"take": 2, "durationSec": 1.8, "asrScore": 0.96},
            {"take": 3, "durationSec": 1.5, "asrScore": 0.80},
        ]
        self.assertEqual(choose_take(takes), 2)
        self.assertEqual(choose_take([{**t, "asrScore": None} for t in takes]), 3)

    def test_trim_and_level(self) -> None:
        t = np.arange(int(1.0 * SR)) / SR
        x = np.concatenate([np.zeros(SR // 2), 0.05 * np.sin(2 * math.pi * 220 * t), np.zeros(SR // 2)]).astype(np.float32)
        y = trim_silence(x, SR)
        self.assertLess(abs(len(y) / SR - 1.08), 0.03)
        z = level_sentence(y, SR, target_db=-20.0, peak_db=-1.0)
        self.assertLess(abs(voiced_rms_db(z, SR) + 20.0), 0.5)
        self.assertLessEqual(float(np.max(np.abs(z))), 10 ** (-1 / 20) + 1e-4)

    def test_h3_rounding_matches_observed(self) -> None:
        for req, out in [(5.8, 5.875), (6.7, 6.583), (8.15, 8.0), (8.24, 8.0), (8.45, 8.708)]:
            self.assertAlmostEqual(h3_output_sec(req), out, places=2)
        sents = [{"startSec": 0.12, "endSec": 3.0}, {"startSec": 3.3, "endSec": 6.2}, {"startSec": 6.5, "endSec": 12.0}]
        sug = suggest_on_camera(sents)
        rec = sug["recommended"]
        self.assertEqual(rec["sentences"], 2)
        self.assertGreaterEqual(rec["h3OutputSec"], 6.2 + 0.05)
        self.assertLess(rec["onCameraSec"], 6.5)

    def test_resolve_cut_times(self) -> None:
        ch = {"id": "01", "cuts": [{"at": 0}, {"at": "onCameraEnd"}, {"atSentence": 2}]}
        sents = [{"index": 1, "startSec": 0.12}, {"index": 2, "startSec": 3.4}]
        self.assertEqual(resolve_cut_times(ch, sentences=sents, on_camera_sec=3.2, host_clip_sec=3.0), [0.0, 3.0, 3.3])


class SentenceDubbingTests(unittest.TestCase):
    def test_redub_only_touches_changed_sentence_and_keeps_host_audio(self) -> None:
        fake = FakeVoice()
        with tempfile.TemporaryDirectory() as td, \
                mock.patch("studio.tts.synthesize_wav", side_effect=fake.synth), \
                mock.patch("studio.tts.transcribe_wavs", side_effect=fake.asr):
            proj = _demo_project(Path(td) / "dub")
            res = run_tts(proj, chapter_gap_sec=0.35)
            ch1 = res["chapters"][0]
            self.assertEqual(ch1["sentences"], 4)
            self.assertEqual(ch1["generated"], [1, 2, 3, 4])
            self.assertEqual(len(fake.calls), 5 * 3)  # 两章共 5 句，每句 3 条
            man = load_json(proj / "work" / "narration" / "01.sentences.json")
            self.assertTrue(all(s["chosenTake"] == 2 for s in man["sentences"]))  # 第 2 条最短
            asr = load_json(proj / "work" / "asr" / "01.json")
            first_tok = asr["transcription"][1]["tokens"][0]["offsets"]["from"]
            self.assertAlmostEqual(first_tok / 1000, man["sentences"][1]["startSec"], delta=0.01)
            self.assertEqual(generate_subtitles(proj)["alignmentModes"]["01"], "asr-anchored")

            old_pcm = load_int16(proj / "work" / "narration" / "01.wav")
            old_tl = load_json(proj / "work" / "timeline.json")
            calls_before = len(fake.calls)

            m = load_json(proj / "project.json")
            m["chapters"][0]["narration"] = "第一句先说结论。第二句讲方法。第三句改成一个更长的数字说法。第四句收尾。"
            m["chapters"][0]["ttsText"] = m["chapters"][0]["narration"]
            save_json(proj / "project.json", m)
            rd = redub(proj, chapter_id="01")
            self.assertEqual([c["index"] for c in rd["changedSentences"]], [3])
            self.assertEqual(rd["reusedSentences"], [1, 2, 4])
            self.assertEqual(len(fake.calls) - calls_before, 3)
            host = {h["chapterId"]: h for h in rd["hostSegments"]}
            self.assertTrue(host["01"]["onCameraAudioUnchanged"])
            new_pcm = load_int16(proj / "work" / "narration" / "01.wav")
            cut = int(man["sentences"][2]["startSec"] * SR)
            self.assertTrue(np.array_equal(old_pcm[:cut], new_pcm[:cut]))
            self.assertGreater(rd["chapterDurationAfterSec"], rd["chapterDurationBeforeSec"])
            self.assertGreater(load_json(proj / "work" / "timeline.json")["chapters"][1]["start"], old_tl["chapters"][1]["start"])
            self.assertEqual(rd["cutsWithFixedSeconds"], [])

            # 重摇一句文字没变的句子，也只动这一句
            rd2 = redub(proj, chapter_id="01", sentences=[4])
            self.assertEqual([c["index"] for c in rd2["changedSentences"]], [4])

            # 改第一句（出镜段里）就必须重做 H3
            m["chapters"][0]["narration"] = "第一句换个说法先说结论。第二句讲方法。第三句改成一个更长的数字说法。第四句收尾。"
            m["chapters"][0]["ttsText"] = m["chapters"][0]["narration"]
            save_json(proj / "project.json", m)
            rd3 = redub(proj, chapter_id="01")
            host3 = {h["chapterId"]: h for h in rd3["hostSegments"]}
            self.assertFalse(host3["01"]["onCameraAudioUnchanged"])

    def test_redub_refuses_legacy_chapter(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            proj = _demo_project(Path(td) / "legacy")
            with self.assertRaises(FileNotFoundError):
                redub(proj, chapter_id="01")


class ComposeCoverAndLoudnessTests(unittest.TestCase):
    def test_cover_lead_in_sentence_cuts_receipt_and_loudness(self) -> None:
        fake = FakeVoice()
        with tempfile.TemporaryDirectory() as td, \
                mock.patch("studio.tts.synthesize_wav", side_effect=fake.synth), \
                mock.patch("studio.tts.transcribe_wavs", side_effect=fake.asr):
            proj = _demo_project(Path(td) / "cover", cover=True)
            Image.new("RGB", (512, 512), "#2a8a4a").save(proj / "assets" / "portrait.png")
            tl = run_tts(proj)["timeline"]
            self.assertAlmostEqual(tl["leadInSec"], 0.7333, places=3)  # 0.75 秒取整到 22 帧
            self.assertAlmostEqual(tl["chapters"][0]["start"], tl["leadInSec"], places=3)
            on_cam = tl["chapters"][0]["onCameraSec"]
            _write_solid_mp4(proj / "work" / "h3" / "accepted" / "01.mp4", on_cam + 0.1, "0x20a848")
            sub = generate_subtitles(proj)
            ass = Path(sub["assPath"]).read_text(encoding="utf-8")
            first = [l for l in ass.splitlines() if l.startswith("Dialogue:")][0]
            self.assertTrue(first.split(",")[1] >= "0:00:00.73")

            # 收据对不上新干声时，合成必须拦下
            save_json(proj / "work" / "h3" / "accepted" / "01.receipt.json", {"drySha256": "0" * 64})
            with self.assertRaises(RuntimeError):
                compose_project(proj)
            files = {f["id"]: f for f in tl["files"]}
            save_json(proj / "work" / "h3" / "accepted" / "01.receipt.json", {"drySha256": files["01"]["onCameraSha256"]})

            comp = compose_project(proj)
            out = Path(comp["output"])
            self.assertEqual(comp["frames"], comp["expectedFrames"])
            self.assertTrue(Path(comp["coverPng"]).is_file())
            self.assertAlmostEqual(comp["mix"]["integratedLufs"], -14.5, delta=0.2)
            cover = _frame(out, 0.2)
            self.assertGreater(int(cover[500, 1400][1]), 110)  # 封面右侧是主讲人参考图（绿）
            self.assertLess(int(cover[500, 1400][0]), 90)
            self.assertGreater(int(cover[30, 30][0]), 220)  # 左上是纸色底
            windows = load_json(proj / "work" / "host-windows.json")
            self.assertEqual(windows[0]["startFrame"], 22)
            self.assertEqual(windows[0]["chapterOriginFrame"], 22)

            qc = run_final_qc(out, report_dir=proj / "outputs" / "qc", windows_json=proj / "work" / "host-windows.json")
            self.assertTrue(qc["passed"], qc["failures"])
            self.assertLessEqual(qc["truePeakDb"], -1.0)

            quiet = run_final_qc(out, report_dir=proj / "outputs" / "qc2", target_lufs=-23.0)
            self.assertFalse(quiet["passed"])


class FactsAndReadingTests(unittest.TestCase):
    def _manifest(self) -> dict:
        with tempfile.TemporaryDirectory() as td:
            m = load_json(init_project(Path(td) / "memt", title="MemT", use_case="U8", industry="I20"))
        m["presenter"]["portraitAuthorized"] = True
        m["brief"]["facts"] = [
            {"claim": "聊到 25 万 token 时回答一次 24 秒", "values": ["25", "24"], "source": "MemT 技术报告 v3", "variant": "Nano"},
            {"claim": "记忆固定 40 MiB", "source": "MemT 技术报告 v3", "unit": "MiB"},
            {"claim": "一万个用户 50 T 对 400 G", "source": "按报告数字推算", "derived": True},
            {"claim": "精确回忆 77%"},
        ]
        m["chapters"][0].update({
            "narration": "聊到 25 万 token，它回答一次要 24 秒。它固定 40 MB。一万个用户，一边是 50 T，一边不到 400 G。",
            "ttsText": "聊到二十五万 token，它回答一次要二十四秒。MemT 固定四十兆。一万个用户，一边是五十 T，一边不到四百 G。",
            "onCameraText": "聊到 25 万 token。",
        })
        return m

    def test_facts_variant_derived_unit_source(self) -> None:
        res = check_project_compliance(self._manifest())
        codes = [w["code"] for w in res["warnings"]]
        self.assertIn("VARIANT_NOT_NAMED", codes)
        self.assertIn("UNIT_MISMATCH", codes)
        self.assertIn("DERIVED_NOT_MARKED", codes)
        self.assertIn("FACT_SOURCE_MISSING", codes)
        self.assertIn("PRONUNCIATION_HINT", codes)
        self.assertIn("DIGITS_IN_READING", codes)  # onCameraText 里的 25
        self.assertTrue(any(h["ruleId"] == "DERIVED-ON-SCREEN" for h in res["humanReviewChecklist"]))

        m = self._manifest()
        m["brief"]["facts"][3]["source"] = "MemT 技术报告 v3"
        m["chapters"][0].update({
            "narration": "聊到 25 万 token，它最小的 Nano 版本回答一次要 24 秒。它固定 40 MiB。一万个用户算下来，一边是 50 T，一边不到 400 G。",
            "ttsText": "聊到二十五万 token，它最小的 Nano 版本回答一次要二十四秒。Mem T 固定四十兆。一万个用户算下来，一边是五十 T，一边不到四百 G。",
            "onCameraText": "聊到二十五万 token。",
        })
        res2 = check_project_compliance(m)
        self.assertEqual(res2["status"], "pass", res2["warnings"])

    def test_sentence_count_mismatch_blocks(self) -> None:
        m = self._manifest()
        m["chapters"][0]["ttsText"] = "只有一句。"
        res = check_project_compliance(m)
        self.assertEqual(res["status"], "blocked")
        self.assertIn("SENTENCE_COUNT_MISMATCH", [e["code"] for e in res["errors"]])


if __name__ == "__main__":
    unittest.main()
