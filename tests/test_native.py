"""H3 原生出声路线：graph 结构、分块计划、逐字对齐、验收判定与拼接（不依赖 GPU）。"""

from __future__ import annotations

import math
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from studio.config import ffmpeg_bin
from studio.h3 import build_h3_native_voice_graph, build_h3_native_voice_prompt, prepare_h3_graphs
from studio.native import (
    align_chars,
    assemble_native,
    chunk_key,
    chunk_problems,
    f0_median,
    f0_range,
    frames_for,
    heard_chars,
    plan_chunks,
    plan_project,
    run_native,
    sentence_spans,
    tail_silence_sec,
)
from studio.project import init_project, load_json, save_json, sha256_file
from studio.sync_gate import video_nframes
from studio.tts import run_tts


def _graph_refs(nodes: dict) -> set[str]:
    refs = set()
    for node in nodes.values():
        for v in node["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                refs.add(v[0])
    return refs


class NativeVoiceGraphTests(unittest.TestCase):
    def test_graph_drops_lock_nodes_and_wires_voice_ref(self) -> None:
        g = build_h3_native_voice_graph(
            remote_voice_ref="f/voice-ref.wav", remote_guide="f/guide.png", guide_is_video=False,
            prompt_text="P", frames=260, seed=11, filename_prefix="f/k00-s11", client_id="c",
        )
        n = g["prompt"]
        self.assertNotIn("2", n)
        self.assertNotIn("15", n)
        self.assertEqual(n["1"], {"class_type": "LoadAudio", "inputs": {"audio": "f/voice-ref.wav"}})
        ci = n["8"]["inputs"]
        self.assertNotIn("drive_audio", ci)
        self.assertEqual(ci["length"], 260)
        self.assertEqual(ci["audio_mode"], "native")
        self.assertFalse(ci["add_source_as_reference"])
        self.assertEqual(ci["prompt_primary_audio_ordinal"], 0)
        self.assertEqual(ci["ref_audios.ref_audio_0"], ["1", 0])
        self.assertEqual(ci["task_type"], "Ref2VA")
        self.assertEqual(n["90"]["inputs"]["images"], ["14", 0])
        self.assertEqual(n["90"]["inputs"]["audio"], ["14", 1])
        self.assertEqual(n["9"]["inputs"]["noise_seed"], 11)
        # 默认 digicam LoRA 还在；每条连线都指向存在的节点（删掉的 2/15 不再被引用）
        self.assertEqual(n["7"]["inputs"]["model"], ["30", 0])
        self.assertLessEqual(_graph_refs(n), set(n))

    def test_graph_without_ref_uses_auto_task_and_rejects_bad_frames(self) -> None:
        g = build_h3_native_voice_graph(
            remote_voice_ref=None, remote_guide="f/guide.mp4", guide_is_video=True,
            prompt_text="P", frames=107, seed=1, filename_prefix="x", client_id="c",
        )
        n = g["prompt"]
        self.assertNotIn("1", n)
        self.assertEqual(n["8"]["inputs"]["task_type"], "auto")
        self.assertNotIn("ref_audios.ref_audio_0", n["8"]["inputs"])
        self.assertEqual(n["20"]["inputs"]["image"], ["19", 0])
        self.assertLessEqual(_graph_refs(n), set(n))
        with self.assertRaisesRegex(ValueError, "17k\\+5"):
            build_h3_native_voice_graph(remote_voice_ref=None, remote_guide="g", guide_is_video=False, prompt_text="P",
                                        frames=100, seed=1, filename_prefix="x", client_id="c")

    def test_prompt_carries_dialogue_and_audio_tag_only_with_ref(self) -> None:
        kw = dict(subject="a man in a dark jacket", scene="a medium shot of <Subject 1>.", seconds=10.83,
                  text="最近，央视播出了一部电视剧。看这部剧的时候。")
        p = build_h3_native_voice_prompt(**kw, multi_sentence=True)
        self.assertIn("<d>[Chinese] 最近，央视播出了一部电视剧。看这部剧的时候。 </d>", p)
        self.assertIn("[Shot 1] 0.00-10.83 seconds", p)
        self.assertIn("From 0.50 seconds (S1) speaks Mandarin Chinese to the lens with the exact voice of <Audio 1>", p)
        self.assertIn("<Audio 1>: timbre_reference", p)
        self.assertIn("short natural pause between sentences", p)
        for section in ("subject_definitions:", "summary:", "retention_analysis:", "detailed_description:",
                        "overall_soundscape:", "non_diegetic_music:"):
            self.assertIn(section, p)
        bare = build_h3_native_voice_prompt(**kw, has_voice_ref=False)
        self.assertNotIn("<Audio 1>", bare)
        self.assertNotIn("pause between sentences", bare)
        with self.assertRaisesRegex(ValueError, "scenePrompt"):
            build_h3_native_voice_prompt(**{**kw, "scene": " "})


def _ch(cid: str, tts: str, narration: str | None = None, **kw) -> dict:
    return {"id": cid, "title": cid, "narration": narration or tts, "ttsText": tts, **kw}


class NativePlanTests(unittest.TestCase):
    def test_frames_follow_long_march_formula(self) -> None:
        # long-march-01 实际计划：50 字 260 帧、20 字 107 帧、54 字 277 帧、17 字 90 帧
        for chars, frames in ((50, 260), (20, 107), (54, 277), (17, 90), (56, 294)):
            self.assertEqual(frames_for(chars, base_sec=0.4, sec_per_char=0.2067), frames)
            self.assertEqual((frames - 5) % 17, 0)

    def test_greedy_chunks_stay_inside_chapter(self) -> None:
        a8, b8, c4 = "第一句话有八个字。", "第二句也有八个字。", "最后一句。"
        chapters = [
            _ch("01", a8 + b8, narration="第一句话有 8 个字。第二句也有 8 个字。"),
            _ch("02", "中间四字。" + c4),
            _ch("03", c4, hostMode="corner"),
        ]
        plan = plan_chunks(chapters, {"maxChars": 10})
        self.assertEqual([c["id"] for c in plan], ["k00", "k01", "k02", "k03"])
        self.assertEqual([c["chapter"] for c in plan], ["01", "01", "02", "03"])
        self.assertEqual([c["lines"] for c in plan], [[1], [2], [3, 4], [5]])
        self.assertEqual([c["sentences"] for c in plan], [[1], [2], [1, 2], [1]])
        self.assertEqual([c["chars"] for c in plan], [8, 8, 8, 4])
        self.assertEqual([c["mode"] for c in plan], ["full", "full", "corner", "corner"])
        self.assertEqual(plan[0]["display"], "第一句话有 8 个字。")
        self.assertEqual(plan[2]["text"], "中间四字。最后一句。")
        self.assertEqual(plan[0]["key"], chunk_key(plan[0]["text"], plan[0]["frames"]))
        # 改第 3 章的字不影响前面块的身份（已验收结果可以沿用）
        plan2 = plan_chunks(chapters[:2] + [_ch("03", "换了一句新的话。")], {"maxChars": 10})
        self.assertEqual([c["key"] for c in plan2[:3]], [c["key"] for c in plan[:3]])

    def test_overlong_sentence_and_mismatched_sentences_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "拆短"):
            plan_chunks([_ch("01", "长" * 80 + "。")])
        with self.assertRaisesRegex(ValueError, "一一对应"):
            plan_chunks([_ch("01", "一句。两句。", narration="只有一句。")])


class NativeAlignmentTests(unittest.TestCase):
    def test_heard_chars_splits_tokens_and_skips_markers(self) -> None:
        tr = [{"tokens": [
            {"text": "[_BEG_]", "offsets": {"from": 0, "to": 0}},
            {"text": "最近", "offsets": {"from": 240, "to": 640}},
            {"text": "，", "offsets": {"from": 640, "to": 660}},
            {"text": "央", "offsets": {"from": 700, "to": 900}},
        ]}]
        self.assertEqual(heard_chars(tr), [("最", 0.24), ("近", 0.44), ("央", 0.7)])

    def test_align_interpolates_homophones_and_stays_monotonic(self) -> None:
        heard = [("懋", 1.0), ("攻", 1.2), ("会", 1.4), ("师", 1.6)]  # 「功」听成「攻」
        chars, times, cover = align_chars("懋功会师。", heard)
        self.assertEqual(chars, list("懋功会师"))
        self.assertAlmostEqual(times[1], 1.2)
        self.assertEqual(times, sorted(times))
        self.assertAlmostEqual(cover, 0.75)
        # 漏听末尾两个字：按最后听到的字往后插值
        chars, times, cover = align_chars("四个字啊", [("四", 0.5), ("个", 0.7)])
        self.assertEqual(len(times), 4)
        self.assertGreater(times[3], times[2])
        self.assertLessEqual(times[3], 0.7 + 0.2 + 1e-9)
        with self.assertRaisesRegex(RuntimeError, "对不上"):
            align_chars("完全不同", [("其", 0.1), ("他", 0.2)])

    def test_sentence_spans_pad_and_clamp(self) -> None:
        times = [0.3, 0.5, 0.7, 1.5, 1.7]
        spans = sentence_spans([3, 2], times, duration=1.9)
        self.assertEqual(spans[0][:2], (0.25, 0.98))
        self.assertEqual(spans[1][:2], (1.45, 1.9))
        self.assertEqual(spans[1][2], [1.5, 1.7])


class NativeChecksTests(unittest.TestCase):
    def test_f0_and_tail_on_synthetic_voice(self) -> None:
        sr = 48000
        t = np.arange(int(sr * 1.0)) / sr
        voice = sum(0.3 / k * np.sin(2 * math.pi * 88.0 * k * t) for k in (1, 2, 3)).astype(np.float32)
        x = np.concatenate([np.zeros(sr // 4, np.float32), voice, np.zeros(int(sr * 0.3), np.float32)])
        self.assertAlmostEqual(f0_median(x, sr), 88.0, delta=1.0)
        self.assertAlmostEqual(tail_silence_sec(x, sr), 0.3, delta=0.011)
        self.assertEqual(f0_median(np.zeros(sr, np.float32), sr), 0.0)
        self.assertEqual(tail_silence_sec(np.zeros(sr, np.float32), sr), 0.0)

    def test_problems_and_f0_window(self) -> None:
        cfg = {"asrMin": 0.75, "tailMinSec": 0.06, "f0ToleranceHz": 9.0, "f0RangeHz": None}
        self.assertEqual(f0_range(cfg, 86.5), (77.5, 95.5))
        self.assertEqual(f0_range({**cfg, "f0RangeHz": [80, 95]}, 86.5), (80.0, 95.0))
        with self.assertRaisesRegex(ValueError, "f0RangeHz"):
            f0_range(cfg, 0.0)
        good = {"sim": 1.0, "tail": 0.3, "f0": 87.0, "syncAccepted": True}
        self.assertEqual(chunk_problems(good, cfg, (80, 95)), [])
        bad = {"sim": 0.6, "tail": 0.02, "f0": 99.0, "syncAccepted": False, "syncReason": "窗口不一致"}
        probs = chunk_problems(bad, cfg, (80, 95))
        self.assertEqual(len(probs), 4)
        self.assertIn("窗口不一致", probs[3])


def _write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def _native_project(root: Path) -> Path:
    pf = init_project(root, title="原生出声测试")
    m = load_json(pf)
    m["cover"]["enabled"] = False
    m["presenter"]["h3NativeVoice"] = {"maxChars": 10}
    m["chapters"] = [
        {**m["chapters"][0], "id": "01", "narration": "第一句话有 8 个字。第二句也有 8 个字。",
         "ttsText": "第一句话有八个字。第二句也有八个字。",
         "cuts": [{"at": 0.0, "layout": "host"}, {"atSentence": 2, "layout": "split", "source": "assets/m.png"}]},
        {**m["chapters"][0], "id": "02", "narration": "最后一句。", "ttsText": "最后一句。",
         "cuts": [{"at": 0.0, "layout": "host"}]},
    ]
    save_json(pf, m)
    return root


class NativeRouteAndAssembleTests(unittest.TestCase):
    def test_new_projects_default_to_native_voice_and_old_commands_refuse(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            proj = _native_project(Path(td) / "p")
            m = load_json(proj / "project.json")
            self.assertEqual(m["presenter"]["h3Route"], "native-voice")
            self.assertEqual(m["presenter"]["voiceRef"], "assets/voice-ref.wav")
            with self.assertRaisesRegex(ValueError, "native"):
                prepare_h3_graphs(proj)
            with self.assertRaisesRegex(RuntimeError, "native"):
                run_tts(proj)
            res = plan_project(proj)
            self.assertEqual(res["count"], 3)
            self.assertTrue((proj / "work/h3/native/plan.json").is_file())

            # 样片先行：还没有任何验收块时，批量跑直接拦下（在连 4090 之前）
            _write_wav(proj / "assets/voice-ref.wav", np.zeros(48000, np.float32), 48000)
            from PIL import Image
            Image.new("RGB", (64, 64)).save(proj / "assets/frame.png")
            boom = mock.Mock(side_effect=AssertionError("不该连 4090"))
            with mock.patch("studio.native.whisper_cli_bin", return_value="whisper-cli"), \
                    mock.patch("studio.native.whisper_model_path", return_value=Path("m.bin")), \
                    mock.patch("studio.native.f0_median", return_value=86.5):
                with self.assertRaisesRegex(RuntimeError, "样片先行"):
                    run_native(proj, endpoint_factory=boom)

            m["presenter"]["h3Route"] = "aicg3d"
            save_json(proj / "project.json", m)
            with self.assertRaisesRegex(RuntimeError, "native-voice"):
                run_native(proj, endpoint_factory=boom)
            with self.assertRaisesRegex(RuntimeError, "native-voice"):
                assemble_native(proj)

    def test_assemble_joins_chunks_on_whole_frames(self) -> None:
        sr = 48000
        with tempfile.TemporaryDirectory() as td:
            proj = _native_project(Path(td) / "p")
            plan = plan_project(proj)["chunks"]
            full = load_json(proj / "work/h3/native/plan.json")
            self.assertEqual([c["frames"] for c in full], [56, 56, 22])
            nd = proj / "work/h3/native"
            state = {}
            texts = {}
            for c in full:
                n = c["frames"]
                t = np.arange(int(n / 24 * sr)) / sr
                _write_wav(nd / f"{c['key']}.src.wav", 0.3 * np.sin(2 * math.pi * 88 * t), sr)
                subprocess.run([
                    ffmpeg_bin(), "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"testsrc=size=96x64:rate=24", "-i", str(nd / f"{c['key']}.src.wav"),
                    "-frames:v", str(n), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    str(nd / "acc" / f"{c['key']}.mp4"),
                ], check=True)
                state[c["key"]] = {"id": c["id"], "status": "ok", "tries": [{"seed": 1}]}
                texts[c["key"]] = c["text"]
            save_json(nd / "state.json", state)

            def fake_asr(wavs):
                out = {}
                for w in wavs:
                    chars = [ch for ch in texts[Path(w).name.split(".")[0]] if "一" <= ch <= "鿿"]
                    out[str(w)] = [{"tokens": [
                        {"text": ch, "offsets": {"from": 300 + 150 * i, "to": 420 + 150 * i}} for i, ch in enumerate(chars)
                    ]}]
                return out

            with mock.patch("studio.native.transcribe_wavs", side_effect=fake_asr):
                res = assemble_native(proj, measure_sync=False)
            self.assertTrue(res["ok"])

            # 第 1 章：56 帧 + 2 帧留白 + 56 帧；第 2 章是末章：22 帧 + 19 帧结尾留白
            self.assertEqual([h["frames"] for h in res["hosts"]], [114, 41])
            for cid, frames in (("01", 114), ("02", 41)):
                with wave.open(str(proj / f"work/narration/{cid}.wav")) as w:
                    self.assertEqual(w.getframerate(), sr)
                    self.assertEqual(w.getnframes(), frames * 2000)
                self.assertEqual(video_nframes(proj / f"work/h3/accepted/{cid}.mp4"), frames)
            tl = load_json(proj / "work/timeline.json")
            self.assertEqual(tl["chapterGapSec"], 0.4)
            self.assertAlmostEqual(tl["chapters"][1]["start"], 114 / 24 + 0.4, places=3)
            files = {f["id"]: f for f in tl["files"]}
            for cid in ("01", "02"):
                rec = load_json(proj / f"work/h3/accepted/{cid}.receipt.json")
                self.assertEqual(rec["drySha256"], files[cid]["onCameraSha256"])
                self.assertEqual(rec["drySha256"], sha256_file(proj / files[cid]["onCameraWav"]))
            m = load_json(proj / "project.json")
            self.assertAlmostEqual(m["chapters"][0]["onCameraSec"], 114 / 24, places=3)

            sents = load_json(proj / "work/narration/01.sentences.json")["sentences"]
            self.assertEqual([s["index"] for s in sents], [1, 2])
            self.assertAlmostEqual(sents[0]["startSec"], 0.25, places=3)
            # 第 2 句在第 2 块：块起点 58 帧
            self.assertAlmostEqual(sents[1]["startSec"], 58 / 24 + 0.25, places=3)
            self.assertAlmostEqual(sents[1]["charTimes"][0][1], 58 / 24 + 0.3, places=3)
            asr = load_json(proj / "work/asr/01.json")["transcription"]
            self.assertEqual(asr[1]["tokens"][0]["offsets"]["from"], round(58 / 24 * 1000) + 300)
            chunks = load_json(nd / "chunks.json")
            self.assertEqual([c["localA"] for c in chunks], [0.0, round(58 / 24, 4), 0.0])
            self.assertEqual(len(load_json(nd / "alignment.json")), 3)
            self.assertEqual([c["id"] for c in plan], ["k00", "k01", "k02"])

            # 有块没验收就不拼
            state[full[2]["key"]]["status"] = "failed"
            save_json(nd / "state.json", state)
            m["presenter"]["h3Route"] = "native-voice"
            with self.assertRaisesRegex(RuntimeError, "k02"):
                assemble_native(proj, measure_sync=False)


if __name__ == "__main__":
    unittest.main()
