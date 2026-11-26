"""CosyVoice3 逐句干声生成（每句多条择优）、局部重配、出镜子段切分与时间轴构建。"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import shlex
import subprocess
import wave
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np

from .config import (
    DEFAULT_COVER_SEC,
    DEFAULT_EDGE_JUMP_MAX,
    DEFAULT_HEAD_PAD_SEC,
    DEFAULT_MAX_TRUE_PEAK_DB,
    DEFAULT_SENTENCE_GAP_SEC,
    DEFAULT_SENTENCE_RMS_DB,
    DEFAULT_SPEED,
    DEFAULT_TAIL_PAD_SEC,
    DEFAULT_TAKES_PER_SENTENCE,
    ffmpeg_bin,
    whisper_cli_bin,
    whisper_model_path,
)
from .project import load_json, load_project, project_dirs, save_json, sha256_file

__all__ = ["sha256_file"]


def wav_info(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
        channels = w.getnchannels()
        sampwidth = w.getsampwidth()
    return {
        "frames": frames,
        "sampleRate": rate,
        "channels": channels,
        "sampleWidth": sampwidth,
        "duration": round(frames / rate, 4) if rate else 0.0,
    }


def _http_or_ssh_request(endpoint: str, path: str, body: bytes | None = None, timeout: int = 240) -> bytes:
    if endpoint.startswith("ssh:"):
        parts = endpoint.split(":")
        if len(parts) != 3:
            raise ValueError(f"SSH 端点格式应为 ssh:<host>:<port>，收到: {endpoint}")
        host, port = parts[1], int(parts[2])
        curl_cmd = ["curl", "-fsS", "--max-time", str(max(15, timeout - 10)), f"http://127.0.0.1:{port}{path}"]
        if body is not None:
            curl_cmd += ["-H", "Content-Type: application/json", "--data-binary", "@-"]
        return subprocess.check_output(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                "ConnectTimeout=8",
                host,
                shlex.join(curl_cmd),
            ],
            input=body,
            timeout=timeout,
        )
    headers = {"Content-Type": "application/json"} if body is not None else {}
    req = Request(endpoint.rstrip("/") + path, data=body, headers=headers)
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def synthesize_wav(
    *,
    endpoint: str,
    voice_id: str,
    text: str,
    speed: float,
    output: Path,
    timeout: int = 240,
) -> Path:
    body = json.dumps({"voiceId": voice_id, "text": text, "speed": speed}, ensure_ascii=False).encode("utf-8")
    meta = json.loads(_http_or_ssh_request(endpoint, "/tts", body=body, timeout=timeout))
    download = meta.get("download_url", "")
    if not meta.get("ok") or not download.startswith("/") or urlsplit(download).netloc:
        raise RuntimeError(f"CosyVoice3 未返回合法本地 WAV 地址: {meta}")
    tmp = output.with_suffix(".tmp")
    tmp.write_bytes(_http_or_ssh_request(endpoint, download, timeout=60))
    info = wav_info(tmp)
    if info["frames"] <= 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"CosyVoice3 生成了空音频: {output.name}")
    tmp.replace(output)
    return output


def normalize_wav(source: Path, target: Path, sample_rate: int = 24000) -> Path:
    tmp = target.with_suffix(".tmp.wav")
    subprocess.run(
        [
            ffmpeg_bin(),
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(tmp),
        ],
        check=True,
    )
    tmp.replace(target)
    return target


def slice_wav(source: Path, target: Path, duration_sec: float, sample_rate: int = 24000) -> Path:
    tmp = target.with_suffix(".tmp.wav")
    subprocess.run(
        [
            ffmpeg_bin(),
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-t",
            f"{duration_sec:.4f}",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(tmp),
        ],
        check=True,
    )
    tmp.replace(target)
    return target


def cover_lead_in_sec(manifest: dict[str, Any], fps: int = 30) -> float:
    """封面时长（整帧）。封面期间不出人声和字幕，所有章节整体后移。"""
    cover = manifest.get("cover") or {}
    if not cover.get("enabled"):
        return 0.0
    sec = float(cover.get("durationSec", DEFAULT_COVER_SEC))
    if sec <= 0:
        return 0.0
    return round(round(sec * fps) / fps, 4)


def _write_silence(path: Path, seconds: float, sample_rate: int) -> Path:
    subprocess.run(
        [
            ffmpeg_bin(), "-nostdin", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"anullsrc=r={sample_rate}:cl=mono",
            "-t", f"{seconds:.4f}", "-c:a", "pcm_s16le", str(path),
        ],
        check=True,
    )
    return path


def build_audio_timeline(
    project_ref: str | Path,
    *,
    chapter_gap_sec: float = 0.35,
    sample_rate: int | None = None,
) -> dict[str, Any]:
    """根据已有分章 WAV 汇总主干声、时间轴，并切出每章 H3 出镜干声。

    sample_rate 不给就用 voice.sampleRate（CosyVoice 24 kHz）；H3 原生出声传 48000，不降采样。"""
    project_dir, manifest = load_project(project_ref)
    dirs = project_dirs(project_dir)
    sample_rate = int(sample_rate or (manifest.get("voice") or {}).get("sampleRate", 24000))
    lead_in = cover_lead_in_sec(manifest)

    timeline: list[dict[str, Any]] = []
    audio_manifest: list[dict[str, Any]] = []
    concat_parts: list[Path] = []
    cursor = 0.0
    if lead_in > 0:
        concat_parts.append(_write_silence(dirs["narration"] / "_leadin.wav", lead_in, sample_rate))
        cursor = lead_in

    gap_wav = dirs["narration"] / "_gap.wav"
    if chapter_gap_sec > 0:
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
                f"anullsrc=r={sample_rate}:cl=mono",
                "-t",
                f"{chapter_gap_sec:.4f}",
                "-c:a",
                "pcm_s16le",
                str(gap_wav),
            ],
            check=True,
        )

    chapters = manifest["chapters"]
    for idx, ch in enumerate(chapters):
        cid = ch["id"]
        full_wav = dirs["narration"] / f"{cid}.wav"
        if not full_wav.is_file():
            raise FileNotFoundError(f"缺少第 {cid} 章干声 WAV: {full_wav}")

        info = wav_info(full_wav)
        if info["sampleRate"] != sample_rate or info["channels"] != 1:
            normalize_wav(full_wav, full_wav, sample_rate=sample_rate)
            info = wav_info(full_wav)

        dur = info["duration"]
        on_cam_sec = ch.get("onCameraSec")
        on_cam_wav: Path | None = None
        on_cam_sha: str | None = None
        on_cam_dur = 0.0
        if on_cam_sec is not None and float(on_cam_sec) > 0:
            actual_slice = min(float(on_cam_sec), dur)
            on_cam_wav = dirs["h3_voice"] / f"voice-{cid}.wav"
            slice_wav(full_wav, on_cam_wav, actual_slice, sample_rate=sample_rate)
            on_cam_dur = wav_info(on_cam_wav)["duration"]
            on_cam_sha = sha256_file(on_cam_wav)

        timeline.append({
            "id": cid,
            "title": ch.get("title", cid),
            "start": round(cursor, 4),
            "duration": dur,
            "onCameraSec": on_cam_dur,
        })
        audio_manifest.append({
            "id": cid,
            "fullWav": str(full_wav.relative_to(project_dir)),
            "fullDuration": dur,
            "fullSha256": sha256_file(full_wav),
            "onCameraWav": str(on_cam_wav.relative_to(project_dir)) if on_cam_wav else None,
            "onCameraDuration": on_cam_dur,
            "onCameraSha256": on_cam_sha,
        })

        concat_parts.append(full_wav)
        cursor += dur
        if chapter_gap_sec > 0 and idx + 1 < len(chapters):
            concat_parts.append(gap_wav)
            cursor += chapter_gap_sec

    ffconcat = dirs["narration"] / "master.ffconcat"
    ffconcat.write_text(
        "ffconcat version 1.0\n"
        + "".join(f"file '{p.name}'\n" for p in concat_parts),
        encoding="utf-8",
    )
    master_wav = project_dir / "work" / "master-dry.wav"
    subprocess.run(
        [
            ffmpeg_bin(),
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(ffconcat),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(master_wav),
        ],
        check=True,
    )

    result = {
        "sampleRate": sample_rate,
        "leadInSec": lead_in,
        "chapterGapSec": chapter_gap_sec,
        "totalDuration": wav_info(master_wav)["duration"],
        "masterDryWav": str(master_wav.relative_to(project_dir)),
        "masterDrySha256": sha256_file(master_wav),
        "chapters": timeline,
        "files": audio_manifest,
    }
    save_json(project_dir / "work" / "timeline.json", result)
    return result




# --------------------------------------------------------------------------
# 逐句配音：每句生成多条，去头尾静音、按有声段均方根对齐音量，用 ASR 回读择优。
# --------------------------------------------------------------------------

SENTENCE_RE = re.compile(r".+?(?:[。！？!?]+[”’」』）)\"']*|$)", re.S)
H3_FPS = 24


def split_sentences(text: str) -> list[str]:
    """按 。！？ 切句（引号、括号跟在句末标点后面一起归到上一句）。"""
    out: list[str] = []
    for m in SENTENCE_RE.finditer((text or "").strip()):
        seg = m.group(0).strip()
        if seg:
            out.append(seg)
    return out


def pair_sentences(chapter: dict[str, Any]) -> list[dict[str, Any]]:
    cid = chapter.get("id", "?")
    narration = chapter.get("narration", "")
    narr = split_sentences(narration)
    tts = split_sentences(chapter.get("ttsText") or narration)
    if len(narr) != len(tts):
        raise ValueError(
            f"第 {cid} 章 narration 切出 {len(narr)} 句，ttsText 切出 {len(tts)} 句。"
            "两边要按 。！？ 一一对应，逐句配音才能和字幕对上。"
        )
    return [{"index": i + 1, "text": a, "tts": b} for i, (a, b) in enumerate(zip(narr, tts))]


def load_mono(path: Path, sample_rate: int) -> np.ndarray:
    """读成 float32 单声道；采样率或位深不对时先用 ffmpeg 转。"""
    with wave.open(str(path), "rb") as w:
        ok = w.getframerate() == sample_rate and w.getsampwidth() == 2 and w.getnchannels() == 1
        raw = w.readframes(w.getnframes()) if ok else b""
    if not ok:
        conv = path.with_suffix(".conv.wav")
        normalize_wav(path, conv, sample_rate=sample_rate)
        with wave.open(str(conv), "rb") as w:
            raw = w.readframes(w.getnframes())
        conv.unlink(missing_ok=True)
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def load_int16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError(f"需要 16 bit 单声道 WAV: {path}")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").copy()


def write_int16(path: Path, samples: np.ndarray, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.wav")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(samples.astype("<i2").tobytes())
    tmp.replace(path)
    return path


def write_float(path: Path, samples: np.ndarray, sample_rate: int) -> Path:
    pcm = np.clip(np.round(samples * 32767.0), -32768, 32767).astype("<i2")
    return write_int16(path, pcm, sample_rate)


def _frame_db(x: np.ndarray, hop: int) -> np.ndarray:
    n = len(x) // hop
    if n == 0:
        return np.array([-120.0])
    frames = x[: n * hop].reshape(n, hop).astype(np.float64)
    return 20.0 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-9)


def trim_silence(x: np.ndarray, sample_rate: int, *, threshold_db: float = -45.0, keep_sec: float = 0.04) -> np.ndarray:
    """去掉头尾低于阈值的静音，两头各留 keep_sec，避免切掉轻辅音。"""
    hop = max(1, int(sample_rate * 0.01))
    db = _frame_db(x, hop)
    loud = np.where(db > threshold_db)[0]
    if len(loud) == 0:
        return x
    keep = int(round(keep_sec * sample_rate))
    a = max(0, int(loud[0]) * hop - keep)
    b = min(len(x), (int(loud[-1]) + 1) * hop + keep)
    return x[a:b]


def trim_lead_ambience(
    x: np.ndarray, sample_rate: int, *, keep_sec: float = 0.15, min_lead_sec: float = 0.25, onset_below_peak_db: float = 18.0,
    fade_sec: float = 0.03,
) -> tuple[np.ndarray, float]:
    """去掉开口前的「垫底噪」：零样本模型常在第一个字前先生成一段 -30~-40 dB 的环境声/气声（参考音频带底噪时最明显），
    它高于 trim_silence 的阈值所以会留下，拼进成片就是每句开头一段杂音。

    开口 = 第一个比最响帧低不到 onset_below_peak_db 的 10 ms 帧。开口前超过 min_lead_sec 时，只保留开口前 keep_sec
    （足够放下 s、f、h 这类轻辅音），并在起点做 fade_sec 淡入。返回 (新波形, 去掉的秒数)。
    """
    hop = max(1, int(sample_rate * 0.01))
    db = _frame_db(x, hop)
    if len(db) < 3:
        return x, 0.0
    onset = int(np.argmax(db >= float(db.max()) - onset_below_peak_db))
    lead = onset * hop / sample_rate
    if lead <= min_lead_sec:
        return x, 0.0
    a = max(0, onset * hop - int(round(keep_sec * sample_rate)))
    y = x[a:].astype(np.float32).copy()
    f = min(len(y), int(round(fade_sec * sample_rate)))
    y[:f] *= np.linspace(0.0, 1.0, f, dtype=np.float32)
    return y, round(a / sample_rate, 3)


def find_head_residue(x: np.ndarray, sample_rate: int) -> float | None:
    """找句首残音：CosyVoice 零样本会把参考音频的最后一点声音接在句首（第 0 个样本就在发声），
    后面跟一个能量谷才进真正的第一个字。返回谷起点（秒）；没有这种形状返回 None。

    只认「前面一团 >= -26 dB 的声音 → 掉下去至少 8 dB 且 <= -27 dB 的谷 → 200 ms 内回到 > -22 dB」，
    只在开头 220 ms 内找。返回谷的起点而不是谷底：谷里可能是 s、sh、f 这类轻辅音。
    """
    hop = max(1, int(sample_rate * 0.01))
    e = _frame_db(x[: hop * 30], hop)
    if len(e) < 30:
        return None
    for v in range(2, 22):
        head = float(e[:v].max())
        if head < -26.0 or e[v] > min(-27.0, head - 8.0):
            continue
        end = v
        while end + 1 < 30 and e[end + 1] <= min(-25.0, head - 6.0):
            end += 1
        if end + 1 < 30 and float(e[end + 1 : end + 21].max()) > -22.0:
            return v * hop / sample_rate
        return None
    return None


def edge_fade(x: np.ndarray, sample_rate: int, *, head_sec: float = 0.004, tail_sec: float = 0.004) -> np.ndarray:
    """头尾各做一个很短的线性淡入淡出，消掉从数字静音直接跳进声音的爆音。不改变长度。"""
    y = x.astype(np.float32).copy()
    h = min(len(y), int(round(head_sec * sample_rate)))
    t = min(len(y), int(round(tail_sec * sample_rate)))
    if h > 0:
        y[:h] *= np.linspace(0.0, 1.0, h, dtype=np.float32)
    if t > 0:
        y[len(y) - t :] *= np.linspace(1.0, 0.0, t, dtype=np.float32)
    return y


def strip_head_residue(x: np.ndarray, sample_rate: int, *, fade_sec: float = 0.015) -> tuple[np.ndarray, float | None]:
    """把句首残音静音（谷起点之前置零，谷起点处 fade_sec 淡入），再做头尾短淡入淡出。长度不变。"""
    at = find_head_residue(x, sample_rate)
    y = x.astype(np.float32).copy()
    if at is not None:
        a = int(round(at * sample_rate))
        y[:a] = 0.0
        f = min(len(y) - a, int(round(fade_sec * sample_rate)))
        y[a : a + f] *= np.linspace(0.0, 1.0, f, dtype=np.float32)
    return edge_fade(y, sample_rate), at


def edge_jump(x: np.ndarray, sample_rate: int, *, window_sec: float = 0.002) -> float:
    """拼进数字静音时，句首/句尾 window_sec 内的最大样本跳变（含从 0 跳到第一个样本）。"""
    if len(x) == 0:
        return 0.0
    n = max(1, int(round(window_sec * sample_rate)))
    head = np.concatenate([[0.0], x[:n].astype(np.float64)])
    tail = np.concatenate([x[-n:].astype(np.float64), [0.0]])
    return round(float(max(np.abs(np.diff(head)).max(), np.abs(np.diff(tail)).max())), 5)


def _first_char(text: str) -> str:
    from .subtitles import normalize_text

    t = normalize_text(text or "")
    return t[:1]


def voiced_rms_db(x: np.ndarray, sample_rate: int) -> float:
    """有声段均方根：只统计比最响帧低 30 dB 以内、且高于 -50 dBFS 的 20 ms 帧。"""
    hop = max(1, int(sample_rate * 0.02))
    n = len(x) // hop
    if n == 0:
        return -120.0
    frames = x[: n * hop].reshape(n, hop).astype(np.float64)
    power = np.mean(frames ** 2, axis=1)
    db = 10.0 * np.log10(power + 1e-12)
    voiced = power[db > max(-50.0, float(db.max()) - 30.0)]
    if len(voiced) == 0:
        return -120.0
    return float(10.0 * np.log10(np.mean(voiced) + 1e-12))


def _alimiter(x: np.ndarray, sample_rate: int, limit: float) -> np.ndarray:
    proc = subprocess.run(
        [
            ffmpeg_bin(), "-nostdin", "-v", "error",
            "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
            "-af", f"alimiter=limit={max(0.0625, min(1.0, limit)):.5f}:attack=5:release=50:level=0:latency=1",
            "-f", "f32le", "pipe:1",
        ],
        input=x.astype("<f4").tobytes(),
        capture_output=True,
        check=True,
    )
    y = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)
    if len(y) >= len(x):
        return y[: len(x)]
    return np.concatenate([y, np.zeros(len(x) - len(y), dtype=np.float32)])


def level_sentence(
    x: np.ndarray,
    sample_rate: int,
    *,
    target_db: float = DEFAULT_SENTENCE_RMS_DB,
    peak_db: float = DEFAULT_MAX_TRUE_PEAK_DB,
    iterations: int = 4,
) -> np.ndarray:
    """按有声段均方根对齐到固定目标，超峰再过 alimiter，循环几次直到两项都稳定。

    目标是固定值而不是相邻句的平均值，所以重配一句不会改动其他句子的采样。
    """
    limit = 10.0 ** (peak_db / 20.0)
    y = x.astype(np.float32)
    for _ in range(iterations):
        cur = voiced_rms_db(y, sample_rate)
        if cur <= -119.0:
            break
        y = y * np.float32(10.0 ** ((target_db - cur) / 20.0))
        if float(np.max(np.abs(y))) > limit:
            y = _alimiter(y, sample_rate, limit)
        if abs(voiced_rms_db(y, sample_rate) - target_db) < 0.3 and float(np.max(np.abs(y))) <= limit * 1.001:
            break
    return np.clip(y, -limit, limit)


def _sentence_key(voice_id: str, speed: float, tts_text: str, sample_rate: int) -> str:
    raw = f"{voice_id}|{speed:.3f}|{sample_rate}|{tts_text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def transcribe_wavs(wavs: list[Path], *, language: str = "zh", prompt: str = "以下是普通话的句子。") -> dict[str, Any] | None:
    """用本地 whisper.cpp 批量回读。没装 whisper-cli 或模型时返回 None（调用方必须如实报告）。"""
    cli = whisper_cli_bin()
    model = whisper_model_path()
    if not cli or not model:
        return None
    results: dict[str, Any] = {}
    for i in range(0, len(wavs), 24):
        batch = wavs[i : i + 24]
        converted: list[Path] = []
        for w in batch:
            c = w.with_suffix(".asr16k.wav")
            subprocess.run(
                [ffmpeg_bin(), "-nostdin", "-v", "error", "-y", "-i", str(w), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(c)],
                check=True,
            )
            converted.append(c)
        cmd = [cli, "-m", str(model), "-l", language, "-ojf", "-np", "--prompt", prompt]
        for c in converted:
            cmd += ["-f", str(c)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for w, c in zip(batch, converted):
            out = Path(str(c) + ".json")
            if out.is_file():
                results[str(w)] = load_json(out).get("transcription") or []
                out.unlink(missing_ok=True)
            c.unlink(missing_ok=True)
    return results


def transcription_text(transcription: list[dict[str, Any]]) -> str:
    return "".join(str(seg.get("text", "")) for seg in transcription).strip()


def asr_similarity(asr_text: str, *refs: str) -> float:
    from .subtitles import normalize_text

    a = normalize_text(asr_text)
    best = 0.0
    for ref in refs:
        b = normalize_text(ref)
        if a and b:
            best = max(best, difflib.SequenceMatcher(None, a, b, autojunk=False).ratio())
    return round(best, 4)


def choose_take(takes: list[dict[str, Any]], *, margin: float = 0.03, edge_jump_max: float = DEFAULT_EDGE_JUMP_MAX) -> int:
    """先排除句首/句尾有爆音（edgeJump 超限）的条目，再在回读最干净的几条里挑最短的一条；没有 ASR 时只能挑最短。
    全部超限时不排除（调用方在报告里如实标出）。"""
    clean = [t for t in takes if t.get("edgeJump") is None or float(t["edgeJump"]) <= edge_jump_max]
    takes = clean or takes
    scored = [t for t in takes if t.get("asrScore") is not None]
    if not scored:
        return int(min(takes, key=lambda t: t["durationSec"])["take"])
    best = max(float(t["asrScore"]) for t in scored)
    near = [t for t in scored if float(t["asrScore"]) >= best - margin]
    return int(min(near, key=lambda t: t["durationSec"])["take"])


def _shift_transcription(transcription: list[dict[str, Any]], offset_ms: int) -> list[dict[str, Any]]:
    shifted: list[dict[str, Any]] = []
    for seg in transcription:
        seg2 = dict(seg)
        if isinstance(seg.get("offsets"), dict):
            seg2["offsets"] = {k: int(v) + offset_ms for k, v in seg["offsets"].items()}
        toks = []
        for tok in seg.get("tokens") or []:
            tok2 = dict(tok)
            if isinstance(tok.get("offsets"), dict):
                tok2["offsets"] = {k: int(v) + offset_ms for k, v in tok["offsets"].items()}
            tok2.pop("timestamps", None)
            toks.append(tok2)
        seg2["tokens"] = toks
        seg2.pop("timestamps", None)
        shifted.append(seg2)
    return shifted


def h3_output_sec(requested_sec: float) -> float:
    """H3 输出帧数只有 17k+5 档（24 fps），按请求帧数就近取整。"""
    frames = requested_sec * H3_FPS
    k = max(0, round((frames - 5) / 17))
    return (17 * k + 5) / H3_FPS


def suggest_on_camera(
    sentences: list[dict[str, Any]],
    *,
    tail_pad_sec: float = DEFAULT_TAIL_PAD_SEC,
    min_sec: float = 5.0,
    max_sec: float = 10.0,
) -> dict[str, Any]:
    """给出出镜取前 N 句时的 onCameraSec：切点落在句间停顿里，并保证 H3 取整后不截断人声。"""
    options: list[dict[str, Any]] = []
    for n in range(1, len(sentences) + 1):
        end = float(sentences[n - 1]["endSec"])
        if end > max_sec + 0.5:
            break
        nxt = float(sentences[n]["startSec"]) if n < len(sentences) else end + tail_pad_sec
        lo, hi = end + 0.08, min(nxt - 0.03, end + 0.45)
        picks = []
        s = lo
        while s <= hi + 1e-9:
            out = h3_output_sec(s)
            if out >= end + 0.05:
                picks.append((0 if out >= s - 1e-9 else 1, abs(out - s), round(s, 2), out))
            s += 0.01
        if not picks:
            options.append({"sentences": n, "speechEndSec": round(end, 3), "ok": False,
                            "reason": "句间停顿里找不到 H3 取整后不截断人声的切点"})
            continue
        picks.sort()
        _, _, s, out = picks[0]
        options.append({
            "sentences": n,
            "speechEndSec": round(end, 3),
            "onCameraSec": s,
            "h3OutputSec": round(out, 3),
            "firstCutAtSec": round(min(s, out), 2),
            "ok": True,
        })
    good = [o for o in options if o["ok"] and o["speechEndSec"] <= max_sec]
    in_range = [o for o in good if o["speechEndSec"] >= min_sec]
    rec = (in_range or good or [None])[-1] if (in_range or good) else None
    return {"options": options, "recommended": rec}


def dub_chapter_by_sentence(
    project_dir: Path,
    manifest: dict[str, Any],
    chapter: dict[str, Any],
    *,
    force: bool = False,
    reroll: set[int] | None = None,
    takes: int | None = None,
) -> dict[str, Any]:
    """逐句配一章：文字没变的句子沿用上次选中的那条，变了的（或指定重摇的）每句重新生成多条择优。"""
    dirs = project_dirs(project_dir)
    voice = manifest["voice"]
    cid = chapter["id"]
    sr = int(voice.get("sampleRate", 24000))
    speed = float(voice.get("speed", DEFAULT_SPEED))
    voice_id = str(voice["voiceId"])
    n_takes = max(1, int(takes or voice.get("takesPerSentence", DEFAULT_TAKES_PER_SENTENCE)))
    gap = float(voice.get("sentenceGapSec", DEFAULT_SENTENCE_GAP_SEC))
    head = float(voice.get("headPadSec", DEFAULT_HEAD_PAD_SEC))
    tail = float(voice.get("tailPadSec", DEFAULT_TAIL_PAD_SEC))
    target_rms = float(voice.get("sentenceRmsDb", DEFAULT_SENTENCE_RMS_DB))
    reroll = set(reroll or ())

    pairs = pair_sentences(chapter)
    sent_dir = dirs["narration"] / "sentences" / cid
    sent_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dirs["narration"] / f"{cid}.sentences.json"
    old = load_json(manifest_path) if manifest_path.is_file() else None
    old_sentences = (old or {}).get("sentences", [])
    pool: dict[str, list[dict[str, Any]]] = {}
    if old and not force:
        for s in old_sentences:
            pool.setdefault(s["key"], []).append(s)
    max_roll: dict[str, int] = {}
    for s in old_sentences:
        max_roll[s["key"]] = max(max_roll.get(s["key"], 0), int(s.get("roll", 1)))

    entries: list[dict[str, Any] | None] = []
    pending: list[tuple[int, dict[str, Any], str, int]] = []
    reused: list[int] = []
    for pair in pairs:
        key = _sentence_key(voice_id, speed, pair["tts"], sr)
        prev = None
        if pair["index"] not in reroll and pool.get(key):
            prev = pool[key].pop(0)
            if not (project_dir / prev["wav"]).is_file():
                prev = None
        if prev is not None:
            e = dict(prev)
            e.update({"index": pair["index"], "text": pair["text"], "tts": pair["tts"]})
            entries.append(e)
            reused.append(pair["index"])
        else:
            roll = max_roll.get(key, 0) + 1
            max_roll[key] = roll
            entries.append(None)
            pending.append((len(entries) - 1, pair, key, roll))

    generated_takes: list[list[dict[str, Any]]] = []
    for _slot, pair, key, roll in pending:
        infos = []
        for k in range(1, n_takes + 1):
            raw = sent_dir / f"{key}-r{roll}-t{k}.raw.wav"
            final = sent_dir / f"{key}-r{roll}-t{k}.wav"
            synthesize_wav(
                endpoint=voice["endpoint"], voice_id=voice_id, text=pair["tts"], speed=speed, output=raw,
            )
            x0, lead_cut = trim_lead_ambience(trim_silence(load_mono(raw, sr), sr), sr)
            stripped, residue = strip_head_residue(x0, sr)
            x = edge_fade(level_sentence(stripped, sr, target_db=target_rms), sr)
            write_float(final, x, sr)
            info = {"take": k, "wav": str(final.relative_to(project_dir)), "durationSec": round(len(x) / sr, 4),
                    "headResidueSec": None if residue is None else round(residue, 3), "edgeJump": edge_jump(x, sr),
                    "leadAmbienceCutSec": lead_cut}
            if residue is not None:
                # 句首残音静音后可能吃掉真的第一个字，留一条只淡入的备选，回读后再定
                keep = sent_dir / f"{key}-r{roll}-t{k}.keep.wav"
                xk = edge_fade(level_sentence(edge_fade(x0, sr), sr, target_db=target_rms), sr)
                write_float(keep, xk, sr)
                info["_keep"] = (str(keep.relative_to(project_dir)), round(len(xk) / sr, 4), edge_jump(xk, sr))
            infos.append(info)
        generated_takes.append(infos)

    new_wavs = [project_dir / t["wav"] for infos in generated_takes for t in infos]
    new_wavs += [project_dir / t["_keep"][0] for infos in generated_takes for t in infos if "_keep" in t]
    asr = transcribe_wavs(new_wavs) if new_wavs else {}
    asr_available = asr is not None
    generated: list[int] = []
    for (slot, pair, key, roll), infos in zip(pending, generated_takes):
        trs: dict[int, Any] = {}
        ref_first = {_first_char(pair["text"]), _first_char(pair["tts"])}
        for t in infos:
            tr = (asr or {}).get(str(project_dir / t["wav"]))
            keep = t.pop("_keep", None)
            if keep is not None and asr is not None:
                tr_k = asr.get(str(project_dir / keep[0]))
                if tr is not None and tr_k is not None:
                    txt_s, txt_k = transcription_text(tr), transcription_text(tr_k)
                    sc_s = asr_similarity(txt_s, pair["text"], pair["tts"])
                    sc_k = asr_similarity(txt_k, pair["text"], pair["tts"])
                    lost_first = _first_char(txt_k) in ref_first and _first_char(txt_s) != _first_char(txt_k)
                    if lost_first or sc_s < sc_k - 0.005:
                        t.update({"wav": keep[0], "durationSec": keep[1], "edgeJump": keep[2]})
                        tr = tr_k
                        t["headResidueStripped"] = False
                    else:
                        t["headResidueStripped"] = True
            if tr is None:
                t["asrText"], t["asrScore"] = None, None
            else:
                t["asrText"] = transcription_text(tr)
                t["asrScore"] = asr_similarity(t["asrText"], pair["text"], pair["tts"])
                trs[t["take"]] = tr
        chosen = choose_take(infos)
        ct = next(t for t in infos if t["take"] == chosen)
        entries[slot] = {
            "index": pair["index"],
            "text": pair["text"],
            "tts": pair["tts"],
            "key": key,
            "roll": roll,
            "takes": infos,
            "chosenTake": chosen,
            "wav": ct["wav"],
            "durationSec": ct["durationSec"],
            "asrText": ct["asrText"],
            "asrScore": ct["asrScore"],
            "asrTranscription": trs.get(chosen),
            "edgeJump": ct.get("edgeJump"),
            "headResidueSec": ct.get("headResidueSec"),
            "headResidueStripped": ct.get("headResidueStripped"),
        }
        generated.append(pair["index"])

    head_n, gap_n, tail_n = (int(round(v * sr)) for v in (head, gap, tail))
    parts: list[np.ndarray] = [np.zeros(head_n, dtype="<i2")]
    cursor = head_n
    final_entries: list[dict[str, Any]] = [e for e in entries if e is not None]
    for i, e in enumerate(final_entries):
        pcm = load_int16(project_dir / e["wav"])
        e["startSec"] = round(cursor / sr, 4)
        cursor += len(pcm)
        e["endSec"] = round(cursor / sr, 4)
        e["sha256"] = sha256_file(project_dir / e["wav"])
        parts.append(pcm)
        if i + 1 < len(final_entries):
            parts.append(np.zeros(gap_n, dtype="<i2"))
            cursor += gap_n
    parts.append(np.zeros(tail_n, dtype="<i2"))
    chapter_wav = dirs["narration"] / f"{cid}.wav"
    write_int16(chapter_wav, np.concatenate(parts), sr)

    transcription: list[dict[str, Any]] = []
    for e in final_entries:
        if e.get("asrTranscription"):
            transcription += _shift_transcription(e["asrTranscription"], int(round(e["startSec"] * 1000)))
    if transcription:
        save_json(dirs["asr"] / f"{cid}.json", {"source": "video-studio sentence ASR (whisper.cpp)", "transcription": transcription})

    save_json(manifest_path, {
        "chapterId": cid,
        "voiceId": voice_id,
        "speed": speed,
        "sampleRate": sr,
        "headPadSec": head,
        "sentenceGapSec": gap,
        "tailPadSec": tail,
        "sentenceRmsDb": target_rms,
        "durationSec": round(cursor / sr + tail, 4),
        "sentences": final_entries,
    })

    on_cam = suggest_on_camera(final_entries, tail_pad_sec=tail)
    applied = None
    want = chapter.get("onCameraSentences")
    if want:
        opt = next((o for o in on_cam["options"] if o["sentences"] == int(want) and o["ok"]), None)
        if opt:
            applied = opt
    low = [
        {"index": e["index"], "asrScore": e["asrScore"], "text": e["text"], "asr": e["asrText"]}
        for e in final_entries
        if e.get("asrScore") is not None and float(e["asrScore"]) < 0.85
    ]
    edge_bad = [
        {"index": e["index"], "edgeJump": e["edgeJump"], "text": e["text"]}
        for e in final_entries
        if e.get("edgeJump") is not None and float(e["edgeJump"]) > DEFAULT_EDGE_JUMP_MAX
    ]
    return {
        "id": cid,
        "mode": "sentence",
        "edgeJumpMax": DEFAULT_EDGE_JUMP_MAX,
        "edgeJumpFailures": edge_bad,
        "headResidueStripped": [e["index"] for e in final_entries if e.get("headResidueStripped")],
        "sentences": len(final_entries),
        "generated": generated,
        "reused": reused,
        "takesPerSentence": n_takes,
        "asrAvailable": asr_available if new_wavs else None,
        "lowAsrSentences": low,
        "boundaries": [
            {"index": e["index"], "startSec": e["startSec"], "endSec": e["endSec"], "asrScore": e.get("asrScore"), "text": e["text"]}
            for e in final_entries
        ],
        "onCamera": on_cam,
        "onCameraApplied": applied,
    }


def run_tts(
    project_ref: str | Path,
    *,
    chapter_ids: list[str] | None = None,
    force: bool = False,
    chapter_gap_sec: float = 0.35,
    reroll: dict[str, set[int]] | None = None,
    takes: int | None = None,
) -> dict[str, Any]:
    """生成干声。默认逐句配音；voice.dubbing = "chapter" 时退回整章一次生成的旧做法。"""
    project_dir, manifest = load_project(project_ref)
    if (manifest.get("presenter") or {}).get("h3Route") == "native-voice":
        raise RuntimeError(
            "这个项目走 H3 原生出声（presenter.h3Route = native-voice），人声由 `video-studio native` 生成，不用 tts/redub。"
            "要改用 CosyVoice 配音加锁音，把 presenter.h3Route 改成 aicg3d 或 native-lock。"
        )
    dirs = project_dirs(project_dir)
    voice = manifest["voice"]
    mode = voice.get("dubbing", "sentence")
    selected = set(chapter_ids) if chapter_ids else None
    reports: list[dict[str, Any]] = []
    project_changed = False

    for ch in manifest["chapters"]:
        cid = ch["id"]
        if selected and cid not in selected:
            continue
        target = dirs["narration"] / f"{cid}.wav"
        if mode == "chapter":
            if target.is_file() and not force:
                reports.append({"id": cid, "mode": "chapter", "skipped": True})
                continue
            synthesize_wav(
                endpoint=voice["endpoint"],
                voice_id=voice["voiceId"],
                text=ch.get("ttsText") or ch["narration"],
                speed=float(voice.get("speed", DEFAULT_SPEED)),
                output=target,
            )
            reports.append({"id": cid, "mode": "chapter"})
            continue
        if target.is_file() and not (dirs["narration"] / f"{cid}.sentences.json").is_file() and not force:
            reports.append({
                "id": cid,
                "mode": "legacy-chapter",
                "skipped": True,
                "note": "这一章是旧的整章配音，没有动。要改成逐句配音，加 --force 重配。",
            })
            continue
        rep = dub_chapter_by_sentence(
            project_dir, manifest, ch, force=force, reroll=(reroll or {}).get(cid), takes=takes,
        )
        if rep.get("onCameraApplied") and float(ch.get("onCameraSec") or 0) != rep["onCameraApplied"]["onCameraSec"]:
            ch["onCameraSec"] = rep["onCameraApplied"]["onCameraSec"]
            project_changed = True
        reports.append(rep)

    if project_changed:
        save_json(project_dir / "project.json", manifest)
    timeline = build_audio_timeline(project_dir, chapter_gap_sec=chapter_gap_sec)
    return {"chapters": reports, "timeline": timeline}


def redub(
    project_ref: str | Path,
    *,
    chapter_id: str,
    sentences: list[int] | None = None,
    takes: int | None = None,
) -> dict[str, Any]:
    """局部重配：只重配改过字的句子（或 --sentence 指定的句子），其余句子的采样原样沿用。

    报告哪些章的出镜段音频没变（已验收的 H3 可以直接沿用），哪些变了（必须重新生成 H3）。
    """
    project_dir, manifest = load_project(project_ref)
    dirs = project_dirs(project_dir)
    ch = next((c for c in manifest["chapters"] if c["id"] == chapter_id), None)
    if ch is None:
        raise KeyError(f"项目里没有第 {chapter_id} 章")
    sent_path = dirs["narration"] / f"{chapter_id}.sentences.json"
    if not sent_path.is_file():
        raise FileNotFoundError(
            f"第 {chapter_id} 章没有逐句配音记录（{sent_path.name}），不能局部重配。"
            f"先用 tts --chapters {chapter_id} --force 逐句配一遍。"
        )
    old_sent = load_json(sent_path)
    tl_path = project_dir / "work" / "timeline.json"
    old_tl = load_json(tl_path) if tl_path.is_file() else None
    gap = float(old_tl.get("chapterGapSec", 0.35)) if old_tl else 0.35
    old_files = {f["id"]: f for f in (old_tl or {}).get("files", [])}
    old_by_index = {s["index"]: s for s in old_sent["sentences"]}

    res = run_tts(
        project_dir,
        chapter_ids=[chapter_id],
        chapter_gap_sec=gap,
        reroll={chapter_id: set(sentences or [])},
        takes=takes,
    )
    rep = res["chapters"][0]
    new_sent = load_json(sent_path)
    new_files = {f["id"]: f for f in res["timeline"]["files"]}

    changed = []
    for e in new_sent["sentences"]:
        if e["index"] not in rep["generated"]:
            continue
        before = old_by_index.get(e["index"])
        changed.append({
            "index": e["index"],
            "before": before["text"] if before else None,
            "after": e["text"],
            "beforeDurationSec": before["durationSec"] if before else None,
            "afterDurationSec": e["durationSec"],
            "startSec": e["startSec"],
            "asrScore": e.get("asrScore"),
            "chosenTake": e["chosenTake"],
        })

    host_impact = []
    for cid, nf in new_files.items():
        if not nf.get("onCameraSha256"):
            continue
        of = old_files.get(cid)
        same = bool(of) and of.get("onCameraSha256") == nf["onCameraSha256"]
        host_impact.append({
            "chapterId": cid,
            "onCameraAudioUnchanged": same,
            "onCameraSha256": nf["onCameraSha256"],
            "next": "沿用已验收的 H3 片段" if same else "出镜段音频变了：重新 h3-prepare → h3-submit → sync-gate",
        })

    first_start = min((c["startSec"] for c in changed), default=None)
    stale_cuts = []
    if first_start is not None:
        for k, cut in enumerate(ch.get("cuts") or []):
            at = cut.get("at")
            if "atSentence" in cut or not isinstance(at, (int, float)):
                continue
            if float(at) >= first_start - 1e-6 and k > 0:
                stale_cuts.append({"cutIndex": k, "at": at, "source": cut.get("source")})

    return {
        "chapterId": chapter_id,
        "changedSentences": changed,
        "reusedSentences": rep["reused"],
        "chapterDurationBeforeSec": old_sent.get("durationSec"),
        "chapterDurationAfterSec": new_sent.get("durationSec"),
        "asrAvailable": rep.get("asrAvailable"),
        "lowAsrSentences": rep.get("lowAsrSentences"),
        "edgeJumpFailures": rep.get("edgeJumpFailures"),
        "headResidueStripped": rep.get("headResidueStripped"),
        "hostSegments": host_impact,
        "cutsWithFixedSeconds": stale_cuts,
        "note": (
            "固定秒数的切点不会跟着句子时长移动，建议改成 atSentence。"
            if stale_cuts else "切点都会跟着句子走。"
        ),
        "nextSteps": ["subtitles", "compose", "qc --windows work/host-windows.json"],
        "timeline": res["timeline"],
    }
