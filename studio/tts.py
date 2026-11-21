"""CosyVoice3 干声生成、外部干声导入、出镜子段切分与时间轴构建。"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import wave
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .config import ffmpeg_bin
from .project import load_project, project_dirs, save_json


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def build_audio_timeline(
    project_ref: str | Path,
    *,
    chapter_gap_sec: float = 0.35,
) -> dict[str, Any]:
    """根据已有分章 WAV 汇总主干声、时间轴，并切出每章 H3 出镜干声。"""
    project_dir, manifest = load_project(project_ref)
    dirs = project_dirs(project_dir)
    sample_rate = int((manifest.get("voice") or {}).get("sampleRate", 24000))

    timeline: list[dict[str, Any]] = []
    audio_manifest: list[dict[str, Any]] = []
    concat_parts: list[Path] = []
    cursor = 0.0

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
        "chapterGapSec": chapter_gap_sec,
        "totalDuration": wav_info(master_wav)["duration"],
        "masterDryWav": str(master_wav.relative_to(project_dir)),
        "masterDrySha256": sha256_file(master_wav),
        "chapters": timeline,
        "files": audio_manifest,
    }
    save_json(project_dir / "work" / "timeline.json", result)
    return result


def run_tts(
    project_ref: str | Path,
    *,
    chapter_ids: list[str] | None = None,
    force: bool = False,
    chapter_gap_sec: float = 0.35,
) -> dict[str, Any]:
    project_dir, manifest = load_project(project_ref)
    dirs = project_dirs(project_dir)
    voice = manifest["voice"]
    selected = set(chapter_ids) if chapter_ids else None

    for ch in manifest["chapters"]:
        cid = ch["id"]
        if selected and cid not in selected:
            continue
        target = dirs["narration"] / f"{cid}.wav"
        if target.is_file() and not force:
            continue
        synthesize_wav(
            endpoint=voice["endpoint"],
            voice_id=voice["voiceId"],
            text=ch.get("ttsText") or ch["narration"],
            speed=float(voice.get("speed", 1.2)),
            output=target,
        )
    return build_audio_timeline(project_dir, chapter_gap_sec=chapter_gap_sec)
