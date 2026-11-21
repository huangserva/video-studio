"""H3 数字人口型与手势硬门禁：波形校验 -> SyncNet 逐窗 -> 24fps 整帧移画 -> handcheck -> 网格图。"""

from __future__ import annotations

import json
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .config import TOOLS_DIR, ffmpeg_bin, ffprobe_bin, handcheck_bin, syncnet_model_path, vision_python
from .project import load_project, load_recipe, project_dirs, save_json


def _run(cmd: list[str], timeout: int = 300) -> bytes:
    return subprocess.check_output(cmd, timeout=timeout)


def video_nframes(path: Path) -> int:
    out = _run([
        ffprobe_bin(),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "csv=p=0",
        str(path),
    ]).decode().strip()
    return int(out)


def audio_stream_sha256(path: Path) -> str:
    return _run([
        ffmpeg_bin(),
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-c:a",
        "copy",
        "-f",
        "hash",
        "-hash",
        "sha256",
        "-",
    ]).decode().strip()


def decode_mono_pcm(path: Path, sample_rate: int = 24000) -> np.ndarray:
    raw = _run([
        ffmpeg_bin(),
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-f",
        "s16le",
        "-",
    ])
    return np.frombuffer(raw, dtype="<i2").astype(np.float64)


def check_waveform_match(
    dry_wav: Path,
    candidate_mp4: Path,
    *,
    corr_min: float = 0.975,
    trailing_peak_max: float = 200.0,
) -> dict[str, Any]:
    a = decode_mono_pcm(dry_wav)
    b = decode_mono_pcm(candidate_mp4)
    n = min(len(a), len(b))
    if n < 2400:
        raise ValueError(f"音频过短（仅 {n} 采样），无法计算波形相关度")
    corr = float(np.corrcoef(a[:n], b[:n])[0, 1])
    trailing_peak = float(np.max(np.abs(a[n:]), initial=0.0))
    ok = corr >= corr_min and trailing_peak < trailing_peak_max
    return {
        "ok": ok,
        "correlation": round(corr, 6),
        "corrMin": corr_min,
        "trailingPeak": round(trailing_peak, 2),
        "trailingPeakMax": trailing_peak_max,
        "drySamples": len(a),
        "candidateSamples": len(b),
    }


def measure_syncnet(
    video: Path,
    *,
    seek: float = 0.0,
    duration: float | None = None,
    crop: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    cmd = [
        vision_python(),
        str(TOOLS_DIR / "syncnet_check.py"),
        str(video),
        "--model",
        str(syncnet_model_path()),
    ]
    if seek > 0:
        cmd += ["--seek", f"{seek:.4f}"]
    if duration is not None:
        cmd += ["--duration", f"{duration:.4f}"]
    if crop is not None:
        cmd += ["--crop", ",".join(str(v) for v in crop)]
    raw = _run(cmd, timeout=300).decode("utf-8").strip()
    return json.loads(raw.splitlines()[-1])


def run_handcheck(video: Path) -> dict[str, Any]:
    binary = handcheck_bin()
    if not binary.is_file():
        return {"available": False, "reason": f"缺少 {binary}"}
    raw = _run([str(binary), str(video)], timeout=180).decode("utf-8").strip()
    frames = json.loads(raw)
    return {
        "available": True,
        "samples": len(frames),
        "oneOrMoreHands": sum(1 for f in frames if f.get("hands")),
        "bothHands": sum(1 for f in frames if len(f.get("hands", [])) == 2),
    }


def _video_dimensions(path: Path) -> tuple[int, int]:
    raw = _run([
        ffprobe_bin(),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0",
        str(path),
    ]).decode().strip()
    w, h = raw.split(",")
    return int(w), int(h)


def make_visual_grid(video: Path, output_jpg: Path, *, count: int = 12, fps: float = 24.0) -> Path:
    n = video_nframes(video)
    vw, vh = _video_dimensions(video)
    duration = max(0.5, n / fps)
    times = np.linspace(min(0.15, duration * 0.1), max(0.2, duration - 0.2), count)
    cols = 4
    rows = (count + cols - 1) // cols
    if vw > vh:
        cell_w = 320
        cell_h = max(120, round(cell_w * vh / vw))
    else:
        cell_h = 240
        cell_w = max(120, round(cell_h * vw / vh))
    label_h = 20
    canvas = Image.new("RGB", (cols * cell_w, rows * (cell_h + label_h)), "#202020")
    draw = ImageDraw.Draw(canvas)
    for i, t in enumerate(times):
        png = _run([
            ffmpeg_bin(),
            "-nostdin",
            "-v",
            "error",
            "-ss",
            f"{t:.3f}",
            "-i",
            str(video),
            "-vf",
            f"scale={cell_w}:{cell_h}:force_original_aspect_ratio=decrease,"
            f"pad={cell_w}:{cell_h}:(ow-iw)/2:(oh-ih)/2:color=#202020",
            "-frames:v",
            "1",
            "-c:v",
            "png",
            "-f",
            "image2pipe",
            "-",
        ])
        x = (i % cols) * cell_w
        y = (i // cols) * (cell_h + label_h)
        canvas.paste(Image.open(BytesIO(png)), (x, y))
        draw.text((x + 6, y + cell_h + 3), f"{t:.2f}s", fill="white")
    output_jpg.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_jpg, quality=90)
    return output_jpg


def retime_video_keep_audio(
    *,
    video_source: Path,
    audio_source: Path,
    output: Path,
    target_frames: int,
    advance_frames: int,
    ai_label: str = "AI 生成",
    font_path: str = "/System/Library/Fonts/PingFang.ttc",
) -> Path:
    start = max(0, advance_frames)
    delay = max(0, -advance_frames)
    vf_parts = [
        f"trim=start_frame={start}",
        "setpts=PTS-STARTPTS",
        f"tpad=start_mode=clone:start={delay}:stop_mode=clone:stop=24",
        f"trim=end_frame={target_frames}",
    ]
    if ai_label:
        escaped = ai_label.replace(":", r"\:").replace("'", r"\'")
        vf_parts.append(
            f"drawtext=fontfile='{font_path}':text='{escaped}':fontcolor=white@0.85:"
            "fontsize=23:box=1:boxcolor=black@0.40:boxborderw=8:x=24:y=24"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg_bin(),
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(video_source),
            "-i",
            str(audio_source),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-vf",
            ",".join(vf_parts),
            "-c:v",
            "libx264",
            "-crf",
            "16",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    return output


def compute_candidate_shifts(
    raw_sync: dict[str, Any],
    *,
    fps: int = 24,
    spread_max_ms: int = 80,
) -> list[int]:
    windows = raw_sync.get("windows") or []
    offsets = [w["picture_late_ms"] for w in windows]
    if not offsets:
        return []
    if max(offsets) - min(offsets) > spread_max_ms:
        return []
    if any(abs(x) >= 360 for x in offsets):
        return []
    mid = (min(offsets) + max(offsets)) / 2.0
    shifts = [round(mid * fps / 1000.0)]
    overall = raw_sync["overall"]["picture_late_ms"]
    if abs(mid - overall) <= 40:
        shifts.append(round(overall * fps / 1000.0))
    return list(dict.fromkeys(shifts))


def is_sync_accepted(
    sync_result: dict[str, Any],
    *,
    abs_max_ms: int = 40,
    confidence_min: float = 1.5,
) -> bool:
    windows = sync_result.get("windows") or []
    if not windows:
        return False
    if abs(sync_result["overall"]["picture_late_ms"]) > abs_max_ms:
        return False
    return all(
        abs(w["picture_late_ms"]) <= abs_max_ms and w["confidence"] >= confidence_min
        for w in windows
    )


def review_clip(
    *,
    candidate_mp4: Path,
    dry_wav: Path,
    accepted_output: Path,
    audit_dir: Path,
    reference_mp4: Path | None = None,
    ai_label: str = "AI 生成",
    gates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    g = {
        "dryWaveformCorrMin": 0.975,
        "trailingPeakMax": 200.0,
        "syncWindowSpreadMaxMs": 80,
        "syncAcceptedAbsMaxMs": 40,
        "syncConfidenceMin": 1.5,
        **(gates or {}),
    }
    audit_dir.mkdir(parents=True, exist_ok=True)
    stem = candidate_mp4.stem
    receipt_path = audit_dir / f"{stem}.json"

    wave_res = check_waveform_match(
        dry_wav,
        candidate_mp4,
        corr_min=float(g["dryWaveformCorrMin"]),
        trailing_peak_max=float(g["trailingPeakMax"]),
    )
    result: dict[str, Any] = {
        "clip": str(candidate_mp4),
        "dryWav": str(dry_wav),
        "waveform": wave_res,
        "accepted": False,
    }
    if not wave_res["ok"]:
        result["verdict"] = "retry_required"
        result["reason"] = "波形相关度不足或尾部截断可听音频"
        save_json(receipt_path, result)
        return result

    raw_sync = measure_syncnet(candidate_mp4)
    result["rawSync"] = raw_sync
    shifts = compute_candidate_shifts(
        raw_sync,
        fps=24,
        spread_max_ms=int(g["syncWindowSpreadMaxMs"]),
    )
    result["candidateShifts"] = shifts
    result["attempts"] = []

    audio_source = reference_mp4 if reference_mp4 else candidate_mp4
    target_frames = video_nframes(audio_source)
    expected_audio_sha = audio_stream_sha256(audio_source)

    def _sync_rank(sync: dict[str, Any]) -> tuple[int, int, float]:
        overall_abs = abs(sync["overall"]["picture_late_ms"])
        win_sum = sum(abs(w["picture_late_ms"]) for w in sync.get("windows", []))
        return (overall_abs, win_sum, float(sync["overall"]["zero_distance"]))

    best_winner: tuple[tuple[int, int, float], int, Path, dict[str, Any]] | None = None
    try:
        for shift in shifts:
            stage = audit_dir / f"_stage-{stem}-shift{shift}.mp4"
            retime_video_keep_audio(
                video_source=candidate_mp4,
                audio_source=audio_source,
                output=stage,
                target_frames=target_frames,
                advance_frames=shift,
                ai_label=ai_label,
            )
            synced = measure_syncnet(stage)
            passed = is_sync_accepted(
                synced,
                abs_max_ms=int(g["syncAcceptedAbsMaxMs"]),
                confidence_min=float(g["syncConfidenceMin"]),
            )
            result["attempts"].append({"advanceFrames": shift, "passed": passed, "sync": synced})
            if not passed:
                stage.unlink(missing_ok=True)
                continue

            rank = _sync_rank(synced)
            if best_winner is None or rank < best_winner[0]:
                if best_winner is not None:
                    best_winner[2].unlink(missing_ok=True)
                best_winner = (rank, shift, stage, synced)
            else:
                stage.unlink(missing_ok=True)

        if best_winner is not None:
            _, shift, stage, synced = best_winner
            actual_frames = video_nframes(stage)
            actual_sha = audio_stream_sha256(stage)
            if actual_frames != target_frames:
                raise RuntimeError(f"校时后帧数漂移: expected={target_frames}, actual={actual_frames}")
            if actual_sha != expected_audio_sha:
                raise RuntimeError(f"校时后音轨哈希改变: expected={expected_audio_sha}, actual={actual_sha}")

            grid_path = audit_dir / f"{stem}-grid.jpg"
            make_visual_grid(stage, grid_path)
            hands = run_handcheck(stage)

            accepted_output.parent.mkdir(parents=True, exist_ok=True)
            stage.replace(accepted_output)
            best_winner = None
            result.update({
                "accepted": True,
                "verdict": "accepted",
                "advanceFrames": shift,
                "frames": actual_frames,
                "audioSha256": actual_sha,
                "acceptedSync": synced,
                "hands": hands,
                "visualGrid": str(grid_path),
                "acceptedPath": str(accepted_output),
            })
            save_json(receipt_path, result)
            return result
    finally:
        if best_winner is not None:
            best_winner[2].unlink(missing_ok=True)

    result["verdict"] = "retry_required"
    result["reason"] = (
        "各窗口偏移不一致或贴近 ±360ms 边界，不允许强行移帧，请换 seed 重生成"
        if not shifts
        else "尝试整帧移画后仍未全部落入 ±40ms 且 confidence >= 1.5，请换 seed 重生成"
    )
    save_json(receipt_path, result)
    return result


def review_project_clip(
    project_ref: str | Path,
    candidate_mp4: Path,
    *,
    chapter_id: str,
    reference_mp4: Path | None = None,
) -> dict[str, Any]:
    project_dir, manifest = load_project(project_ref)
    dirs = project_dirs(project_dir)
    recipe = load_recipe(manifest.get("recipe", "curator-h3-explainer"))
    dry = dirs["h3_voice"] / f"voice-{chapter_id}.wav"
    if not dry.is_file():
        raise FileNotFoundError(f"找不到对应出镜干声: {dry}")
    return review_clip(
        candidate_mp4=candidate_mp4,
        dry_wav=dry,
        accepted_output=dirs["h3_accepted"] / f"{chapter_id}.mp4",
        audit_dir=dirs["h3_audit"],
        reference_mp4=reference_mp4,
        ai_label=(manifest.get("presenter") or {}).get("aiLabel", "AI 生成"),
        gates=recipe.get("gates"),
    )
