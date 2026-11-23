"""成片终检闸：容器规格 + 响度 + 12帧抽检图 + 人物小窗 SyncNet 复测 + 音轨哈希。"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_LUFS_TOLERANCE,
    DEFAULT_MAX_TRUE_PEAK_DB,
    DEFAULT_TARGET_LUFS,
    ffmpeg_bin,
    ffprobe_bin,
)
from .project import load_json, save_json
from .sync_gate import audio_stream_sha256, make_visual_grid, measure_syncnet, video_nframes

LUFS_RE = re.compile(r"I:\s*(-?\d+(?:\.\d+)?)\s*LUFS")
TRUE_PEAK_RE = re.compile(r"True peak:\s*\n\s*Peak:\s*(-?\d+(?:\.\d+)?|-inf)\s*dBFS")


def probe_video(path: Path) -> dict[str, Any]:
    raw = subprocess.check_output([
        ffprobe_bin(),
        "-v",
        "error",
        "-show_entries",
        "format=duration,size,bit_rate:stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels",
        "-of",
        "json",
        str(path),
    ])
    data = json.loads(raw)
    vstream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    astream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    if not vstream:
        raise RuntimeError(f"找不到视频流: {path}")
    num, den = (int(x) for x in vstream["r_frame_rate"].split("/"))
    fps = round(num / den, 4) if den else float(num)
    return {
        "width": int(vstream["width"]),
        "height": int(vstream["height"]),
        "fps": fps,
        "videoCodec": vstream.get("codec_name"),
        "audioCodec": astream.get("codec_name") if astream else None,
        "audioSampleRate": int(astream["sample_rate"]) if astream and astream.get("sample_rate") else None,
        "audioChannels": int(astream["channels"]) if astream and astream.get("channels") else None,
        "durationSec": round(float(data["format"]["duration"]), 4),
        "sizeBytes": int(data["format"]["size"]),
        "frames": video_nframes(path),
    }


def measure_loudness(path: Path) -> dict[str, float | None]:
    """整体响度（LUFS）和真峰值（dBTP），取 ebur128 最后的汇总。"""
    proc = subprocess.run(
        [
            ffmpeg_bin(),
            "-nostdin",
            "-v",
            "info",
            "-i",
            str(path),
            "-vn",
            "-af",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    lufs = LUFS_RE.findall(proc.stderr)
    peaks = TRUE_PEAK_RE.findall(proc.stderr)
    tp: float | None = None
    if peaks:
        tp = float("-inf") if peaks[-1] == "-inf" else float(peaks[-1])
    return {"integratedLufs": float(lufs[-1]) if lufs else None, "truePeakDb": tp}


def measure_integrated_lufs(path: Path) -> float | None:
    proc = subprocess.run(
        [
            ffmpeg_bin(),
            "-nostdin",
            "-v",
            "info",
            "-i",
            str(path),
            "-af",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    matches = LUFS_RE.findall(proc.stderr)
    return float(matches[-1]) if matches else None


def run_final_qc(
    video: Path,
    *,
    report_dir: Path,
    windows_json: Path | None = None,
    base_audio_video: Path | None = None,
    expected_width: int = 1920,
    expected_height: int = 1080,
    expected_fps: float = 30.0,
    max_sync_abs_ms: int = 40,
    min_window_duration_sec: float = 2.0,
    target_lufs: float | None = DEFAULT_TARGET_LUFS,
    lufs_tolerance: float = DEFAULT_LUFS_TOLERANCE,
    max_true_peak_db: float = DEFAULT_MAX_TRUE_PEAK_DB,
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    spec = probe_video(video)
    loud = measure_loudness(video) if spec["audioCodec"] else {"integratedLufs": None, "truePeakDb": None}
    lufs = loud["integratedLufs"]
    audio_sha = audio_stream_sha256(video) if spec["audioCodec"] else None
    grid_path = report_dir / f"{video.stem}-qc-grid.jpg"
    make_visual_grid(video, grid_path, count=12, fps=spec["fps"])

    failures: list[str] = []
    if spec["width"] != expected_width or spec["height"] != expected_height:
        failures.append(f"分辨率不符: {spec['width']}x{spec['height']} != {expected_width}x{expected_height}")
    if abs(spec["fps"] - expected_fps) > 0.05:
        failures.append(f"帧率不符: {spec['fps']} != {expected_fps}")
    if not spec["audioCodec"]:
        failures.append("缺少音频流")
    elif target_lufs is not None:
        if lufs is None:
            failures.append("测不出整体响度")
        elif abs(lufs - target_lufs) > lufs_tolerance + 1e-9:
            failures.append(
                f"整体响度 {lufs:.1f} LUFS 不在 {target_lufs - lufs_tolerance:.1f} 到 {target_lufs + lufs_tolerance:.1f} LUFS 之间"
            )
        tp = loud["truePeakDb"]
        if tp is not None and tp > max_true_peak_db + 1e-9:
            failures.append(f"真峰值 {tp:.1f} dBTP 超过 {max_true_peak_db:.1f} dBTP")

    audio_match = None
    if base_audio_video is not None:
        base_sha = audio_stream_sha256(base_audio_video)
        audio_match = audio_sha == base_sha
        if not audio_match:
            failures.append(f"音轨哈希与底片不一致: {audio_sha} != {base_sha}")

    host_checks: list[dict[str, Any]] = []
    if windows_json and windows_json.is_file():
        windows = load_json(windows_json)
        for idx, w in enumerate(windows):
            start_sec = float(w["startSec"])
            end_sec = float(w["endSec"])
            dur = end_sec - start_sec
            if dur < min_window_duration_sec:
                host_checks.append({
                    "windowIndex": idx,
                    "chapterId": w.get("chapterId"),
                    "skipped": True,
                    "reason": f"窗口时长 {dur:.2f}s 短于 {min_window_duration_sec}s",
                })
                continue
            crop = tuple(int(v) for v in w["crop"])
            try:
                sync = measure_syncnet(video, seek=start_sec, duration=dur, crop=crop)
                worst = max(
                    [abs(sync["overall"]["picture_late_ms"])]
                    + [abs(win["picture_late_ms"]) for win in sync["windows"]]
                )
                ok = worst <= max_sync_abs_ms
                if not ok:
                    failures.append(
                        f"成片小窗 #{idx} (章 {w.get('chapterId')}) 最大绝对偏移 {worst}ms > {max_sync_abs_ms}ms"
                    )
                host_checks.append({
                    "windowIndex": idx,
                    "chapterId": w.get("chapterId"),
                    "layout": w.get("layout"),
                    "startSec": start_sec,
                    "durationSec": round(dur, 3),
                    "crop": list(crop),
                    "worstAbsMs": worst,
                    "ok": ok,
                    "sync": sync,
                })
            except Exception as exc:
                failures.append(f"成片小窗 #{idx} (章 {w.get('chapterId')}) 复测异常: {exc}")
                host_checks.append({
                    "windowIndex": idx,
                    "chapterId": w.get("chapterId"),
                    "ok": False,
                    "error": str(exc),
                })

    report = {
        "video": str(video),
        "passed": len(failures) == 0,
        "failures": failures,
        "spec": spec,
        "integratedLufs": lufs,
        "truePeakDb": loud["truePeakDb"],
        "loudnessTarget": None if target_lufs is None else {
            "lufs": target_lufs, "toleranceLu": lufs_tolerance, "maxTruePeakDb": max_true_peak_db,
        },
        "audioSha256": audio_sha,
        "audioMatchedBase": audio_match,
        "visualGrid": str(grid_path),
        "hostWindowChecks": host_checks,
    }
    save_json(report_dir / f"{video.stem}-qc.json", report)
    return report
