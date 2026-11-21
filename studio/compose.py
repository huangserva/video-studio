"""策展版式外壳生成、整数帧分镜合成、混音、字幕烧录与无损换人回填。"""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .config import ffmpeg_bin
from .project import load_json, load_project, load_style, project_dirs, save_json
from .sync_gate import audio_stream_sha256, video_nframes

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}


def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    p = Path(path)
    if p.is_file():
        return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def render_shells(
    *,
    manifest: dict[str, Any],
    style: dict[str, Any],
    shells_dir: Path,
) -> None:
    """按样式 JSON 渲染每章的 split / wide / host 1920x1080 底板。"""
    w = int(style["canvas"]["width"])
    h = int(style["canvas"]["height"])
    c = style["colors"]
    fonts = style["fonts"]
    sections = manifest.get("sections") or []
    title = manifest.get("title", "")
    subtitle = manifest.get("subtitle", "")

    for ch in manifest["chapters"]:
        cid = ch["id"]
        sec_idx = int(ch.get("sectionIndex", 0))
        ch_title = ch.get("title", cid)

        for variant in ("split", "wide"):
            im = Image.new("RGB", (w, h), c["background"])
            d = ImageDraw.Draw(im)
            d.text((72, 36), title, fill=c["ink"], font=_load_font(fonts["serif"], 47))
            if subtitle:
                d.text((360, 55), subtitle, fill=c["muted"], font=_load_font(fonts["sans"], 23))

            if sections:
                pill_step = 113
                x = max(720, 1848 - len(sections) * pill_step + 15)
                for j, label in enumerate(sections):
                    active = sec_idx == j
                    if active:
                        d.rounded_rectangle((x - 15, 48, x + 85, 91), radius=21, fill=c["accent"])
                    d.text(
                        (x + 34, 68),
                        label,
                        anchor="mm",
                        fill=c["accentText"] if active else c["muted"],
                        font=_load_font(fonts["sans"], 21),
                    )
                    x += pill_step

            d.line((72, 113, 1848, 113), fill=c["rule"], width=2)
            d.text((72, 943), ch_title, fill=c["caption"], font=_load_font(fonts["sans"], 25))
            d.line((72, 1040, 1848, 1040), fill=c["ruleSoft"], width=2)

            if variant == "split":
                m = style["layouts"]["split"]["media"]
                hb = style["layouts"]["split"]["host"]
                d.rectangle((m["x"] - 3, m["y"] - 3, m["x"] + m["w"] + 3, m["y"] + m["h"] + 3), outline=c["frame"], width=3)
                d.rectangle((hb["x"] - 3, hb["y"] - 3, hb["x"] + hb["w"] + 3, hb["y"] + hb["h"] + 3), outline=c["frame"], width=3)

            im.save(shells_dir / f"shell-{cid}-{variant}.png")

        hero_lines = ch.get("heroLines") or [title, ch_title]
        im = Image.new("RGB", (w, h), c["background"])
        d = ImageDraw.Draw(im)
        d.line((88, 120, 1808, 120), fill=c["rule"], width=2)
        d.text((96, 435), str(hero_lines[0]), fill=c["ink"], font=_load_font(fonts["serif"], 76))
        if len(hero_lines) > 1:
            d.text((100, 546), str(hero_lines[1]), fill=c["accent"], font=_load_font(fonts["sans"], 43))
        hb = style["layouts"]["host"]["host"]
        d.rectangle((hb["x"] - 3, hb["y"] - 3, hb["x"] + hb["w"] + 3, hb["y"] + hb["h"] + 3), outline=c["frame"], width=3)
        im.save(shells_dir / f"shell-{cid}-host.png")


def _fit_filter(w: int, h: int) -> str:
    return f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,crop={w}:{h},setsar=1"


def _media_input(path: Path, start: float, fps: int) -> list[str]:
    if path.suffix.lower() in VIDEO_EXTS:
        return ["-ss", f"{start:.4f}", "-i", str(path)]
    return ["-loop", "1", "-framerate", str(fps), "-i", str(path)]


def _escape_ass_path(path: Path) -> str:
    raw = str(path.resolve())
    return raw.replace("\\", r"\\").replace(":", r"\:").replace("'", r"\'")


def compose_project(
    project_ref: str | Path,
    *,
    output_name: str | None = None,
    burn_subtitles: bool = True,
    max_workers: int = 2,
) -> dict[str, Any]:
    """按 project.json + work/timeline.json 整数帧合成 1080p 成片。"""
    project_dir, manifest = load_project(project_ref)
    dirs = project_dirs(project_dir)
    style = load_style(manifest.get("style", "museum-paper"))
    timeline = load_json(project_dir / "work" / "timeline.json")

    w = int(style["canvas"]["width"])
    h = int(style["canvas"]["height"])
    fps = int(style["canvas"]["fps"])
    layouts = style["layouts"]

    render_shells(manifest=manifest, style=style, shells_dir=dirs["shells"])

    tl_entries = timeline["chapters"]
    ch_by_id = {c["id"]: c for c in manifest["chapters"]}
    total_dur = float(timeline["totalDuration"])

    shot_plan: list[dict[str, Any]] = []
    for idx, entry in enumerate(tl_entries):
        cid = entry["id"]
        ch = ch_by_id[cid]
        ch_start = float(entry["start"])
        ch_end = float(tl_entries[idx + 1]["start"]) if idx + 1 < len(tl_entries) else total_dur
        ch_span = ch_end - ch_start

        cuts = ch.get("cuts") or []
        if not cuts:
            raise ValueError(f"第 {cid} 章没有定义 cuts")
        for k, cut in enumerate(cuts):
            local_start = float(cut.get("at", 0.0))
            local_end = float(cuts[k + 1]["at"]) if k + 1 < len(cuts) else ch_span
            if local_end <= local_start:
                continue
            shot_plan.append({
                "index": len(shot_plan),
                "chapterId": cid,
                "chapterOriginSec": ch_start,
                "globalStart": ch_start + local_start,
                "globalEnd": ch_start + local_end,
                "localStart": local_start,
                "layout": cut["layout"],
                "source": cut.get("source", ""),
                "sourceStart": float(cut.get("sourceStart", 0.0)),
            })

    host_windows: list[dict[str, Any]] = []
    frame_cursor = 0
    for s in shot_plan:
        count = max(1, round(s["globalEnd"] * fps) - round(s["globalStart"] * fps))
        s["frames"] = count
        s["startFrame"] = frame_cursor
        s["endFrame"] = frame_cursor + count
        if s["layout"] in ("split", "host"):
            hb = layouts[s["layout"]]["host"]
            host_windows.append({
                "chapterId": s["chapterId"],
                "layout": s["layout"],
                "chapterOriginFrame": round(s["chapterOriginSec"] * fps),
                "startFrame": s["startFrame"],
                "endFrame": s["endFrame"],
                "startSec": round(s["startFrame"] / fps, 4),
                "endSec": round(s["endFrame"] / fps, 4),
                "crop": [hb["x"], hb["y"], hb["w"], hb["h"]],
            })
        frame_cursor += count

    def render_shot(s: dict[str, Any]) -> Path:
        cid = s["chapterId"]
        layout = s["layout"]
        count = s["frames"]
        target = dirs["shots"] / f"{s['index']:03d}-{cid}-{layout}.mp4"
        host_clip = dirs["h3_accepted"] / f"{cid}.mp4"

        cmd = [ffmpeg_bin(), "-nostdin", "-y", "-v", "error", "-filter_complex_threads", "2", "-threads", "2"]
        if layout == "full":
            src = (project_dir / s["source"]).resolve()
            if not src.is_file():
                raise FileNotFoundError(f"缺少镜头素材: {src}")
            cmd += _media_input(src, s["sourceStart"], fps)
            flt = f"[0:v]{_fit_filter(w, h)},fps={fps},format=yuv420p[v]"
        elif layout == "host":
            if not host_clip.is_file():
                raise FileNotFoundError(f"缺少已验收数字人片段: {host_clip}")
            hb = layouts["host"]["host"]
            cmd += _media_input(host_clip, s["localStart"], fps)
            cmd += _media_input(dirs["shells"] / f"shell-{cid}-host.png", 0.0, fps)
            flt = (
                f"[0:v]{_fit_filter(hb['w'], hb['h'])},tpad=stop_mode=clone:stop_duration=0.25[host];"
                f"[1:v][host]overlay={hb['x']}:{hb['y']},fps={fps},format=yuv420p[v]"
            )
        elif layout in ("split", "wide"):
            src = (project_dir / s["source"]).resolve()
            if not src.is_file():
                raise FileNotFoundError(f"缺少镜头素材: {src}")
            mb = layouts[layout]["media"]
            cmd += _media_input(src, s["sourceStart"], fps)
            cmd += _media_input(dirs["shells"] / f"shell-{cid}-{layout}.png", 0.0, fps)
            flt = f"[0:v]{_fit_filter(mb['w'], mb['h'])}[main];[1:v][main]overlay={mb['x']}:{mb['y']}[page]"
            if layout == "split":
                if not host_clip.is_file():
                    raise FileNotFoundError(f"缺少已验收数字人片段: {host_clip}")
                hb = layouts["split"]["host"]
                cmd += _media_input(host_clip, s["localStart"], fps)
                flt += (
                    f";[2:v]{_fit_filter(hb['w'], hb['h'])},tpad=stop_mode=clone:stop_duration=0.25[host];"
                    f"[page][host]overlay={hb['x']}:{hb['y']},fps={fps},format=yuv420p[v]"
                )
            else:
                flt += f";[page]fps={fps},format=yuv420p[v]"
        else:
            raise ValueError(f"未知版式: {layout}")

        cmd += [
            "-filter_complex",
            flt,
            "-map",
            "[v]",
            "-an",
            "-frames:v",
            str(count),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "17",
            "-threads",
            "2",
            str(target),
        ]
        subprocess.run(cmd, check=True)
        actual = video_nframes(target)
        if actual != count:
            raise RuntimeError(f"镜头 {s['index']} 帧数不匹配: expected={count}, actual={actual}")
        return target

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        shot_files = list(pool.map(render_shot, shot_plan))

    concat_file = dirs["shots"] / "shots.ffconcat"
    concat_file.write_text(
        "ffconcat version 1.0\n" + "".join(f"file '{f.name}'\n" for f in shot_files),
        encoding="utf-8",
    )

    master_dry = project_dir / timeline["masterDryWav"]
    duration = frame_cursor / fps
    bgm_cfg = manifest.get("bgm") or {}
    bgm_rel = (bgm_cfg.get("path") or "").strip()
    bgm_path = (project_dir / bgm_rel).resolve() if bgm_rel else None

    from .qc import measure_integrated_lufs

    target_lufs = float((manifest.get("voice") or {}).get("targetLufs", -16.5))
    src_lufs = measure_integrated_lufs(master_dry)
    voice_gain_db = 0.0
    if src_lufs is not None and src_lufs > -60.0:
        voice_gain_db = max(-12.0, min(18.0, target_lufs - src_lufs))

    ass_path = dirs["subtitles"] / "narration.ass"
    use_ass = burn_subtitles and ass_path.is_file()

    out_file = dirs["outputs"] / (output_name or f"{manifest.get('id', 'final')}.mp4")
    stage_file = out_file.with_suffix(".stage.mp4")

    cmd = [
        ffmpeg_bin(),
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-filter_complex_threads",
        "2",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-i",
        str(master_dry),
    ]
    filter_chains: list[str] = []
    if bgm_path and bgm_path.is_file():
        cmd += ["-stream_loop", "-1", "-i", str(bgm_path)]
        vol_voice = float(bgm_cfg.get("volumeUnderVoice", 0.085))
        fade_st = max(0.0, duration - 1.4)
        filter_chains.append(
            f"[2:a]volume={vol_voice},atrim=end={duration:.4f},apad=pad_dur=0.25,"
            f"afade=t=out:st={fade_st:.4f}:d=1.4[m];"
            f"[1:a]volume={voice_gain_db:+.2f}dB,apad,atrim=end={duration:.4f}[voice];"
            "[voice][m]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95:level=false[a]"
        )
        audio_map = "[a]"
    else:
        filter_chains.append(
            f"[1:a]volume={voice_gain_db:+.2f}dB,apad,atrim=end={duration:.4f},"
            "alimiter=limit=0.95:level=false[a]"
        )
        audio_map = "[a]"

    if use_ass:
        filter_chains.append(f"[0:v]ass='{_escape_ass_path(ass_path)}'[v]")
        video_map = "[v]"
    else:
        video_map = "0:v:0"

    cmd += [
        "-filter_complex",
        ";".join(filter_chains),
        "-map",
        video_map,
        "-map",
        audio_map,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "4",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        "-t",
        f"{duration:.4f}",
        str(stage_file),
    ]
    subprocess.run(cmd, check=True)
    stage_file.replace(out_file)

    save_json(project_dir / "work" / "host-windows.json", host_windows)
    return {
        "output": str(out_file),
        "frames": video_nframes(out_file),
        "expectedFrames": frame_cursor,
        "durationSec": round(duration, 4),
        "subtitlesBurned": use_ass,
        "hostWindows": len(host_windows),
        "audioSha256": audio_stream_sha256(out_file),
    }


def swap_hosts(
    *,
    base_video: Path,
    windows_json: Path,
    hosts_dir: Path,
    output_video: Path,
    fps: int = 30,
) -> dict[str, Any]:
    """只替换成片中的人物小窗，整数帧时钟精确叠画，原音轨 -c:a copy 并校验哈希不变。"""
    windows = load_json(windows_json)
    if not windows:
        raise ValueError("host-windows.json 为空，没有需要替换的人物窗口")

    chapter_ids = list(dict.fromkeys(w["chapterId"] for w in windows))
    clip_map: dict[str, Path] = {}
    for cid in chapter_ids:
        direct = hosts_dir / f"{cid}.mp4"
        if direct.is_file():
            clip_map[cid] = direct
            continue
        matches = sorted(hosts_dir.glob(f"{cid}*.mp4"))
        if not matches:
            raise FileNotFoundError(f"在 {hosts_dir} 找不到章节 {cid} 的替换人物视频")
        clip_map[cid] = matches[0]

    args = [ffmpeg_bin(), "-nostdin", "-y", "-v", "error", "-filter_complex_threads", "3", "-i", str(base_video)]
    input_idx: dict[str, int] = {}
    for cid in chapter_ids:
        input_idx[cid] = len(input_idx) + 1
        args += ["-i", str(clip_map[cid])]

    filters = [f"[0:v]settb=1/{fps},setpts=N[base]"]
    for cid in chapter_ids:
        items = [i for i, w in enumerate(windows) if w["chapterId"] == cid]
        origin = int(windows[items[0]]["chapterOriginFrame"])
        src = f"[{input_idx[cid]}:v]fps={fps},settb=1/{fps},setpts=N+{origin}"
        split = f",split={len(items)}" if len(items) > 1 else ""
        outs = "".join(f"[src{i}]" for i in items)
        filters.append(src + split + outs)

    last = "base"
    for i, w in enumerate(windows):
        x, y, bw, bh = w["crop"]
        start = int(w["startFrame"])
        end = int(w["endFrame"])
        filters.append(f"[src{i}]scale={bw}:{bh}:flags=lanczos,setsar=1[host{i}]")
        enable = f"gte(t,{(start - 0.01) / fps:.9f})*lt(t,{(end - 0.01) / fps:.9f})"
        filters.append(f"[{last}][host{i}]overlay={x}:{y}:enable='{enable}'[shot{i}]")
        last = f"shot{i}"

    graph_file = output_video.with_suffix(".ffgraph")
    graph_file.parent.mkdir(parents=True, exist_ok=True)
    graph_file.write_text(";\n".join(filters), encoding="utf-8")

    stage = output_video.with_suffix(".stage.mp4")
    args += [
        "-filter_complex_script",
        str(graph_file),
        "-map",
        f"[{last}]",
        "-map",
        "0:a:0",
        "-c:v",
        "libx264",
        "-crf",
        "16",
        "-preset",
        "veryfast",
        "-threads",
        "4",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(stage),
    ]
    subprocess.run(args, check=True)

    base_frames = video_nframes(base_video)
    out_frames = video_nframes(stage)
    base_sha = audio_stream_sha256(base_video)
    out_sha = audio_stream_sha256(stage)
    if out_frames != base_frames:
        raise RuntimeError(f"换人后帧数不一致: base={base_frames}, out={out_frames}")
    if out_sha != base_sha:
        raise RuntimeError(f"换人后音轨哈希改变: base={base_sha}, out={out_sha}")

    stage.replace(output_video)
    return {
        "output": str(output_video),
        "frames": out_frames,
        "audioSha256": out_sha,
        "audioHashMatchedBase": True,
        "replacedWindows": len(windows),
    }
