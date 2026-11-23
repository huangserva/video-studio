"""策展版式外壳生成、整数帧分镜合成、混音、字幕烧录与无损换人回填。"""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .config import DEFAULT_MAX_TRUE_PEAK_DB, DEFAULT_TARGET_LUFS, ffmpeg_bin
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


def _fit_image(im: Image.Image, w: int, h: int) -> Image.Image:
    scale = max(w / im.width, h / im.height)
    im = im.resize((max(w, round(im.width * scale)), max(h, round(im.height * scale))), Image.LANCZOS)
    x = (im.width - w) // 2
    y = (im.height - h) // 2
    return im.crop((x, y, x + w, y + h))


def _draw_ai_label(d: ImageDraw.ImageDraw, x: int, y: int, label: str, font: Any) -> None:
    if not label:
        return
    l, t, r, b = d.textbbox((0, 0), label, font=font)
    d.rounded_rectangle((x, y, x + (r - l) + 28, y + (b - t) + 18), radius=8, fill=(20, 20, 20))
    d.text((x + 14 - l, y + 9 - t), label, fill=(255, 255, 255), font=font)


def render_cover(
    *,
    project_dir: Path,
    manifest: dict[str, Any],
    style: dict[str, Any],
    target: Path,
) -> Path:
    """封面：标题和副标题由我们自己排版叠字，不让生图模型写中文。

    cover.image 有图就用图做底；没有就用样式底色 + 主讲人参考图。
    """
    w = int(style["canvas"]["width"])
    h = int(style["canvas"]["height"])
    c = style["colors"]
    fonts = style["fonts"]
    cover = manifest.get("cover") or {}
    presenter = manifest.get("presenter") or {}
    headline = str(cover.get("headline") or manifest.get("title", ""))
    subline = str(cover.get("subline") or manifest.get("subtitle", ""))
    ai_label = (presenter.get("aiLabel") or "").strip() if presenter.get("enabled") else ""
    label_font = _load_font(fonts["sans"], 26)

    image_rel = (cover.get("image") or "").strip()
    if image_rel:
        src = (project_dir / image_rel).resolve()
        if not src.is_file():
            raise FileNotFoundError(f"缺少封面底图: {src}")
        im = _fit_image(Image.open(src).convert("RGB"), w, h)
        shade = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        sd = ImageDraw.Draw(shade)
        for i in range(420):
            sd.line((0, h - i, w, h - i), fill=(0, 0, 0, int(170 * (1 - i / 420))))
        im = Image.alpha_composite(im.convert("RGBA"), shade).convert("RGB")
        d = ImageDraw.Draw(im)
        d.text((96, h - 300), headline, fill=(255, 255, 255), font=_load_font(fonts["serif"], 96))
        if subline:
            d.text((100, h - 170), subline, fill=(235, 235, 235), font=_load_font(fonts["sans"], 44))
        _draw_ai_label(d, 40, 36, ai_label, label_font)
    else:
        im = Image.new("RGB", (w, h), c["background"])
        d = ImageDraw.Draw(im)
        d.line((88, 120, 1808, 120), fill=c["rule"], width=2)
        hb = style["layouts"]["host"]["host"]
        portrait_rel = presenter.get("portraitImage") or presenter.get("waistImage") or ""
        portrait = (project_dir / portrait_rel) if portrait_rel else None
        text_right = w - 120
        if presenter.get("enabled") and portrait and portrait.is_file():
            ph = _fit_image(Image.open(portrait).convert("RGB"), hb["w"], hb["h"])
            im.paste(ph, (hb["x"], hb["y"]))
            d.rectangle((hb["x"] - 3, hb["y"] - 3, hb["x"] + hb["w"] + 3, hb["y"] + hb["h"] + 3), outline=c["frame"], width=3)
            _draw_ai_label(d, hb["x"] + 18, hb["y"] + 18, ai_label, label_font)
            text_right = hb["x"] - 48
        size = 104
        font = _load_font(fonts["serif"], size)
        while size > 56 and d.textlength(headline, font=font) > text_right - 96:
            size -= 4
            font = _load_font(fonts["serif"], size)
        d.text((96, 400), headline, fill=c["ink"], font=font)
        if subline:
            sub_size = 44
            sub_font = _load_font(fonts["sans"], sub_size)
            while sub_size > 28 and d.textlength(subline, font=sub_font) > text_right - 100:
                sub_size -= 2
                sub_font = _load_font(fonts["sans"], sub_size)
            d.text((100, 400 + size + 40), subline, fill=c["accent"], font=sub_font)
    target.parent.mkdir(parents=True, exist_ok=True)
    im.save(target)
    return target


def _host_clip_sec(path: Path, host_fps: int = 24) -> float | None:
    if not path.is_file():
        return None
    return video_nframes(path) / host_fps


def resolve_cut_times(
    chapter: dict[str, Any],
    *,
    sentences: list[dict[str, Any]] | None,
    on_camera_sec: float,
    host_clip_sec: float | None,
    lead_sec: float = 0.10,
) -> list[float]:
    """切点三种写法：固定秒数 at；atSentence（第 N 句开口前 0.1 秒）；at = "onCameraEnd"（出镜段结尾，不超过 H3 实际时长）。"""
    cid = chapter.get("id", "?")
    times: list[float] = []
    for k, cut in enumerate(chapter.get("cuts") or []):
        if "atSentence" in cut:
            if not sentences:
                raise ValueError(f"第 {cid} 章第 {k + 1} 个切点用了 atSentence，但这一章没有逐句配音记录")
            n = int(cut["atSentence"])
            hit = next((s for s in sentences if int(s["index"]) == n), None)
            if hit is None:
                raise ValueError(f"第 {cid} 章没有第 {n} 句（atSentence={n}）")
            t = max(0.0, float(hit["startSec"]) - lead_sec)
        elif cut.get("at") == "onCameraEnd":
            t = on_camera_sec
            if host_clip_sec is not None:
                t = min(t, host_clip_sec)
        else:
            t = float(cut.get("at", 0.0))
        times.append(round(t, 4))
    return times


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
    lead_in = float(timeline.get("leadInSec", 0.0))
    files_by_id = {f["id"]: f for f in timeline.get("files", [])}

    # 出镜干声和已验收 H3 的收据对不上，说明配音改过但 H3 没重做：直接拦下，不出假绿成片。
    unverified_hosts: list[str] = []
    for entry in tl_entries:
        cid = entry["id"]
        uses_host = any(c.get("layout") in ("split", "host") for c in (ch_by_id[cid].get("cuts") or []))
        if not uses_host:
            continue
        receipt_path = dirs["h3_accepted"] / f"{cid}.receipt.json"
        want = (files_by_id.get(cid) or {}).get("onCameraSha256")
        if receipt_path.is_file() and want:
            got = load_json(receipt_path).get("drySha256")
            if got and got != want:
                raise RuntimeError(
                    f"第 {cid} 章出镜段干声已经变了（现在 {want[:12]}，H3 验收时 {got[:12]}）。"
                    "已验收的 H3 片段对不上新配音，必须重新 h3-prepare → h3-submit → sync-gate。"
                )
        else:
            unverified_hosts.append(cid)

    shot_plan: list[dict[str, Any]] = []
    if lead_in > 0:
        cover_png = render_cover(
            project_dir=project_dir, manifest=manifest, style=style, target=dirs["shells"] / "cover.png",
        )
        shot_plan.append({
            "index": 0,
            "chapterId": "cover",
            "chapterOriginSec": 0.0,
            "globalStart": 0.0,
            "globalEnd": lead_in,
            "localStart": 0.0,
            "layout": "cover",
            "source": str(cover_png),
            "sourceStart": 0.0,
        })
    for idx, entry in enumerate(tl_entries):
        cid = entry["id"]
        ch = ch_by_id[cid]
        ch_start = float(entry["start"])
        ch_end = float(tl_entries[idx + 1]["start"]) if idx + 1 < len(tl_entries) else total_dur
        ch_span = ch_end - ch_start

        cuts = ch.get("cuts") or []
        if not cuts:
            raise ValueError(f"第 {cid} 章没有定义 cuts")
        sent_file = dirs["narration"] / f"{cid}.sentences.json"
        sentences = load_json(sent_file)["sentences"] if sent_file.is_file() else None
        cut_times = resolve_cut_times(
            ch,
            sentences=sentences,
            on_camera_sec=float(entry.get("onCameraSec") or 0.0),
            host_clip_sec=_host_clip_sec(dirs["h3_accepted"] / f"{cid}.mp4"),
        )
        for k, cut in enumerate(cuts):
            local_start = cut_times[k]
            local_end = cut_times[k + 1] if k + 1 < len(cuts) else ch_span
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
        if layout == "cover":
            cmd += _media_input(Path(s["source"]), 0.0, fps)
            flt = f"[0:v]{_fit_filter(w, h)},fps={fps},format=yuv420p[v]"
        elif layout == "full":
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
    voice_cfg = manifest.get("voice") or {}
    target_lufs = float(voice_cfg.get("targetLufs", DEFAULT_TARGET_LUFS))
    max_tp = float(voice_cfg.get("maxTruePeakDb", DEFAULT_MAX_TRUE_PEAK_DB))

    mix_wav = project_dir / "work" / "master-mix.wav"
    mix_report = render_mix(
        master_dry=master_dry,
        bgm_path=bgm_path if bgm_path and bgm_path.is_file() else None,
        bgm_volume=float(bgm_cfg.get("volumeUnderVoice", 0.085)),
        duration=duration,
        target_lufs=target_lufs,
        max_true_peak_db=max_tp,
        output=mix_wav,
    )

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
        str(mix_wav),
    ]
    if use_ass:
        cmd += ["-filter_complex", f"[0:v]ass='{_escape_ass_path(ass_path)}'[v]", "-map", "[v]"]
    else:
        cmd += ["-map", "0:v:0"]
    cmd += [
        "-map",
        "1:a:0",
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

    cover_png = None
    if lead_in > 0:
        cover_png = out_file.with_name(f"{out_file.stem}-cover.png")
        subprocess.run(
            [ffmpeg_bin(), "-nostdin", "-y", "-v", "error", "-i", str(out_file), "-frames:v", "1", str(cover_png)],
            check=True,
        )

    save_json(project_dir / "work" / "host-windows.json", host_windows)
    return {
        "output": str(out_file),
        "frames": video_nframes(out_file),
        "expectedFrames": frame_cursor,
        "durationSec": round(duration, 4),
        "leadInSec": lead_in,
        "coverPng": str(cover_png) if cover_png else None,
        "subtitlesBurned": use_ass,
        "hostWindows": len(host_windows),
        "hostClipsWithoutReceipt": unverified_hosts,
        "mix": mix_report,
        "audioSha256": audio_stream_sha256(out_file),
    }


def render_mix(
    *,
    master_dry: Path,
    bgm_path: Path | None,
    bgm_volume: float,
    duration: float,
    target_lufs: float = DEFAULT_TARGET_LUFS,
    max_true_peak_db: float = DEFAULT_MAX_TRUE_PEAK_DB,
    output: Path,
    max_passes: int = 4,
) -> dict[str, Any]:
    """人声整体线性增益 + 限幅，测一遍再微调，直到整体响度落在目标 ±0.15 LU 以内。

    限幅门限比真峰值上限再低 1 dB，给 AAC 编码留余量。latency=1 补偿限幅器的前瞻延迟，不让音画错位。
    """
    from .qc import measure_loudness

    limit = 10.0 ** ((max_true_peak_db - 1.0) / 20.0)
    src = measure_loudness(master_dry)["integratedLufs"]
    gain_db = 0.0 if src is None or src <= -60.0 else target_lufs - src
    passes: list[dict[str, Any]] = []
    measured: dict[str, Any] = {}
    for _ in range(max_passes):
        gain_db = max(-12.0, min(24.0, gain_db))
        cmd = [ffmpeg_bin(), "-nostdin", "-y", "-v", "error", "-i", str(master_dry)]
        limiter = f"alimiter=limit={limit:.5f}:attack=5:release=50:level=false:latency=1"
        if bgm_path is not None:
            cmd += ["-stream_loop", "-1", "-i", str(bgm_path)]
            fade_st = max(0.0, duration - 1.4)
            flt = (
                f"[1:a]volume={bgm_volume},atrim=end={duration:.4f},apad=pad_dur=0.25,"
                f"afade=t=out:st={fade_st:.4f}:d=1.4,aformat=sample_rates=48000:channel_layouts=stereo[m];"
                f"[0:a]volume={gain_db:+.2f}dB,apad,atrim=end={duration:.4f},"
                "aresample=48000,aformat=sample_rates=48000:channel_layouts=stereo[voice];"
                f"[voice][m]amix=inputs=2:duration=first:normalize=0,{limiter}[a]"
            )
        else:
            flt = (
                f"[0:a]volume={gain_db:+.2f}dB,apad,atrim=end={duration:.4f},"
                f"aresample=48000,aformat=sample_rates=48000:channel_layouts=stereo,{limiter}[a]"
            )
        cmd += ["-filter_complex", flt, "-map", "[a]", "-t", f"{duration:.4f}", "-c:a", "pcm_s16le", str(output)]
        subprocess.run(cmd, check=True)
        measured = measure_loudness(output)
        lufs = measured["integratedLufs"]
        passes.append({"gainDb": round(gain_db, 2), "lufs": lufs, "truePeakDb": measured["truePeakDb"]})
        if lufs is None or abs(lufs - target_lufs) <= 0.15:
            break
        gain_db += target_lufs - lufs
    return {
        "targetLufs": target_lufs,
        "maxTruePeakDb": max_true_peak_db,
        "limiterDbfs": round(max_true_peak_db - 1.0, 2),
        "integratedLufs": measured.get("integratedLufs"),
        "truePeakDb": measured.get("truePeakDb"),
        "passes": passes,
        "wav": str(output),
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
