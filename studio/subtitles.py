"""中文字幕分句、ASR 字符级锚点对齐与 ASS 字幕生成。"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Any

from .project import load_json, load_project, load_style, project_dirs

ALNUM_ZH_RE = re.compile(r"[\w\u4e00-\u9fff]")
CLAUSE_RE = re.compile(r"[^，。；！？、]+[，。；！？、]?")


def normalize_text(text: str) -> str:
    return "".join(ALNUM_ZH_RE.findall(text.lower()))


def format_ass_timestamp(sec: float) -> str:
    centis = max(0, round(sec * 100))
    return f"{centis // 360000}:{(centis // 6000) % 60:02}:{(centis // 100) % 60:02}.{centis % 100:02}"


def split_phrases(text: str, max_chars: int = 24) -> list[str]:
    parts = CLAUSE_RE.findall(text.strip())
    out: list[str] = []
    buf = ""
    for part in parts:
        if buf and (len(buf + part) > max_chars or buf.endswith(("。", "！", "？"))):
            out.append(buf)
            buf = ""
        buf += part
    if buf:
        out.append(buf)
    return out


def _extract_asr_char_offsets(
    asr_payload: list[dict[str, Any]] | dict[str, Any] | None,
    duration_sec: float,
) -> tuple[list[str], list[float]]:
    """兼容 whisper.cpp (transcription/tokens) 与 Python whisper (segments/words) 两种 JSON 结构。"""
    chars: list[str] = []
    offsets: list[float] = []
    if not asr_payload:
        return chars, offsets

    segments: list[dict[str, Any]] = []
    if isinstance(asr_payload, list):
        segments = asr_payload
    elif isinstance(asr_payload, dict):
        if isinstance(asr_payload.get("transcription"), list):
            segments = asr_payload["transcription"]
        elif isinstance(asr_payload.get("segments"), list):
            segments = asr_payload["segments"]

    for segment in segments:
        tokens = segment.get("tokens")
        words = segment.get("words")
        if isinstance(tokens, list) and tokens and isinstance(tokens[0], dict):
            for token in tokens:
                val = str(token.get("text", ""))
                if val.startswith("[_"):
                    continue
                norm = normalize_text(val)
                if not norm:
                    continue
                off = token.get("offsets") or {}
                a = min(duration_sec, max(0.0, float(off.get("from", 0.0)) / 1000.0))
                b = min(duration_sec, max(a, float(off.get("to", a * 1000.0)) / 1000.0))
                for i, ch in enumerate(norm):
                    chars.append(ch)
                    offsets.append(a + (b - a) * i / len(norm))
        elif isinstance(words, list) and words:
            for w in words:
                norm = normalize_text(str(w.get("word", "")))
                if not norm:
                    continue
                a = min(duration_sec, max(0.0, float(w.get("start", 0.0))))
                b = min(duration_sec, max(a, float(w.get("end", a))))
                for i, ch in enumerate(norm):
                    chars.append(ch)
                    offsets.append(a + (b - a) * i / len(norm))
        elif "text" in segment and "start" in segment and "end" in segment:
            norm = normalize_text(str(segment.get("text", "")))
            if not norm:
                continue
            a = min(duration_sec, max(0.0, float(segment["start"])))
            b = min(duration_sec, max(a, float(segment["end"])))
            for i, ch in enumerate(norm):
                chars.append(ch)
                offsets.append(a + (b - a) * i / len(norm))
    return chars, offsets


def align_chapter_cues(
    *,
    narration: str,
    start_sec: float,
    duration_sec: float,
    asr_transcription: list[dict[str, Any]] | dict[str, Any] | None = None,
    max_chars: int = 24,
) -> tuple[list[dict[str, Any]], str]:
    """把分句对齐到实际语音时间。有 ASR 时按字符锚点对齐，无 ASR 时按字数比例兜底。"""
    phrases = split_phrases(narration, max_chars=max_chars)
    if not phrases:
        return [], "empty"

    target = normalize_text(narration)
    if not target:
        return [], "empty"

    anchors: dict[int, float] = {0: 0.0, len(target): duration_sec}
    mode = "proportional"

    if asr_transcription:
        chars, offsets = _extract_asr_char_offsets(asr_transcription, duration_sec)
        if chars:
            offsets.append(duration_sec)
            matcher = difflib.SequenceMatcher(None, target, "".join(chars), autojunk=False)
            for block in matcher.get_matching_blocks():
                for i in range(block.size):
                    anchors[block.a + i] = min(duration_sec, offsets[block.b + i])
            mode = "asr-anchored"

    keys = sorted(anchors)

    def lookup(idx: int) -> float:
        if idx in anchors:
            return anchors[idx]
        left = max(k for k in keys if k < idx)
        right = min(k for k in keys if k > idx)
        return anchors[left] + (anchors[right] - anchors[left]) * (idx - left) / (right - left)

    pos = 0
    timed: list[tuple[float, str]] = []
    for phrase in phrases:
        t = lookup(pos)
        if timed:
            t = max(t, timed[-1][0] + 0.47)
        if t >= duration_sec - 0.16 and timed:
            prev_t, prev_txt = timed.pop()
            timed.append((prev_t, prev_txt + phrase))
        else:
            timed.append((t, phrase))
        pos += len(normalize_text(phrase))

    cues: list[dict[str, Any]] = []
    for i, (cue_start, txt) in enumerate(timed):
        cue_end = timed[i + 1][0] if i + 1 < len(timed) else duration_sec
        # Clamp strictly inside chapter duration so subtitles never bleed across scenes.
        cue_start = min(duration_sec, max(0.0, cue_start))
        cue_end = min(duration_sec, max(cue_start + 0.05, cue_end))
        cues.append({
            "start": round(start_sec + cue_start, 4),
            "end": round(start_sec + cue_end, 4),
            "text": txt.rstrip("，。；、"),
        })
    return cues, mode


def build_ass_header(style: dict[str, Any]) -> str:
    canvas = style["canvas"]
    sub = style["subtitle"]
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {canvas['width']}\n"
        f"PlayResY: {canvas['height']}\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Narration,{sub['fontName']},{sub['fontSize']},{sub['primaryColor']},"
        f"{sub['primaryColor']},{sub['outlineColor']},{sub['backColor']},"
        f"0,0,0,0,100,100,0,0,3,10,0,2,108,108,{sub['marginV']},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )


def generate_subtitles(project_ref: str | Path) -> dict[str, Any]:
    project_dir, manifest = load_project(project_ref)
    dirs = project_dirs(project_dir)
    style = load_style(manifest.get("style", "museum-paper"))
    timeline = load_json(project_dir / "work" / "timeline.json")
    max_chars = int(style["subtitle"].get("maxCharsPerLine", 24))

    ch_by_id = {c["id"]: c for c in manifest["chapters"]}
    all_cues: list[dict[str, Any]] = []
    modes: dict[str, str] = {}

    for entry in timeline["chapters"]:
        cid = entry["id"]
        ch = ch_by_id[cid]
        asr_file = dirs["asr"] / f"{cid}.json"
        asr_data = load_json(asr_file) if asr_file.is_file() else None
        cues, mode = align_chapter_cues(
            narration=ch["narration"],
            start_sec=float(entry["start"]),
            duration_sec=float(entry["duration"]),
            asr_transcription=asr_data,
            max_chars=max_chars,
        )
        all_cues.extend(cues)
        modes[cid] = mode

    lines = [
        f"Dialogue: 0,{format_ass_timestamp(c['start'])},{format_ass_timestamp(c['end'])},"
        f"Narration,,0,0,0,,{c['text']}"
        for c in all_cues
    ]
    ass_path = dirs["subtitles"] / "narration.ass"
    ass_path.write_text(build_ass_header(style) + "\n".join(lines) + "\n", encoding="utf-8")
    return {
        "assPath": str(ass_path),
        "cueCount": len(all_cues),
        "alignmentModes": modes,
    }
