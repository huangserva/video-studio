"""项目清单（project.json）的读写、初始化与路径规范。"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .config import (
    DATA_DIR,
    DEFAULT_COVER_SEC,
    DEFAULT_H3_ENDPOINT,
    DEFAULT_QWEN_EDIT_ENDPOINT,
    DEFAULT_SENTENCE_GAP_SEC,
    DEFAULT_SPEED,
    DEFAULT_VOICE_ID,
    VOICE_DEFAULT_SPEED,
    DEFAULT_TAKES_PER_SENTENCE,
    DEFAULT_TARGET_LUFS,
    DEFAULT_TTS_ENDPOINT,
)


SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_recipe(recipe_id: str) -> dict[str, Any]:
    path = DATA_DIR / "recipes" / f"{recipe_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"找不到配方: {recipe_id} ({path})")
    return load_json(path)


def load_style(style_id: str) -> dict[str, Any]:
    path = DATA_DIR / "styles" / f"{style_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"找不到样式: {style_id} ({path})")
    return load_json(path)


def resolve_project_file(project_ref: str | Path) -> Path:
    p = Path(project_ref).expanduser().resolve()
    if p.is_dir():
        return p / "project.json"
    return p


def load_project(project_ref: str | Path) -> tuple[Path, dict[str, Any]]:
    pf = resolve_project_file(project_ref)
    if not pf.is_file():
        raise FileNotFoundError(f"找不到项目清单: {pf}")
    return pf.parent, load_json(pf)


def project_dirs(project_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": project_dir,
        "assets": project_dir / "assets",
        "narration": project_dir / "work" / "narration",
        "asr": project_dir / "work" / "asr",
        "h3_voice": project_dir / "work" / "h3" / "voice",
        "h3_graphs": project_dir / "work" / "h3" / "graphs",
        "h3_jobs": project_dir / "work" / "h3" / "jobs",
        "h3_raw": project_dir / "work" / "h3" / "raw",
        "h3_audit": project_dir / "work" / "h3" / "audit",
        "h3_accepted": project_dir / "work" / "h3" / "accepted",
        "shells": project_dir / "work" / "shells",
        "shots": project_dir / "work" / "shots",
        "subtitles": project_dir / "work" / "subtitles",
        "outputs": project_dir / "outputs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def init_project(
    project_dir: Path,
    *,
    title: str,
    subtitle: str = "",
    use_case: str = "U8",
    industry: str = "I20",
    recipe: str = "curator-h3-explainer",
    style: str = "museum-paper",
    voice_id: str = DEFAULT_VOICE_ID,
) -> Path:
    project_dir = project_dir.expanduser().resolve()
    if (project_dir / "project.json").exists():
        raise FileExistsError(f"项目已存在: {project_dir / 'project.json'}")
    load_recipe(recipe)
    load_style(style)
    project_dirs(project_dir)
    manifest: dict[str, Any] = {
        "schemaVersion": 1,
        "id": project_dir.name,
        "title": title,
        "subtitle": subtitle,
        "useCase": use_case,
        "industry": industry,
        "recipe": recipe,
        "style": style,
        "sections": ["开场", "方法", "实测", "结论"],
        "brief": {
            "goal": "",
            "audience": "",
            "platform": "B站/横屏",
            "facts": [],
            "mustInclude": [],
            "mustAvoid": [],
        },
        "voice": {
            "provider": "cosyvoice3",
            "endpoint": DEFAULT_TTS_ENDPOINT,
            "voiceId": voice_id,
            "speed": VOICE_DEFAULT_SPEED.get(voice_id, DEFAULT_SPEED),
            "sampleRate": 24000,
            "takesPerSentence": DEFAULT_TAKES_PER_SENTENCE,
            "sentenceGapSec": DEFAULT_SENTENCE_GAP_SEC,
            "targetLufs": DEFAULT_TARGET_LUFS,
        },
        "cover": {
            "enabled": True,
            "durationSec": DEFAULT_COVER_SEC,
            "image": "",
            "headline": title,
            "subline": subtitle,
        },
        "presenter": {
            "enabled": True,
            "name": "主讲人",
            "portraitAuthorized": False,
            "aiLabel": "AI 生成",
            "portraitImage": "assets/portrait.png",
            "waistImage": "assets/waist.png",
            "subjectDescription": "a Chinese documentary presenter in a clean studio backdrop",
            "h3Endpoint": DEFAULT_H3_ENDPOINT,
            "qwenEditEndpoint": DEFAULT_QWEN_EDIT_ENDPOINT,
            "remoteFolder": f"video-studio-{project_dir.name}",
            "seedBase": 7822077,
            # 默认 H3 原生出声（2026-11-26 用户拍板）：画面和声音同时生成，不配音锁音，走 `video-studio native`。
            # 要改回 CosyVoice 配音加锁音，把 h3Route 改成 aicg3d 或 native-lock。
            "h3Route": "native-voice",
            "voiceRef": "assets/voice-ref.wav",
            "guideImage": "assets/frame.png",
            "scenePrompt": (
                "an eye-level medium shot of <Subject 1> at the center of the frame in a clean studio, the camera still. "
                "<Subject 1> looks into the lens with a calm, sincere expression, the hands staying relaxed."
            ),
        },
        "bgm": {
            "path": "",
            "volumeUnderVoice": 0.085,
            "volumeOutro": 0.215,
        },
        "chapters": [
            {
                "id": "01",
                "title": "先看结果",
                "sectionIndex": 0,
                "narration": "先看结果。这里是第一章完整旁白，前五到十秒由数字人出镜，后面切到资料画面。",
                "onCameraText": "先看结果。这里是第一章开场出镜句。",
                "onCameraSec": 6.0,
                "heroLines": [title, subtitle or "核心结论"],
                "gestures": [
                    {
                        "start": 1.2,
                        "end": 2.6,
                        "phrase": "先看结果",
                        "action": "offer one relaxed open palm toward screen left below chest height",
                    }
                ],
                "cuts": [
                    {"at": 0.0, "layout": "split", "source": "assets/ch01-media.png"},
                    {"at": 6.0, "layout": "wide", "source": "assets/ch01-media.png"},
                ],
            }
        ],
    }
    return save_json(project_dir / "project.json", manifest)
