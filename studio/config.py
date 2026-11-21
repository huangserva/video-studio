"""环境路径与远程/本地服务端点解析。"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
BIN_DIR = ROOT / "bin"
TOOLS_DIR = ROOT / "studio" / "tools"
PROJECTS_DIR = ROOT / "projects"

DEFAULT_TTS_ENDPOINT = os.environ.get("VIDEO_STUDIO_TTS_ENDPOINT", "http://127.0.0.1:8001")
DEFAULT_H3_ENDPOINT = os.environ.get("VIDEO_STUDIO_H3_ENDPOINT", "http://127.0.0.1:8190")
DEFAULT_QWEN_EDIT_ENDPOINT = os.environ.get("VIDEO_STUDIO_QWEN_EDIT_ENDPOINT", "http://127.0.0.1:8192")


def ffmpeg_bin() -> str:
    return os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg"


def ffprobe_bin() -> str:
    return os.environ.get("FFPROBE_BIN") or shutil.which("ffprobe") or "ffprobe"


def syncnet_model_path() -> Path:
    env = os.environ.get("SYNCNET_MODEL_PATH")
    if env:
        return Path(env)
    return MODELS_DIR / "syncnet_v2.model"


def handcheck_bin() -> Path:
    return BIN_DIR / "handcheck"


def vision_python() -> str:
    """选择装有 torch + cv2 + scipy 的 Python，避免默认虚拟环境缺包。"""
    candidates = []
    if os.environ.get("VIDEO_STUDIO_PYTHON"):
        candidates.append(os.environ["VIDEO_STUDIO_PYTHON"])
    candidates.extend([
        "/opt/homebrew/bin/python3",
        sys.executable,
        shutil.which("python3") or "",
    ])
    seen = set()
    for cand in candidates:
        if not cand or cand in seen or not Path(cand).exists():
            continue
        seen.add(cand)
        proc = subprocess.run(
            [cand, "-c", "import torch, cv2, scipy"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if proc.returncode == 0:
            return cand
    return sys.executable
