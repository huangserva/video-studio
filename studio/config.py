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

# 配音与响度默认值（2026-11 起按 MemT 传播版的做法）
DEFAULT_SPEED = 1.3
# 新项目默认音色：qiuzhi2046-female-v3a（参考音频裁到完整句、最后一个字后淡出，2026-11-25 用户试听选定：节奏最好）。
# v3b（另过 htdemucs 去背景音乐）更干净但同 speed 下慢约 10%，要用时语速给 1.43。
DEFAULT_VOICE_ID = "qiuzhi2046-female-v3a"
VOICE_DEFAULT_SPEED = {"qiuzhi2046-female-v3a": 1.3, "qiuzhi2046-female-v3b": 1.43}
DEFAULT_TAKES_PER_SENTENCE = 3
DEFAULT_SENTENCE_GAP_SEC = 0.30
DEFAULT_HEAD_PAD_SEC = 0.12
DEFAULT_TAIL_PAD_SEC = 0.25
DEFAULT_SENTENCE_RMS_DB = -20.0
# 句首/句尾 2ms 内允许的最大样本跳变（满幅 1.0）。超过就是拼接时能听到的「啪」声
DEFAULT_EDGE_JUMP_MAX = 0.01
DEFAULT_TARGET_LUFS = -14.5
DEFAULT_LUFS_TOLERANCE = 0.5
DEFAULT_MAX_TRUE_PEAK_DB = -1.0
DEFAULT_COVER_SEC = 0.75


def ffmpeg_bin() -> str:
    return os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg"


def ffprobe_bin() -> str:
    return os.environ.get("FFPROBE_BIN") or shutil.which("ffprobe") or "ffprobe"


def syncnet_model_path() -> Path:
    env = os.environ.get("SYNCNET_MODEL_PATH")
    if env:
        return Path(env)
    return MODELS_DIR / "syncnet_v2.model"


def whisper_cli_bin() -> str | None:
    env = os.environ.get("VIDEO_STUDIO_WHISPER_CLI")
    if env:
        return env if Path(env).exists() else None
    return shutil.which("whisper-cli")


def whisper_model_path() -> Path | None:
    """whisper.cpp 模型：环境变量优先，其次 models/ggml-small.bin。"""
    candidates = []
    if os.environ.get("VIDEO_STUDIO_WHISPER_MODEL"):
        candidates.append(Path(os.environ["VIDEO_STUDIO_WHISPER_MODEL"]).expanduser())
    candidates += [
        MODELS_DIR / "ggml-small.bin",
        Path.home() / ".cache" / "whisper.cpp" / "ggml-small.bin",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


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
