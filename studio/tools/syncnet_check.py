#!/usr/bin/env python3
"""SyncNet 逐窗音画口型偏移检测工具。

支持：
1. 单独检测 H3 原始/校时片段（960x960，24fps）
2. 通过 --crop x,y,w,h --seek S --duration D 直接从 1080p 成片裁出人物小窗复测
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.fft import dct
from scipy.ndimage import median_filter
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from SyncNetModel import S  # noqa: E402

torch.set_num_threads(4)
cv2.setNumThreads(2)


def default_model_path() -> Path:
    env = os.environ.get("SYNCNET_MODEL_PATH")
    if env:
        return Path(env)
    return HERE.parent.parent / "models" / "syncnet_v2.model"


def decode(
    path: Path,
    *,
    seek: float = 0.0,
    duration: float | None = None,
    crop: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    ffmpeg = os.environ.get("FFMPEG_BIN", "ffmpeg")
    args = [ffmpeg, "-nostdin", "-v", "error"]
    if seek > 0:
        args += ["-ss", f"{seek:.6f}"]
    args += ["-i", str(path)]
    if duration is not None:
        args += ["-t", f"{duration:.6f}"]

    vf_parts = []
    if crop is not None:
        x, y, w, h = crop
        vf_parts.append(f"crop={w}:{h}:{x}:{y}")
    vf_parts.append(
        "fps=25,scale=480:480:force_original_aspect_ratio=decrease,"
        "pad=480:480:(ow-iw)/2:(oh-ih)/2"
    )
    vf = ",".join(vf_parts)

    vid = subprocess.check_output(
        args + ["-an", "-vf", vf, "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    )
    aud = subprocess.check_output(
        args + ["-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"]
    )
    frames = np.frombuffer(vid, np.uint8).reshape(-1, 480, 480, 3)
    audio = np.frombuffer(aud, "<i2")
    return frames, audio


def crop_faces(frames: np.ndarray) -> np.ndarray:
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    known: dict[int, tuple[float, float, float]] = {}
    for i in range(0, len(frames), 4):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        boxes = detector.detectMultiScale(gray, 1.10, 4, minSize=(50, 50))
        if len(boxes):
            x, y, w, h = max(boxes, key=lambda p: p[2] * p[3])
            known[i] = (x + w / 2.0, y + h / 2.0, max(w, h) / 2.0)
    if len(known) < 8:
        raise ValueError(f"人脸检出帧数不足（仅 {len(known)} 帧，至少需要 8 帧）")
    coords = [
        median_filter(np.interp(np.arange(len(frames)), list(known), arr), size=9)
        for arr in np.asarray(list(known.values())).T
    ]
    out = []
    for frame, (x, y, s) in zip(frames, zip(*coords)):
        scale = 224.0 / (2.8 * s)
        m = np.array([[scale, 0.0, 112.0 - x * scale], [0.0, scale, (s - y) * scale]])
        out.append(cv2.warpAffine(frame, m, (224, 224), borderValue=(110, 110, 110)))
    return np.stack(out)


def mfcc(signal: np.ndarray) -> np.ndarray:
    x = signal.astype(np.float64)
    x = np.r_[x[0], x[1:] - 0.97 * x[:-1]]
    n = int(np.ceil((len(x) - 400) / 160)) + 1
    x = np.pad(x, (0, max(0, (n - 1) * 160 + 400 - len(x))))
    frames = np.lib.stride_tricks.sliding_window_view(x, 400)[::160][:n]
    power = abs(np.fft.rfft(frames, 512)) ** 2 / 512
    mel = np.linspace(0, 2595 * np.log10(1 + 8000 / 700), 28)
    bins = np.floor(513 * 700 * (10 ** (mel / 2595) - 1) / 16000).astype(int)
    filters = np.zeros((26, 257))
    for i, (a, b, c) in enumerate(zip(bins, bins[1:], bins[2:])):
        if b > a:
            filters[i, a:b] = np.arange(b - a) / (b - a)
        if c > b:
            filters[i, b:c] = np.arange(c - b, 0, -1) / (c - b)
    eps = np.finfo(float).eps
    feat = dct(np.log(np.maximum(power @ filters.T, eps)), type=2, axis=1, norm="ortho")[:, :13]
    feat *= 1 + 11 * np.sin(np.pi * np.arange(13) / 22)
    feat[:, 0] = np.log(np.maximum(power.sum(axis=1), eps))
    return feat.T


def load_model(model_path: Path) -> tuple[S, str]:
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到 SyncNet 权重文件: {model_path}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = S(num_layers_in_fc_layers=1024)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
    model.eval().to(device)
    return model, device


@torch.inference_mode()
def extract_features(
    model: S, device: str, frames: np.ndarray, audio: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    lip_frames = torch.as_tensor(crop_faces(frames).copy()).permute(3, 0, 1, 2).float()
    cc = torch.as_tensor(mfcc(audio).copy()).float().unsqueeze(0)
    n = min(lip_frames.shape[1] - 4, (cc.shape[2] - 20) // 4 + 1)
    if n <= 20:
        raise ValueError(f"可用音画帧过短（仅 {n} 帧），无法做 ±9 位移检测")
    v, a = [], []
    for start in range(0, n, 12):
        batch = list(range(start, min(n, start + 12)))
        v.append(
            model.forward_lip(torch.stack([lip_frames[:, i : i + 5] for i in batch]).to(device))
            .cpu()
            .numpy()
        )
        a.append(
            model.forward_aud(
                torch.stack([cc[:, :, 4 * i : 4 * i + 20] for i in batch]).to(device)
            )
            .cpu()
            .numpy()
        )
    return np.concatenate(v), np.concatenate(a)


def compute_metrics(v: np.ndarray, a: np.ndarray) -> dict:
    shifts = np.arange(-9, 10)
    ix = np.arange(9, len(v) - 9)
    d = np.stack([np.linalg.norm(v[ix] - a[ix - s], axis=1) for s in shifts])

    def measure(region: slice) -> dict:
        m = d[:, region].mean(axis=1)
        best = int(m.argmin())
        return {
            "picture_late_ms": int(shifts[best] * 40),
            "confidence": round(float(np.median(m) - m[best]), 3),
            "distance": round(float(m[best]), 3),
            "zero_distance": round(float(m[9]), 3),
        }

    windows = [
        {"at": round(float((ix[i] + 2) / 25.0), 2), **measure(slice(i, i + 50))}
        for i in range(0, max(0, len(ix) - 45), 30)
    ]
    if not windows and len(ix) > 0:
        windows = [{"at": round(float((ix[0] + 2) / 25.0), 2), **measure(slice(None))}]
    return {"overall": measure(slice(None)), "windows": windows}


def parse_crop(raw: str | None) -> tuple[int, int, int, int] | None:
    if not raw:
        return None
    parts = [int(p.strip()) for p in raw.split(",")]
    if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
        raise ValueError(f"--crop 格式必须为 x,y,w,h，收到: {raw}")
    return parts[0], parts[1], parts[2], parts[3]


def main() -> int:
    ap = argparse.ArgumentParser(description="SyncNet 口型同步逐窗测量")
    ap.add_argument("video", type=Path, help="视频路径")
    ap.add_argument("--seek", type=float, default=0.0, help="起始秒数")
    ap.add_argument("--duration", type=float, default=None, help="截取秒数")
    ap.add_argument("--crop", type=str, default=None, help="裁切人物小窗 x,y,w,h")
    ap.add_argument("--model", type=Path, default=default_model_path(), help="SyncNet 权重路径")
    args = ap.parse_args()

    crop = parse_crop(args.crop)
    model, device = load_model(args.model)
    frames, audio = decode(args.video, seek=args.seek, duration=args.duration, crop=crop)
    v, a = extract_features(model, device, frames, audio)
    result = compute_metrics(v, a)
    result["device"] = device
    result["file"] = str(args.video)
    if crop:
        result["crop"] = list(crop)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
