"""H3 原生出声路线（2026-11-26 起新项目默认）：H3 同时生成画面和声音，挂同一条音色参考，不再配音锁音。

三步，都由 `video-studio native` 驱动：
1. 分块计划（`--plan`，不碰 GPU）：每章按整句切块，不跨章，每块不超过 maxChars 字；
   时长 = baseSec + 字数 × secPerChar（默认 0.4 + 字数 × 0.2067 s，约每秒 4.8 字），取整到 17k+5 帧。
   语速只靠时长控制，提示词里写快写慢基本没用。
2. 生成并逐块验收（可断点续跑，状态在 work/h3/native/state.json）：whisper 回读相似度 >= asrMin、
   片尾留白 >= tailMinSec、基频在参考 ± f0ToleranceHz 以内（或 f0RangeHz）、sync-gate 以这块自带的音轨为准；
   不过就换 seed，最多 maxTries 次。4090 共用：只在队列没有排队任务时提交，不清队列。
3. 拼接与逐字对齐（`--assemble`）：每章把验收块按整帧（24 fps）首尾相接，块间留 gapSec（取整到整帧），
   写出 work/narration/<章>.wav、逐句时间（<章>.sentences.json）、whisper 时间戳（work/asr/<章>.json），
   再汇总 timeline.json，并把同一章的验收块拼成 work/h3/accepted/<章>.mp4（章内时刻 = 片内时刻，即取帧 PAD=0）。
   之后照常 subtitles → compose → qc --windows。

参考实现：projects/long-march-01/work/native_full.py、native_assemble.py。
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode

import numpy as np

from .config import ffmpeg_bin, whisper_cli_bin, whisper_model_path
from .h3 import (
    ComfyEndpoint,
    build_h3_native_voice_graph,
    build_h3_native_voice_prompt,
    h3_route,
    native_settings,
)
from .project import load_json, load_project, load_recipe, project_dirs, save_json, sha256_file
from .sync_gate import measure_syncnet, review_clip, video_nframes
from .tts import (
    _shift_transcription,
    asr_similarity,
    build_audio_timeline,
    load_mono,
    pair_sentences,
    transcribe_wavs,
    transcription_text,
    write_float,
)

H3_FPS = 24
SR = 48000
HAN_RE = re.compile(r"[一-鿿]")

NATIVE_VOICE_DEFAULTS: dict[str, Any] = {
    "baseSec": 0.4,
    "secPerChar": 0.2067,
    "maxChars": 62,
    "maxFrames": 345,  # H3 单段上限 15 s 以内最大的 17k+5 档
    "speechStartSec": 0.5,
    "gapSec": 0.10,  # 章内块间留白，取整到整帧
    "chapterGapSec": 0.40,  # 换章留白
    "tailSec": 0.8,  # 全片结尾留白（加在最后一章）
    "asrMin": 0.75,
    "tailMinSec": 0.06,
    "f0ToleranceHz": 9.0,
    "f0RangeHz": None,  # 写成 [80, 95] 时优先于「参考 ± 容差」
    "maxTries": 3,
    "pollSec": 20,
}


# --------------------------------------------------------------------------
# 设置与分块计划
# --------------------------------------------------------------------------


def native_voice_settings(manifest: dict[str, Any], recipe: dict[str, Any]) -> dict[str, Any]:
    """默认值 ← 配方 hostSegment.nativeVoice ← 项目 presenter.h3NativeVoice。"""
    host = recipe.get("hostSegment") or {}
    presenter = manifest.get("presenter") or {}
    return {**NATIVE_VOICE_DEFAULTS, **(host.get("nativeVoice") or {}), **(presenter.get("h3NativeVoice") or {})}


def count_chars(text: str) -> int:
    """念法稿里的汉字数（语速按汉字算；念法稿应已把数字、英文写成读法）。"""
    return len(HAN_RE.findall(text))


def frames_for(chars: int, *, base_sec: float, sec_per_char: float) -> int:
    """时长 = base + 字数 × 每字秒数，换算成 24 fps 帧数后就近取整到 17k+5。"""
    want = (base_sec + chars * sec_per_char) * H3_FPS
    return 17 * max(0, round((want - 5) / 17)) + 5


def chunk_key(text: str, frames: int) -> str:
    """块的身份由台词和帧数决定：改稿后没变的块沿用已验收结果，块号顺延也不影响。"""
    return hashlib.sha1(f"{text}|{frames}".encode("utf-8")).hexdigest()[:12]


def plan_chunks(chapters: list[dict[str, Any]], settings: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """按章内整句贪心切块：不跨章，每块不超过 maxChars 字（单句超长时独占一块）。

    lines 是全片句号（从 1 开始），sentences 是章内句号。mode 只是排版提示：
    chapters[].hostMode 写了就用，否则首章和末章 full（整屏），其余 corner（小窗）。
    """
    cfg = {**NATIVE_VOICE_DEFAULTS, **(settings or {})}
    max_chars = int(cfg["maxChars"])
    out: list[dict[str, Any]] = []
    line_no = 0
    last = len(chapters) - 1
    for ci, ch in enumerate(chapters):
        mode = ch.get("hostMode") or ("full" if ci in (0, last) else "corner")
        groups: list[list[dict[str, Any]]] = []
        cur: list[dict[str, Any]] = []
        for pair in pair_sentences(ch):
            line_no += 1
            item = {**pair, "line": line_no, "chars": count_chars(pair["tts"])}
            if cur and sum(x["chars"] for x in cur) + item["chars"] > max_chars:
                groups.append(cur)
                cur = []
            cur.append(item)
        if cur:
            groups.append(cur)
        for g in groups:
            n = sum(x["chars"] for x in g)
            frames = frames_for(n, base_sec=float(cfg["baseSec"]), sec_per_char=float(cfg["secPerChar"]))
            text = "".join(x["tts"] for x in g)
            if frames > int(cfg["maxFrames"]):
                raise ValueError(
                    f"第 {ch['id']} 章第 {g[0]['index']} 句 {n} 字，要 {frames / H3_FPS:.2f}s，超过 H3 单段上限；把这句拆短。"
                )
            out.append({
                "id": f"k{len(out):02d}",
                "key": chunk_key(text, frames),
                "chapter": ch["id"],
                "sentences": [x["index"] for x in g],
                "lines": [x["line"] for x in g],
                "part": ch.get("part", ch["id"]),
                "mode": mode,
                "chars": n,
                "frames": frames,
                "sec": round(frames / H3_FPS, 4),
                "text": text,
                "display": "".join(x["text"] for x in g),
            })
    return out


def plan_project(project_ref: str | Path) -> dict[str, Any]:
    project_dir, manifest = load_project(project_ref)
    recipe = load_recipe(manifest.get("recipe", "curator-h3-explainer"))
    cfg = native_voice_settings(manifest, recipe)
    chunks = plan_chunks(manifest["chapters"], cfg)
    nd = native_dir(project_dir)
    save_json(nd / "plan.json", chunks)
    return {
        "plan": str(nd / "plan.json"),
        "chunks": [{k: c[k] for k in ("id", "key", "chapter", "mode", "lines", "chars", "frames", "sec")} for c in chunks],
        "count": len(chunks),
        "speechSec": round(sum(c["sec"] for c in chunks), 1),
        "settings": {k: cfg[k] for k in ("baseSec", "secPerChar", "maxChars")},
    }


def native_dir(project_dir: Path) -> Path:
    d = project_dir / "work" / "h3" / "native"
    for sub in ("raw", "acc", "wav", "audit"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------
# 逐块验收的纯函数部分
# --------------------------------------------------------------------------


def f0_median(x: np.ndarray, sr: int = SR) -> float:
    """有声帧（RMS >= 0.02）自相关基频的中位数，60–300 Hz；测不出来返回 0。"""
    hop, win = int(0.01 * sr), int(0.04 * sr)
    lo, hi = int(sr / 300), int(sr / 60)
    out: list[float] = []
    for i in range(0, len(x) - win, hop):
        s = x[i:i + win].astype(np.float64)
        if np.sqrt((s ** 2).mean()) < 0.02:
            continue
        s = s - s.mean()
        n = 1 << (2 * win - 1).bit_length()
        spec = np.fft.rfft(s, n)
        ac = np.fft.irfft(spec * np.conj(spec), n)[:win]
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.45 * ac[0]:
            out.append(sr / k)
    return float(np.median(out)) if out else 0.0


def tail_silence_sec(x: np.ndarray, sr: int = SR, *, threshold: float = 0.05) -> float:
    """最后一个有声 10 ms 帧（RMS > threshold）之后到片尾的秒数；全片无声返回 0。"""
    hop = int(0.01 * sr)
    n = len(x) // hop
    if n == 0:
        return 0.0
    e = np.sqrt((x[: n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(axis=1))
    on = np.where(e > threshold)[0]
    if not len(on):
        return 0.0
    return round(len(x) / sr - (on[-1] + 1) * hop / sr, 3)


def f0_range(cfg: dict[str, Any], ref_f0: float | None) -> tuple[float, float]:
    if cfg.get("f0RangeHz"):
        lo, hi = cfg["f0RangeHz"]
        return float(lo), float(hi)
    if not ref_f0:
        raise ValueError("测不出音色参考的基频，请在 presenter.h3NativeVoice.f0RangeHz 里写明范围，例如 [80, 95]")
    tol = float(cfg["f0ToleranceHz"])
    return round(ref_f0 - tol, 1), round(ref_f0 + tol, 1)


def chunk_problems(metrics: dict[str, Any], cfg: dict[str, Any], f0_lohi: tuple[float, float]) -> list[str]:
    """逐块验收：任何一条不过就换 seed。"""
    probs = []
    if metrics["sim"] < float(cfg["asrMin"]):
        probs.append(f"回读相似度 {metrics['sim']}")
    if metrics["tail"] < float(cfg["tailMinSec"]):
        probs.append(f"片尾只剩 {metrics['tail']}s 疑似截字")
    lo, hi = f0_lohi
    if not lo <= metrics["f0"] <= hi:
        probs.append(f"基频 {metrics['f0']}Hz 不在 {lo}–{hi}Hz，音色漂")
    if not metrics.get("syncAccepted"):
        probs.append(f"口型未过（{metrics.get('syncReason')}）")
    return probs


# --------------------------------------------------------------------------
# 逐字对齐
# --------------------------------------------------------------------------


def heard_chars(transcription: list[dict[str, Any]]) -> list[tuple[str, float]]:
    """whisper.cpp tokens → [(汉字, 开口时刻秒)]；一个 token 含多个字时在 token 内均分。"""
    heard: list[tuple[str, float]] = []
    for seg in transcription or []:
        for t in seg.get("tokens") or []:
            if str(t.get("text", "")).strip().startswith("["):
                continue
            off = t.get("offsets") or {}
            a, b = float(off.get("from", 0)) / 1000, float(off.get("to", 0)) / 1000
            units = HAN_RE.findall(str(t.get("text", "")))
            for i, u in enumerate(units):
                heard.append((u, a + (b - a) * i / max(1, len(units))))
    return heard


def align_chars(text: str, heard: list[tuple[str, float]]) -> tuple[list[str], list[float], float]:
    """念法稿逐字 → 开口时刻。按顺序匹配回读（同音错字对不上的靠前后已知字插值），时刻单调不减。

    返回（逐字、逐字时刻、对上比例）。一个字也对不上时报错。
    """
    want = HAN_RE.findall(text)
    if not want:
        return [], [], 1.0
    times: list[float | None] = [None] * len(want)
    sm = SequenceMatcher(None, want, [h[0] for h in heard], autojunk=False)
    for a, b, n in sm.get_matching_blocks():
        for j in range(n):
            times[a + j] = heard[b + j][1]
    known = [(j, t) for j, t in enumerate(times) if t is not None]
    if not known:
        raise RuntimeError("回读一个字也对不上念法稿")
    end = heard[-1][1] + 0.2
    for j in range(len(times)):
        if times[j] is None:
            p = max((k for k in known if k[0] < j), default=(-1, known[0][1] - 0.2), key=lambda k: k[0])
            q = min((k for k in known if k[0] > j), default=(len(times), end), key=lambda k: k[0])
            times[j] = p[1] + (q[1] - p[1]) * (j - p[0]) / (q[0] - p[0])
    out = [float(t) for t in times]  # type: ignore[arg-type]
    for j in range(1, len(out)):
        out[j] = max(out[j], out[j - 1])
    return want, out, len(known) / len(want)


def sentence_spans(
    sentence_chars: list[int], times: list[float], duration: float, *, lead: float = 0.05, tail: float = 0.28,
) -> list[tuple[float, float, list[float]]]:
    """按每句字数切逐字时刻，得每句（起、止、本句逐字时刻）。起 = 首字 - lead，止 = 末字 + tail，夹在块内。"""
    spans = []
    k = 0
    for n in sentence_chars:
        tt = times[k:k + n]
        k += n
        if not tt:
            raise ValueError("有一句没有汉字，无法逐字对齐")
        s0 = max(0.0, tt[0] - lead)
        s1 = min(duration, tt[-1] + tail)
        spans.append((round(s0, 4), round(max(s1, s0), 4), tt))
    return spans


# --------------------------------------------------------------------------
# 生成与验收（需要 4090）
# --------------------------------------------------------------------------


def _log(nd: Path, msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, file=sys.stderr, flush=True)
    with (nd / "run.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _wait_queue_free(e: ComfyEndpoint, *, poll: int = 30) -> None:
    """4090 共用：等队列里没有排队任务再提交（只排在别人后面，不清队列、不打断别人）。"""
    while True:
        try:
            if not json.loads(e.request("/queue")).get("queue_pending"):
                return
        except Exception:  # noqa: BLE001 跳板机偶发断线，等一会再看
            pass
        time.sleep(poll)


def _queue_idle(e: ComfyEndpoint) -> bool:
    try:
        q = json.loads(e.request("/queue"))
    except Exception:  # noqa: BLE001
        return False
    return not q.get("queue_pending") and not q.get("queue_running")


def _extract_wav(src: Path, wav: Path, sr: int = SR) -> Path:
    subprocess.run(
        [ffmpeg_bin(), "-nostdin", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(sr), str(wav)],
        check=True,
    )
    return wav


def _route_guard(manifest: dict[str, Any], recipe: dict[str, Any]) -> None:
    route = h3_route(manifest.get("presenter") or {}, recipe.get("hostSegment") or {})
    if route != "native-voice":
        raise RuntimeError(
            f"这个项目的 H3 路线是 {route}，不是 native-voice。原生出声会改写 work/narration 和 project.json 的 onCameraSec；"
            "确定要走这条路线，请先把 presenter.h3Route 设成 \"native-voice\"（--plan 不受限制）。"
        )


def _guide(project_dir: Path, presenter: dict[str, Any]) -> tuple[Path, bool]:
    rel = presenter.get("guideVideo")
    if rel:
        return project_dir / rel, True
    return project_dir / (presenter.get("guideImage") or presenter["waistImage"]), False


def check_chunk(
    raw: Path,
    chunk: dict[str, Any],
    *,
    nd: Path,
    cfg: dict[str, Any],
    f0_lohi: tuple[float, float],
    gates: dict[str, Any] | None,
    ai_label: str,
) -> tuple[list[str], dict[str, Any]]:
    """一块的全部验收：回读、片尾、基频、sync-gate（以自身音轨为准，允许整帧移画）。"""
    wav = _extract_wav(raw, nd / "wav" / f"{raw.stem}.wav")
    x = load_mono(wav, SR)
    tr = (transcribe_wavs([wav]) or {}).get(str(wav))
    txt = transcription_text(tr) if tr else ""
    acc = nd / "acc" / f"{chunk['key']}.mp4"
    r = review_clip(
        candidate_mp4=raw, dry_wav=wav, accepted_output=acc, audit_dir=nd / "audit", ai_label=ai_label, gates=gates,
    )
    metrics = {
        "asr": txt,
        "sim": round(asr_similarity(txt, chunk["display"], chunk["text"]), 3) if txt else 0.0,
        "tail": tail_silence_sec(x),
        "f0": round(f0_median(x), 1),
        "syncAccepted": bool(r.get("accepted")),
        "syncReason": r.get("reason"),
        "verdict": r.get("verdict"),
        "shift": r.get("advanceFrames"),
        "raw": [w["picture_late_ms"] for w in (r.get("rawSync") or {}).get("windows", [])],
    }
    probs = chunk_problems(metrics, cfg, f0_lohi)
    if not probs:
        metrics["minconf"] = round(min(w["confidence"] for w in r["acceptedSync"]["windows"]), 2)
    else:
        acc.unlink(missing_ok=True)
        acc.with_suffix(".receipt.json").unlink(missing_ok=True)
    return probs, metrics


def run_native(
    project_ref: str | Path,
    *,
    chunk_ids: list[str] | None = None,
    pilot: bool = False,
    redo: list[str] | None = None,
    endpoint_factory: Callable[[str], ComfyEndpoint] = ComfyEndpoint,
) -> dict[str, Any]:
    """生成并逐块验收。可中断续跑：已过的块不再生成；排队中的块接着等结果。"""
    project_dir, manifest = load_project(project_ref)
    recipe = load_recipe(manifest.get("recipe", "curator-h3-explainer"))
    _route_guard(manifest, recipe)
    presenter = manifest["presenter"]
    host_cfg = recipe.get("hostSegment") or {}
    cfg = native_voice_settings(manifest, recipe)
    nd = native_dir(project_dir)
    plan = plan_chunks(manifest["chapters"], cfg)
    save_json(nd / "plan.json", plan)

    if not whisper_cli_bin() or not whisper_model_path():
        raise RuntimeError("没装 whisper-cli 或模型，原生出声无法逐块回读验收（禁止假绿）。先跑 doctor 补齐。")
    ref_rel = presenter.get("voiceRef")
    ref_local = project_dir / ref_rel if ref_rel else None
    if ref_local is None or not ref_local.is_file():
        raise FileNotFoundError(
            f"缺少音色参考 presenter.voiceRef（{ref_local}）：约 10 秒、一两句完整的干净整句，全片每块都挂同一条。"
        )
    guide_local, guide_is_video = _guide(project_dir, presenter)
    if not guide_local.is_file():
        raise FileNotFoundError(f"缺少首帧引导: {guide_local}")

    ref_f0 = round(f0_median(load_mono(ref_local, SR)), 1)
    lohi = f0_range(cfg, ref_f0)

    state_path = nd / "state.json"
    state: dict[str, Any] = load_json(state_path) if state_path.is_file() else {}
    by_id = {c["id"]: c for c in plan}
    for cid in redo or []:
        c = by_id.get(cid)
        if c is None:
            raise KeyError(f"计划里没有块 {cid}")
        old = state.pop(c["key"], None)
        if old:
            state.setdefault("_history", []).append({"key": c["key"], **old})
        (nd / "acc" / f"{c['key']}.mp4").unlink(missing_ok=True)

    targets = [c for c in plan if not chunk_ids or c["id"] in chunk_ids]
    unknown = set(chunk_ids or []) - set(by_id)
    if unknown:
        raise KeyError(f"计划里没有块 {', '.join(sorted(unknown))}")
    accepted_any = any(isinstance(v, dict) and v.get("status") == "ok" for k, v in state.items() if not k.startswith("_"))
    if pilot:
        targets = targets[:1]
    elif len(targets) > 1 and not accepted_any:
        raise RuntimeError("样片先行：还没有任何一块通过验收。先加 --pilot 跑第 1 块，人工试听确认后再批量生成。")

    folder = presenter.get("remoteFolder") or f"video-studio-{project_dir.name}"
    folder = f"{folder}-native"
    seed_base = int(presenter.get("seedBase", host_cfg.get("seedBase", 7822077)))
    stride = int(host_cfg.get("seedRetryStride", 104729))
    max_tries = int(cfg["maxTries"])
    graph_cfg = native_settings(presenter, host_cfg)
    ai_label = "" if presenter.get("aiLabelOnScreen") is False else (presenter.get("aiLabel") or "")

    e = endpoint_factory(presenter["h3Endpoint"])
    remote_guide = e.upload(guide_local, subfolder=folder, filename=f"guide{guide_local.suffix}")
    remote_ref = e.upload(ref_local, subfolder=folder, filename=f"voice-ref{ref_local.suffix}")
    state["_meta"] = {"voiceRef": str(ref_rel), "voiceRefSha256": sha256_file(ref_local), "refF0": ref_f0, "f0Range": list(lohi)}
    save_json(state_path, state)
    _log(nd, f"开始：{len(targets)} 块，音色参考基频 {ref_f0} Hz，基频验收 {lohi[0]}–{lohi[1]} Hz")

    for c in targets:
        rec = state.setdefault(c["key"], {"id": c["id"], "text": c["text"], "tries": []})
        rec["id"] = c["id"]
        if rec.get("status") == "ok" and not (nd / "acc" / f"{c['key']}.mp4").is_file():
            rec["status"] = "retry"  # 验收片被删了就重做
        while rec.get("status") not in ("ok", "failed"):
            if rec.get("status") != "queued":
                _wait_queue_free(e)
                seed = seed_base + len(rec["tries"]) * stride
                scene = (next(ch for ch in manifest["chapters"] if ch["id"] == c["chapter"]).get("scenePrompt")
                         or presenter.get("scenePrompt") or "")
                prompt = build_h3_native_voice_prompt(
                    subject=presenter.get("subjectDescription") or "",
                    scene=scene,
                    seconds=c["sec"],
                    text=c["text"],
                    multi_sentence=len(c["sentences"]) > 1,
                    has_voice_ref=True,
                    speech_start=float(cfg["speechStartSec"]),
                    **{k: cfg[k] for k in ("style", "pace", "soundscape") if cfg.get(k)},
                    **({"language": graph_cfg["language"]} if graph_cfg.get("language") else {}),
                    **({"dialogue_tag": cfg["dialogueTag"]} if cfg.get("dialogueTag") else {}),
                )
                (nd / "audit" / f"{c['key']}.prompt.txt").write_text(prompt, encoding="utf-8")
                graph = build_h3_native_voice_graph(
                    remote_voice_ref=remote_ref,
                    remote_guide=remote_guide,
                    guide_is_video=guide_is_video,
                    prompt_text=prompt,
                    frames=c["frames"],
                    seed=seed,
                    filename_prefix=f"{folder}/{c['key']}-s{seed}",
                    client_id=f"{folder}-{c['key']}",
                    settings=graph_cfg,
                )
                resp = json.loads(e.request("/prompt", json.dumps(graph, ensure_ascii=False).encode("utf-8")))
                if resp.get("node_errors") or not resp.get("prompt_id"):
                    _log(nd, f"{c['id']} 提交被拒 {json.dumps(resp, ensure_ascii=False)[:800]}")
                    raise RuntimeError(f"ComfyUI 拒绝提交 {c['id']}: {resp}")
                rec["tries"].append({"seed": seed, "pid": resp["prompt_id"]})
                rec["status"] = "queued"
                save_json(state_path, state)
                _log(nd, f"{c['id']} 入队 seed={seed} {c['sec']:.2f}s {c['chars']}字 第{c['chapter']}章句{c['sentences']}")
            tr = rec["tries"][-1]
            time.sleep(int(cfg["pollSec"]))
            try:
                hist = json.loads(e.request(f"/history/{tr['pid']}")).get(tr["pid"])
            except Exception:  # noqa: BLE001 跳板机断线，下一轮再查
                continue
            if not hist:
                continue
            if hist.get("status", {}).get("status_str") != "success":
                _log(nd, f"{c['id']} 生成失败 {json.dumps(hist.get('status'), ensure_ascii=False)[:300]}")
                tr["error"] = hist.get("status")
                rec["status"] = "failed" if len(rec["tries"]) >= max_tries else "retry"
                save_json(state_path, state)
                continue
            item = next(
                it for recs in hist["outputs"]["91"].values() if isinstance(recs, list)
                for it in recs if isinstance(it, dict) and str(it.get("filename", "")).endswith(".mp4")
            )
            raw = nd / "raw" / f"{c['key']}-s{tr['seed']}.mp4"
            if not raw.is_file():
                tmp = raw.with_suffix(".tmp.mp4")
                tmp.write_bytes(e.request("/view?" + urlencode({k: item[k] for k in ("filename", "subfolder", "type")}), timeout=300))
                tmp.replace(raw)
            probs, info = check_chunk(raw, c, nd=nd, cfg=cfg, f0_lohi=lohi, gates=recipe.get("gates"), ai_label=ai_label)
            tr.update(info)
            if not probs:
                rec["status"] = "ok"
                _log(nd, f"{c['id']} 验收通过 移{info['shift']}帧 最低置信{info['minconf']} 基频{info['f0']} "
                         f"片尾{info['tail']}s 回读{info['sim']}")
            elif len(rec["tries"]) < max_tries:
                rec["status"] = "retry"
                _log(nd, f"{c['id']} 未过：{'；'.join(probs)}，换 seed。回读：{info['asr']}")
            else:
                rec["status"] = "failed"
                _log(nd, f"{c['id']} {max_tries} 个 seed 都没过：{'；'.join(probs)}，需人工处理")
            tr["problems"] = probs
            save_json(state_path, state)

    if _queue_idle(e):
        try:
            e.request("/free", json.dumps({"unload_models": True, "free_memory": True}).encode())
            _log(nd, "本批完成，队列空闲，已释放 H3 显存")
        except Exception:  # noqa: BLE001
            pass
    else:
        _log(nd, "本批完成；队列里还有别人的任务，不释放显存")

    summary = [
        {"id": c["id"], "key": c["key"], "status": state.get(c["key"], {}).get("status", "pending"),
         "tries": len(state.get(c["key"], {}).get("tries", []))}
        for c in plan
    ]
    return {
        "pilot": pilot,
        "state": str(state_path),
        "chunks": summary,
        "ok": sum(1 for s in summary if s["status"] == "ok"),
        "failed": [s["id"] for s in summary if s["status"] == "failed"],
        "pending": [s["id"] for s in summary if s["status"] not in ("ok", "failed")],
        "note": "样片通过后请人工试听，确认音色和语速再批量跑；交片时提醒用户连着听几块（块间音色会有轻微起伏）。"
        if pilot else "全部通过后跑 native --assemble。",
    }


# --------------------------------------------------------------------------
# 拼接与逐字对齐
# --------------------------------------------------------------------------


def _fit_len(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.concatenate([x, np.zeros(n - len(x), dtype=x.dtype)])


def _build_host_clip(
    chunks: list[tuple[Path, int, int]], audio_wav: Path, target: Path, total_frames: int,
) -> Path:
    """把同一章的验收块按整帧首尾相接：每块后面用最后一帧补够留白帧数（嘴已闭合）。音轨挂本章人声。"""
    cmd = [ffmpeg_bin(), "-nostdin", "-v", "error", "-y"]
    parts = []
    for i, (path, _frames, pad) in enumerate(chunks):
        cmd += ["-i", str(path)]
        parts.append(f"[{i}:v]fps=24,setsar=1,tpad=stop_mode=clone:stop={pad}[v{i}]" if pad else f"[{i}:v]fps=24,setsar=1[v{i}]")
    cmd += ["-i", str(audio_wav)]
    flt = ";".join(parts) + ";" + "".join(f"[v{i}]" for i in range(len(chunks))) + f"concat=n={len(chunks)}:v=1:a=0[v]"
    tmp = target.with_suffix(".tmp.mp4")
    cmd += [
        "-filter_complex", flt, "-map", "[v]", "-map", f"{len(chunks)}:a:0", "-frames:v", str(total_frames),
        "-c:v", "libx264", "-preset", "medium", "-crf", "14", "-pix_fmt", "yuv420p", "-r", "24",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(tmp),
    ]
    subprocess.run(cmd, check=True)
    got = video_nframes(tmp)
    if got != total_frames:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{target.name} 帧数 {got}，应为 {total_frames}")
    tmp.replace(target)
    return target


def assemble_native(project_ref: str | Path, *, measure_sync: bool = True) -> dict[str, Any]:
    """拼人声、逐字对齐，写出 compose/subtitles 要的全部文件。"""
    project_dir, manifest = load_project(project_ref)
    recipe = load_recipe(manifest.get("recipe", "curator-h3-explainer"))
    _route_guard(manifest, recipe)
    dirs = project_dirs(project_dir)
    cfg = native_voice_settings(manifest, recipe)
    nd = native_dir(project_dir)
    plan = plan_chunks(manifest["chapters"], cfg)
    state = load_json(nd / "state.json") if (nd / "state.json").is_file() else {}
    missing = [c["id"] for c in plan
               if state.get(c["key"], {}).get("status") != "ok" or not (nd / "acc" / f"{c['key']}.mp4").is_file()]
    if missing:
        raise RuntimeError(f"还有块没通过验收：{', '.join(missing)}（先跑 native，失败的块用 --redo 重做或人工处理）")

    gap_frames = max(1, round(float(cfg["gapSec"]) * H3_FPS))
    tail_frames = max(0, round(float(cfg["tailSec"]) * H3_FPS))
    spf = SR // H3_FPS  # 每帧 2000 个采样
    chapters = manifest["chapters"]
    acc_wavs = {c["key"]: _extract_wav(nd / "acc" / f"{c['key']}.mp4", nd / "wav" / f"{c['key']}.acc.wav") for c in plan}
    asr = transcribe_wavs(list(acc_wavs.values()))
    if asr is None:
        raise RuntimeError("没装 whisper-cli 或模型，无法逐字对齐")
    chunk_rows: list[dict[str, Any]] = []
    sent_rows: list[dict[str, Any]] = []
    host_specs: dict[str, tuple[list[tuple[Path, int, int]], int]] = {}

    for ci, ch in enumerate(chapters):
        cid = ch["id"]
        mine = [c for c in plan if c["chapter"] == cid]
        pairs = {p["index"]: p for p in pair_sentences(ch)}
        audio: list[np.ndarray] = []
        clips: list[tuple[Path, int, int]] = []
        sentences: list[dict[str, Any]] = []
        transcription: list[dict[str, Any]] = []
        cursor = 0  # 章内帧
        for k, c in enumerate(mine):
            acc = nd / "acc" / f"{c['key']}.mp4"
            frames = video_nframes(acc)
            wav = acc_wavs[c["key"]]
            x = _fit_len(load_mono(wav, SR), frames * spf)
            dur = frames / H3_FPS
            tr = asr.get(str(wav)) or []
            want, times, cover = align_chars(c["text"], heard_chars(tr))
            spans = sentence_spans([count_chars(pairs[i]["tts"]) for i in c["sentences"]], times, dur)
            a = cursor / H3_FPS
            for idx, (s0, s1, tt) in zip(c["sentences"], spans):
                p = pairs[idx]
                chars = HAN_RE.findall(p["tts"])
                sentences.append({
                    "index": idx, "text": p["text"], "tts": p["tts"], "chunk": c["id"], "chunkKey": c["key"],
                    "startSec": round(a + s0, 4), "endSec": round(a + s1, 4),
                    "charTimes": [[ch_, round(a + t, 3)] for ch_, t in zip(chars, tt)],
                })
            transcription += _shift_transcription(tr, int(round(a * 1000)))
            last_chunk = k == len(mine) - 1
            pad = (tail_frames if ci == len(chapters) - 1 else 0) if last_chunk else gap_frames
            audio += [x, np.zeros(pad * spf, dtype=x.dtype)]
            clips.append((acc, frames, pad))
            chunk_rows.append({"id": c["id"], "key": c["key"], "chapter": cid, "mode": c["mode"], "lines": c["lines"],
                               "localA": round(a, 4), "localB": round(a + dur, 4), "frames": frames,
                               "charCoverage": round(cover, 3)})
            cursor += frames + pad
        chapter_wav = dirs["narration"] / f"{cid}.wav"
        write_float(chapter_wav, np.concatenate(audio), SR)
        if transcription:
            save_json(dirs["asr"] / f"{cid}.json", {"source": "video-studio H3 native voice ASR (whisper.cpp)",
                                                    "transcription": transcription})
        save_json(dirs["narration"] / f"{cid}.sentences.json", {
            "chapterId": cid, "mode": "h3-native-voice", "sampleRate": SR, "durationSec": round(cursor / H3_FPS, 4),
            "sentences": sentences,
        })
        # 整章都是出镜段（画面和声音同源）；向上取到 4 位小数，切出镜干声时不会少采样
        ch["onCameraSec"] = round(cursor / H3_FPS + 0.00005, 4)
        host_specs[cid] = (clips, cursor)
        sent_rows += [{"chapter": cid, **s} for s in sentences]

    save_json(project_dir / "project.json", manifest)
    timeline = build_audio_timeline(project_dir, chapter_gap_sec=float(cfg["chapterGapSec"]), sample_rate=SR)
    starts = {e["id"]: float(e["start"]) for e in timeline["chapters"]}
    files = {f["id"]: f for f in timeline["files"]}

    hosts = []
    for cid, (clips, total) in host_specs.items():
        target = dirs["h3_accepted"] / f"{cid}.mp4"
        voice = project_dir / files[cid]["onCameraWav"]
        _build_host_clip(clips, voice, target, total)
        sync = measure_syncnet(target) if measure_sync else None
        windows = (sync or {}).get("windows") or []
        sync_ok = None if sync is None else bool(windows) and all(
            abs(w["picture_late_ms"]) <= 40 and w["confidence"] >= 1.5 for w in windows)
        save_json(target.with_suffix(".receipt.json"), {
            "route": "native-voice",
            "acceptedPath": str(target),
            "dryWav": str(voice),
            "drySha256": files[cid]["onCameraSha256"],
            "frames": total,
            "chunks": [{"key": p.stem, "frames": f, "padFrames": pad} for p, f, pad in clips],
            "chapterSync": sync,
            "chapterSyncOk": sync_ok,
        })
        hosts.append({"chapter": cid, "path": str(target), "frames": total, "syncOk": sync_ok,
                      "maxAbsMs": max((abs(w["picture_late_ms"]) for w in windows), default=None),
                      "minConfidence": min((w["confidence"] for w in windows), default=None)})

    for r in chunk_rows:
        r["a"] = round(starts[r["chapter"]] + r["localA"], 4)
        r["b"] = round(starts[r["chapter"]] + r["localB"], 4)
    for s in sent_rows:
        s["globalStart"] = round(starts[s["chapter"]] + s["startSec"], 4)
        s["globalEnd"] = round(starts[s["chapter"]] + s["endSec"], 4)
    save_json(nd / "chunks.json", chunk_rows)
    save_json(nd / "alignment.json", sent_rows)
    low = [r["id"] for r in chunk_rows if r["charCoverage"] < 0.6]
    bad_sync = [h["chapter"] for h in hosts if h["syncOk"] is False]
    return {
        "timeline": str(project_dir / "work" / "timeline.json"),
        "totalDuration": timeline["totalDuration"],
        "chunks": str(nd / "chunks.json"),
        "alignment": str(nd / "alignment.json"),
        "hosts": hosts,
        "lowCharCoverage": low,
        "chapterSyncFailures": bad_sync,
        "ok": not bad_sync,
        "next": "subtitles → compose → qc --windows work/host-windows.json；交片时提醒用户连着听几块。",
    }
