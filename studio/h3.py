"""4090 本地 H3（AICG3D 13 号无限分段）与 Qwen Image Edit 腰上构图工作流。"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import ffmpeg_bin
from .project import load_json, load_project, load_recipe, project_dirs, save_json

UUID_RE = re.compile(r"[\da-f-]{36}")


class ComfyEndpoint:
    """支持直连 HTTP（http://127.0.0.1:8190）或 SSH 跳板（ssh:<gpu-host>:8190）。"""

    def __init__(self, spec: str):
        self.spec = spec
        if spec.startswith("ssh:"):
            parts = spec.split(":")
            if len(parts) != 3:
                raise ValueError(f"SSH 端点格式应为 ssh:<host>:<port>，收到: {spec}")
            self.mode = "ssh"
            self.host = parts[1]
            self.port = int(parts[2])
        elif spec.startswith(("http://", "https://")):
            self.mode = "http"
            self.base_url = spec.rstrip("/")
        else:
            raise ValueError(f"不支持的 ComfyUI 端点格式: {spec}")

    def _ssh_prefix(self) -> list[str]:
        return [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ConnectTimeout=8",
            self.host,
        ]

    def request(self, path: str, body: bytes | None = None, timeout: int = 70) -> bytes:
        if self.mode == "http":
            headers = {"Content-Type": "application/json"} if body is not None else {}
            req = Request(self.base_url + path, data=body, headers=headers)
            with urlopen(req, timeout=timeout) as resp:
                return resp.read()
        args = ["curl", "-fsS", "--max-time", str(max(10, timeout - 10)), f"http://127.0.0.1:{self.port}{path}"]
        if body is not None:
            args += ["-H", "Content-Type: application/json", "--data-binary", "@-"]
        return subprocess.check_output(self._ssh_prefix() + [shlex.join(args)], input=body, timeout=timeout)

    def upload(self, local_file: Path, *, subfolder: str, filename: str | None = None) -> str:
        name = filename or local_file.name
        if self.mode == "http":
            out = subprocess.check_output(
                [
                    "curl",
                    "-fsS",
                    "--max-time",
                    "45",
                    "-F",
                    f"image=@{local_file};filename={name}",
                    "-F",
                    f"subfolder={subfolder}",
                    f"{self.base_url}/upload/image",
                ],
                timeout=60,
            )
            result = json.loads(out)
        else:
            remote = (
                "t=$(mktemp); trap 'rm -f \"$t\"' EXIT; cat > \"$t\"; "
                f"curl -fsS --max-time 30 -F \"image=@$t;filename={name}\" "
                f"-F \"subfolder={subfolder}\" http://127.0.0.1:{self.port}/upload/image"
            )
            with local_file.open("rb") as handle:
                out = subprocess.check_output(self._ssh_prefix() + [remote], stdin=handle, timeout=60)
            result = json.loads(out)
        if result.get("subfolder") != subfolder or result.get("type") != "input":
            raise RuntimeError(f"上传到 ComfyUI input 失败: {result}")
        return f"{subfolder}/{result['name']}"


def build_waist_edit_prompt(subject_description: str) -> str:
    return (
        "Edit this exact studio portrait by pulling the camera back significantly. "
        f"Keep the identity, face, clothing, backdrop and soft studio lighting of {subject_description}. "
        "Recompose to an eye-level frontal medium shot from the top of the head down through the hips, "
        "looking straight at camera with mouth gently closed. Both elbows and forearms, wrists and both "
        "entire relaxed open hands are clearly visible near the waist, hands loosely separated with all "
        "fingertips inside the frame and fingers naturally grouped. Leave generous clear room to both sides "
        "for subsequent hand gestures. Head height about 30% of the frame. One subject centered in a square "
        "frame. No text, logo, scenery, accessories or props."
    )


def build_waist_edit_graph(remote_image: str, subject_description: str, remote_folder: str, seed: int = 7821953) -> dict[str, Any]:
    def node(kind: str, **inputs: Any) -> dict[str, Any]:
        return {"class_type": kind, "inputs": inputs}

    prompt = build_waist_edit_prompt(subject_description)
    return {
        "unet": node("UNETLoader", unet_name="qwen_image_edit_2511_fp8_lightning_4steps.safetensors", weight_dtype="default"),
        "clip": node("CLIPLoader", clip_name="qwen_2.5_vl_7b_fp8_scaled.safetensors", type="qwen_image", device="default"),
        "vae": node("VAELoader", vae_name="qwen_image_vae.safetensors"),
        "sampling": node("ModelSamplingAuraFlow", model=["unet", 0], shift=3.1),
        "cfg": node("CFGNorm", model=["sampling", 0], strength=1.0, pre_cfg=False),
        "source": node("LoadImage", image=remote_image),
        "scale": node("FluxKontextImageScale", image=["source", 0]),
        "pos": node("TextEncodeQwenImageEditPlus", clip=["clip", 0], vae=["vae", 0], image1=["scale", 0], prompt=prompt),
        "neg": node("TextEncodeQwenImageEditPlus", clip=["clip", 0], vae=["vae", 0], image1=["scale", 0], prompt=""),
        "posref": node("FluxKontextMultiReferenceLatentMethod", conditioning=["pos", 0], reference_latents_method="index_timestep_zero"),
        "negref": node("FluxKontextMultiReferenceLatentMethod", conditioning=["neg", 0], reference_latents_method="index_timestep_zero"),
        "canvas": node("EmptySD3LatentImage", width=1024, height=1024, batch_size=1),
        "render": node(
            "KSampler",
            model=["cfg", 0],
            positive=["posref", 0],
            negative=["negref", 0],
            latent_image=["canvas", 0],
            seed=seed,
            steps=4,
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
            denoise=1.0,
        ),
        "decode": node("VAEDecode", samples=["render", 0], vae=["vae", 0]),
        "save": node("SaveImage", images=["decode", 0], filename_prefix=f"{remote_folder}/waist"),
    }


def format_gesture_lines(gestures: list[dict[str, Any]]) -> str:
    if not gestures:
        return "Keep subtle, natural open-palm gestures timed to the key phrase, then return hands smoothly to rest."
    parts = []
    for g in gestures:
        phrase = g.get("phrase", "").strip()
        start = g.get("start")
        end = g.get("end")
        action = g.get("action", "").strip()
        timing = f" ({start:.2f}-{end:.2f} s)" if start is not None and end is not None else ""
        if phrase:
            parts.append(f"When saying “{phrase}”{timing}, {action}.")
        else:
            parts.append(f"At{timing}, {action}.")
    return " ".join(parts)


def build_h3_prompt(
    *,
    subject_description: str,
    dialogue: str,
    gestures: list[dict[str, Any]],
    dual_reference: bool,
) -> str:
    if dual_reference:
        subject_def = (
            "<Subject 1> is the same presenter as <Picture 1> and <Picture 2>. "
            "Use the exact camera framing, starting pose, outfit and studio backdrop of <Picture 1>. "
            "Preserve the exact face, skin texture, hair and facial features of <Picture 2>."
        )
    else:
        subject_def = (
            f"<Subject 1> is {subject_description} from <Picture 1>. "
            "Preserve the exact face, clothing, backdrop and soft neutral lighting."
        )
    gesture_block = format_gesture_lines(gestures)
    return (
        "subject_definitions:\n"
        f"{subject_def}\n"
        "<Audio 1> is the exact Mandarin dialogue and its timing. Match it syllable by syllable.\n\n"
        "summary:\n"
        "Use the locked square eye-level waist-up studio framing of Picture 1, same camera distance "
        "and lighting throughout. Keep both elbows, forearms and entire hands visible inside the frame. "
        "Mouth and eyes remain clearly readable.\n\n"
        "detailed_description:\n"
        "[Shot 1] <Subject 1> addresses the camera, follows the exact Mandarin audio, with realistic "
        "closed lips on bilabials and natural jaw movement. "
        f"He says: <d>[Chinese] {dialogue.strip()} </d>\n"
        f"{gesture_block}\n"
        "Keep gestures compact, phrase-timed and below chest height. Each wrist connects naturally to "
        "the forearm; hands stay in frame, never cover the mouth, and return to rest between gestures. "
        "Natural blinks, steady eyeline, minimal head nodding. One person, one shot, no subtitles or graphics.\n\n"
        "overall_soundscape: Only supplied <Audio 1>, locked.\n"
        "non_diegetic_music: N/A\n"
    )


def build_h3_graph(
    *,
    remote_waist: str,
    remote_portrait: str | None,
    remote_audio: str,
    prompt_text: str,
    seconds: float,
    seed: int,
    filename_prefix: str,
    client_id: str,
    recipe_host: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = recipe_host or {}
    refine = cfg.get("refine") or {"enabled": True, "steps": 4, "denoise": 0.35, "scale": 1.5}
    images = [{"filename": remote_waist}]
    if remote_portrait and remote_portrait != remote_waist:
        images.append({"filename": remote_portrait})

    graph = {
        "1": {
            "class_type": "AICG3D_H3Loader",
            "inputs": {
                "fl2va_model": "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors",
                "ref2va_model": "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors",
                "text_encoder": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                "video_vae": "minimax_h3_video_vae_fp16.safetensors",
                "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
                "lora_1": "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
                "lora_1_strength": 1.0,
                "lora_2": "none",
                "lora_2_strength": 1.0,
                "lora_3": "none",
                "lora_3_strength": 1.0,
                "lora_4": "none",
                "lora_4_strength": 1.0,
                "non_h3_policy": "自动回落",
                "lora_5": "none",
                "lora_5_strength": 1.0,
                "lora_6": "none",
                "lora_6_strength": 1.0,
                "lora_7": "none",
                "lora_7_strength": 1.0,
                "lora_8": "none",
                "lora_8_strength": 1.0,
                "lora_9": "none",
                "lora_9_strength": 1.0,
                "注意力加速(Sage)": "自动（Sage 优先）",
            },
        },
        "2": {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {"shift_video": 12.0, "shift_audio": 2.0, "model": ["1", 1]},
        },
        "3": {
            "class_type": "AICG3D_H3SequenceGlobal",
            "inputs": {
                "resolution": cfg.get("resolution", "480P"),
                "aspect_ratio": cfg.get("aspectRatio", "1:1"),
                "fps": float(cfg.get("fps", 24.0)),
                "handoff_frames": 5,
                "handoff_mode": "尾帧续写（latent）",
                "ref_image_size": "1k",
                "sampler_name": "res_multistep",
                "scheduler": "simple",
                "steps": 8,
                "denoise": 1.0,
                "vram_policy": "释放缓存",
                "sample_preview": "关",
                "preview_interval": 1,
                "prompt_optimizer_settings": False,
                "audio_mode": cfg.get("audioMode", "数字人（锁定音频）"),
                "vram_tier": "自动（按显卡）",
                "exposure_lock": "关闭",
                "keep_full": False,
                "h3_bundle": ["1", 0],
                "model": ["2", 0],
            },
        },
        "4": {"class_type": "LoadImage", "inputs": {"image": remote_waist}},
        "11": {
            "class_type": "AICG3D_H3SequenceSegment",
            "inputs": {
                "prompt": prompt_text,
                "seconds": round(float(seconds), 3),
                "seed": int(seed),
                "guide_frame_idx": 0,
                "inline_media_state": json.dumps(
                    {"images": images, "audios": [{"filename": remote_audio}]},
                    ensure_ascii=False,
                ),
                "guide_image": ["4", 0],
                "sequence_config": ["3", 0],
            },
        },
        "91": {
            "class_type": "AICG3D_H3SequenceCombine",
            "inputs": {
                "filename_prefix": filename_prefix,
                "final_segment": ["11", 0],
                "refine_pass2": bool(refine.get("enabled", True)),
                "refine_steps": int(refine.get("steps", 4)),
                "refine_denoise": float(refine.get("denoise", 0.35)),
                "refine_scale": float(refine.get("scale", 1.5)),
            },
        },
    }
    return {"prompt": graph, "client_id": client_id}


def prepare_h3_graphs(
    project_ref: str | Path,
    *,
    chapter_ids: list[str] | None = None,
    upload_media: bool = False,
) -> list[Path]:
    project_dir, manifest = load_project(project_ref)
    dirs = project_dirs(project_dir)
    recipe = load_recipe(manifest.get("recipe", "curator-h3-explainer"))
    host_cfg = recipe.get("hostSegment", {})
    presenter = manifest["presenter"]
    folder = presenter.get("remoteFolder") or f"video-studio-{project_dir.name}"
    seed_base = int(presenter.get("seedBase", host_cfg.get("seedBase", 7822077)))

    waist_local = project_dir / presenter["waistImage"]
    portrait_rel = presenter.get("portraitImage")
    portrait_local = (project_dir / portrait_rel) if portrait_rel else None
    dual_ref = bool(portrait_local and portrait_local != waist_local and portrait_local.is_file())

    endpoint = ComfyEndpoint(presenter["h3Endpoint"]) if upload_media else None
    if upload_media:
        if not waist_local.is_file():
            raise FileNotFoundError(f"缺少腰上参考图: {waist_local}")
        remote_waist = endpoint.upload(waist_local, subfolder=folder, filename="waist.png")
        remote_portrait = (
            endpoint.upload(portrait_local, subfolder=folder, filename="portrait.png")
            if dual_ref and portrait_local
            else None
        )
    else:
        remote_waist = f"{folder}/waist.png"
        remote_portrait = f"{folder}/portrait.png" if dual_ref else None

    timeline_path = project_dir / "work" / "timeline.json"
    timeline_dur = {}
    if timeline_path.is_file():
        for item in load_json(timeline_path).get("chapters", []):
            timeline_dur[item["id"]] = float(item.get("onCameraSec", 0.0))

    selected = set(chapter_ids) if chapter_ids else None
    written: list[Path] = []
    for ch in manifest["chapters"]:
        cid = ch["id"]
        if selected and cid not in selected:
            continue
        sec = timeline_dur.get(cid) or float(ch.get("onCameraSec") or 0.0)
        if sec <= 0:
            continue
        if sec > float(host_cfg.get("maxDurationSec", 15.0)):
            raise ValueError(f"第 {cid} 章出镜时长 {sec:.2f}s 超过 H3 单段上限 15s")

        voice_local = dirs["h3_voice"] / f"voice-{cid}.wav"
        if upload_media:
            if not voice_local.is_file():
                raise FileNotFoundError(f"缺少出镜干声: {voice_local}（先跑 tts）")
            remote_audio = endpoint.upload(voice_local, subfolder=folder, filename=f"voice-{cid}.wav")
        else:
            remote_audio = f"{folder}/voice-{cid}.wav"

        prompt_text = build_h3_prompt(
            subject_description=presenter.get("subjectDescription", "the presenter"),
            dialogue=ch.get("onCameraText") or ch["narration"],
            gestures=ch.get("gestures") or [],
            dual_reference=dual_ref,
        )
        payload = build_h3_graph(
            remote_waist=remote_waist,
            remote_portrait=remote_portrait,
            remote_audio=remote_audio,
            prompt_text=prompt_text,
            seconds=sec,
            seed=seed_base,
            filename_prefix=f"{folder}/shot_{cid}",
            client_id=f"{folder}-{cid}",
            recipe_host=host_cfg,
        )
        written.append(save_json(dirs["h3_graphs"] / f"{cid}.json", payload))
    return written


def submit_h3(
    project_ref: str | Path,
    *,
    chapter_ids: list[str] | None = None,
    take: int = 1,
    pilot_only: bool = False,
) -> list[dict[str, Any]]:
    project_dir, manifest = load_project(project_ref)
    dirs = project_dirs(project_dir)
    recipe = load_recipe(manifest.get("recipe", "curator-h3-explainer"))
    stride = int(recipe.get("hostSegment", {}).get("seedRetryStride", 104729))

    target_ids = list(chapter_ids or [c["id"] for c in manifest["chapters"] if c.get("onCameraSec")])
    if pilot_only:
        target_ids = target_ids[:1]
    elif take == 1 and len(target_ids) > 1 and not any(dirs["h3_accepted"].glob("*.mp4")):
        raise RuntimeError(
            "样片先行规则：accepted/ 中尚无已通过验收的样片。请先加 --pilot 跑 1 段样片并通过 sync-gate，再批量提交。"
        )

    prepare_h3_graphs(project_dir, chapter_ids=target_ids, upload_media=True)
    endpoint = ComfyEndpoint(manifest["presenter"]["h3Endpoint"])
    receipts: list[dict[str, Any]] = []

    for cid in target_ids:
        key = f"{cid}-{take}"
        receipt_path = dirs["h3_jobs"] / f"{key}.json"
        if receipt_path.is_file():
            receipts.append({"key": key, "skipped": True, **load_json(receipt_path)})
            continue
        graph = load_json(dirs["h3_graphs"] / f"{cid}.json")
        if take > 1:
            graph["client_id"] += f"-take{take}"
            graph["prompt"]["11"]["inputs"]["seed"] += (take - 1) * stride
            graph["prompt"]["91"]["inputs"]["filename_prefix"] += f"_take{take}"
        raw = endpoint.request("/prompt", json.dumps(graph, ensure_ascii=False).encode("utf-8"))
        resp = json.loads(raw)
        if not UUID_RE.fullmatch(resp.get("prompt_id", "")) or resp.get("node_errors"):
            raise RuntimeError(f"ComfyUI 拒绝提交 {key}: {resp}")
        receipt_path.write_bytes(raw)
        receipts.append({"key": key, "skipped": False, **resp})
    return receipts


def collect_h3(project_ref: str | Path, *, wait: bool = False, poll_sec: int = 12) -> list[dict[str, Any]]:
    project_dir, manifest = load_project(project_ref)
    dirs = project_dirs(project_dir)
    endpoint = ComfyEndpoint(manifest["presenter"]["h3Endpoint"])

    while True:
        results: list[dict[str, Any]] = []
        pending = 0
        for receipt in sorted(dirs["h3_jobs"].glob("*.json")):
            target = dirs["h3_raw"] / f"{receipt.stem}.mp4"
            err_file = dirs["h3_raw"] / f"{receipt.stem}.error.json"
            if target.is_file():
                results.append({"key": receipt.stem, "status": "downloaded", "path": str(target)})
                continue
            if err_file.is_file():
                results.append({"key": receipt.stem, "status": "failed", "error": str(err_file)})
                continue

            job_id = load_json(receipt)["prompt_id"]
            history = json.loads(endpoint.request(f"/history/{job_id}")).get(job_id)
            if not history:
                pending += 1
                results.append({"key": receipt.stem, "status": "running"})
                continue
            status = history.get("status", {})
            if status.get("status_str") != "success":
                save_json(err_file, status)
                results.append({"key": receipt.stem, "status": "failed", "error": str(err_file)})
                continue

            item = history["outputs"]["91"]["images"][0]
            qs = urlencode({k: item[k] for k in ("filename", "subfolder", "type")})
            data = endpoint.request(f"/view?{qs}", timeout=180)
            tmp = target.with_suffix(".tmp.mp4")
            tmp.write_bytes(data)
            subprocess.run(
                [ffmpeg_bin(), "-nostdin", "-v", "error", "-xerror", "-i", str(tmp), "-f", "null", "-"],
                check=True,
            )
            tmp.replace(target)
            results.append({"key": receipt.stem, "status": "downloaded", "path": str(target)})

        if not wait or pending == 0:
            return results
        time.sleep(poll_sec)
