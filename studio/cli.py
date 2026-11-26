"""video-studio 统一命令行入口：每个原子步骤可单独运行与重跑。"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .compliance import check_project_compliance, list_industries, list_use_cases
from .compose import compose_project, swap_hosts
from .config import (
    DATA_DIR,
    DEFAULT_LUFS_TOLERANCE,
    DEFAULT_MAX_TRUE_PEAK_DB,
    DEFAULT_TARGET_LUFS,
    PROJECTS_DIR,
    TOOLS_DIR,
    ffmpeg_bin,
    ffprobe_bin,
    handcheck_bin,
    syncnet_model_path,
    vision_python,
    whisper_cli_bin,
    whisper_model_path,
)
from .h3 import collect_h3, prepare_h3_graphs, submit_h3
from .project import init_project, load_project
from .qc import run_final_qc
from .subtitles import generate_subtitles
from .sync_gate import review_clip, review_project_clip
from .tts import build_audio_timeline, redub, run_tts


def _emit(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_doctor(_args: argparse.Namespace) -> int:
    vpy = vision_python()
    torch_check = subprocess.run(
        [vpy, "-c", "import torch, cv2, scipy; print(torch.__version__, torch.backends.mps.is_available())"],
        capture_output=True,
        text=True,
    )
    model = syncnet_model_path()
    hc = handcheck_bin()
    report = {
        "ffmpeg": shutil.which(ffmpeg_bin()) or ffmpeg_bin(),
        "ffprobe": shutil.which(ffprobe_bin()) or ffprobe_bin(),
        "node": shutil.which("node"),
        "visionPython": vpy,
        "visionStackOk": torch_check.returncode == 0,
        "visionStackDetail": (torch_check.stdout or torch_check.stderr).strip(),
        "syncnetModel": {"path": str(model), "exists": model.is_file()},
        "handcheck": {"path": str(hc), "exists": hc.is_file()},
        "whisper": {
            "cli": whisper_cli_bin(),
            "model": str(whisper_model_path()) if whisper_model_path() else None,
            "note": "逐句配音靠它回读择优；缺了只能按时长挑，字幕也会退回按字数比例对齐。",
        },
        "tools": {
            "measureFidelityV2": (TOOLS_DIR / "measure-fidelity-v2.mjs").is_file(),
            "fillSlots": (TOOLS_DIR / "fill-slots.mjs").is_file(),
            "transientRecover": (TOOLS_DIR / "transient_recover.py").is_file(),
        },
        "dataCounts": {
            "useCases": len(list_use_cases()),
            "industries": len(list_industries()),
            "recipes": len(list((DATA_DIR / "recipes").glob("*.json"))),
            "styles": len(list((DATA_DIR / "styles").glob("*.json"))),
        },
    }
    _emit(report)
    return 0 if report["visionStackOk"] and report["syncnetModel"]["exists"] else 1


def cmd_catalog(args: argparse.Namespace) -> int:
    if args.kind == "use-cases":
        _emit([{"id": u["id"], "name": u["name"], "driver": u["defaultTimelineDriver"]} for u in list_use_cases()])
    elif args.kind == "industries":
        _emit([
            {"id": i["id"], "name": i["name"], "category": i["category"], "availability": i["availability"]}
            for i in list_industries()
        ])
    elif args.kind == "recipes":
        _emit([p.stem for p in sorted((DATA_DIR / "recipes").glob("*.json"))])
    else:
        _emit([p.stem for p in sorted((DATA_DIR / "styles").glob("*.json"))])
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    target = Path(args.dir) if args.dir else (PROJECTS_DIR / args.name)
    pf = init_project(
        target,
        title=args.title or args.name,
        subtitle=args.subtitle or "",
        use_case=args.use_case,
        industry=args.industry,
        recipe=args.recipe,
        style=args.style,
        **({"voice_id": args.voice_id} if args.voice_id else {}),
    )
    _emit({"created": str(pf)})
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    _, manifest = load_project(args.project)
    res = check_project_compliance(manifest)
    _emit(res)
    return 2 if res["status"] == "blocked" else 0


def cmd_tts(args: argparse.Namespace) -> int:
    if args.build_only:
        res = build_audio_timeline(args.project, chapter_gap_sec=args.gap)
    else:
        res = run_tts(
            args.project,
            chapter_ids=args.chapters or None,
            force=args.force,
            chapter_gap_sec=args.gap,
            takes=args.takes,
        )
    _emit(res)
    return 0


def cmd_redub(args: argparse.Namespace) -> int:
    res = redub(args.project, chapter_id=args.chapter, sentences=args.sentence or None, takes=args.takes)
    _emit(res)
    return 0


def cmd_subtitles(args: argparse.Namespace) -> int:
    _emit(generate_subtitles(args.project))
    return 0


def cmd_h3_prepare(args: argparse.Namespace) -> int:
    written = prepare_h3_graphs(args.project, chapter_ids=args.chapters or None, upload_media=args.upload)
    _emit({"graphs": [str(p) for p in written]})
    return 0


def cmd_h3_submit(args: argparse.Namespace) -> int:
    receipts = submit_h3(
        args.project,
        chapter_ids=args.chapters or None,
        take=args.take,
        pilot_only=args.pilot,
    )
    _emit(receipts)
    return 0


def cmd_h3_collect(args: argparse.Namespace) -> int:
    _emit(collect_h3(args.project, wait=args.wait, poll_sec=args.poll))
    return 0


def cmd_sync_gate(args: argparse.Namespace) -> int:
    if args.project:
        res = review_project_clip(
            args.project,
            Path(args.candidate),
            chapter_id=args.chapter,
            reference_mp4=Path(args.reference) if args.reference else None,
        )
    else:
        if not args.dry or not args.out or not args.audit_dir:
            raise SystemExit("独立模式必须提供 --dry、--out 和 --audit-dir")
        res = review_clip(
            candidate_mp4=Path(args.candidate),
            dry_wav=Path(args.dry),
            accepted_output=Path(args.out),
            audit_dir=Path(args.audit_dir),
            reference_mp4=Path(args.reference) if args.reference else None,
            ai_label=args.ai_label,
        )
    _emit(res)
    return 0 if res.get("accepted") else 1


def cmd_compose(args: argparse.Namespace) -> int:
    res = compose_project(
        args.project,
        output_name=args.output,
        burn_subtitles=not args.no_subtitles,
    )
    _emit(res)
    return 0


def cmd_swap_hosts(args: argparse.Namespace) -> int:
    res = swap_hosts(
        base_video=Path(args.base),
        windows_json=Path(args.windows),
        hosts_dir=Path(args.hosts_dir),
        output_video=Path(args.out),
        fps=args.fps,
    )
    _emit(res)
    return 0


def cmd_qc(args: argparse.Namespace) -> int:
    video = Path(args.video)
    report_dir = Path(args.report_dir) if args.report_dir else (video.parent / "qc")
    res = run_final_qc(
        video,
        report_dir=report_dir,
        windows_json=Path(args.windows) if args.windows else None,
        base_audio_video=Path(args.base_audio) if args.base_audio else None,
        expected_width=args.width,
        expected_height=args.height,
        expected_fps=args.fps,
        max_sync_abs_ms=args.max_sync_ms,
        target_lufs=None if args.no_loudness_gate else args.target_lufs,
        lufs_tolerance=args.lufs_tolerance,
        max_true_peak_db=args.max_true_peak,
    )
    _emit(res)
    return 0 if res["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="video-studio", description="极简本地视频生产与硬门禁工具链")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="检查本地依赖、SyncNet 权重、handcheck 和数据包")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("catalog", help="列出内置用途、行业包、配方或样式")
    p.add_argument("kind", choices=["use-cases", "industries", "recipes", "styles"])
    p.set_defaults(func=cmd_catalog)

    p = sub.add_parser("init", help="初始化一个新视频项目目录与 project.json")
    p.add_argument("name", help="项目短名（英文/数字/连字符）")
    p.add_argument("--dir", help="自定义目录路径（默认在 projects/<name>）")
    p.add_argument("--title", help="视频主标题")
    p.add_argument("--subtitle", help="视频副标题")
    p.add_argument("--use-case", default="U8", help="用途 ID（默认 U8 知识科普讲解）")
    p.add_argument("--industry", default="I20", help="行业 ID（默认 I20 软件和 App）")
    p.add_argument("--recipe", default="curator-h3-explainer")
    p.add_argument("--style", default="museum-paper")
    p.add_argument("--voice-id", default=None, help="CosyVoice 音色 ID，默认 qiuzhi2046-female-v3a（语速 1.3）")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("check", help="检查项目简报的用途/行业拦截与合规清单")
    p.add_argument("project", help="项目目录或 project.json 路径")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("tts", help="逐句调用 CosyVoice3 生成干声（每句多条择优），或仅汇总已有分章干声时间轴")
    p.add_argument("project")
    p.add_argument("--chapters", nargs="*", help="仅生成指定章节 ID")
    p.add_argument("--force", action="store_true", help="整章重配（不沿用已有句子）")
    p.add_argument("--takes", type=int, help="每句生成几条（默认取 voice.takesPerSentence，3）")
    p.add_argument("--build-only", action="store_true", help="不调 TTS，直接汇总已有干声并切分出镜段")
    p.add_argument("--gap", type=float, default=0.35, help="章间停顿秒数")
    p.set_defaults(func=cmd_tts)

    p = sub.add_parser("redub", help="局部重配：只重配改过字或指定的句子，并报告 H3 出镜段是否还能沿用")
    p.add_argument("project")
    p.add_argument("--chapter", required=True, help="章节 ID")
    p.add_argument("--sentence", type=int, nargs="*", help="文字没改也要重摇的句子序号（从 1 开始）")
    p.add_argument("--takes", type=int, help="每句生成几条")
    p.set_defaults(func=cmd_redub)

    p = sub.add_parser("subtitles", help="生成与干声时间轴对齐的 ASS 字幕")
    p.add_argument("project")
    p.set_defaults(func=cmd_subtitles)

    p = sub.add_parser("h3-prepare", help="生成每章 H3 ComfyUI 工作流 JSON（可离线预览）")
    p.add_argument("project")
    p.add_argument("--chapters", nargs="*")
    p.add_argument("--upload", action="store_true", help="同时上传参考图和出镜干声到 ComfyUI")
    p.set_defaults(func=cmd_h3_prepare)

    p = sub.add_parser("h3-submit", help="提交 H3 任务（默认强制先过 1 段 --pilot 样片）")
    p.add_argument("project")
    p.add_argument("--chapters", nargs="*")
    p.add_argument("--take", type=int, default=1, help="重试 take 号（>1 自动按 +104729 递增 seed）")
    p.add_argument("--pilot", action="store_true", help="仅提交第 1 段样片")
    p.set_defaults(func=cmd_h3_submit)

    p = sub.add_parser("h3-collect", help="收集已完成的 H3 原始片段并做 ffmpeg 完整性校验")
    p.add_argument("project")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--poll", type=int, default=12)
    p.set_defaults(func=cmd_h3_collect)

    p = sub.add_parser("sync-gate", help="逐段 H3 验收：波形相关度 + SyncNet 逐窗 + 24fps 整帧移画 + handcheck")
    p.add_argument("candidate", help="待验收 H3 视频路径")
    p.add_argument("--project", help="所属项目目录（自动定位干声与输出路径）")
    p.add_argument("--chapter", help="章节 ID（配合 --project 使用）")
    p.add_argument("--dry", help="独立模式：对应出镜干声 WAV")
    p.add_argument("--out", help="独立模式：验收通过后的输出 MP4")
    p.add_argument("--audit-dir", help="独立模式：审计 JSON 与 12 帧网格图目录")
    p.add_argument("--reference", help="可选：换人模式下的参考片段（锁总帧数和原音轨）")
    p.add_argument("--ai-label", default="AI 生成", help="独立模式左上角标识（留空则不烧）")
    p.set_defaults(func=cmd_sync_gate)

    p = sub.add_parser("compose", help="按策展外壳版式整数帧合成 1080p 成片")
    p.add_argument("project")
    p.add_argument("--output", help="输出文件名（默认 <id>.mp4）")
    p.add_argument("--no-subtitles", action="store_true")
    p.set_defaults(func=cmd_compose)

    p = sub.add_parser("swap-hosts", help="只替换成片中的人物小窗，原音轨 copy 并核对哈希")
    p.add_argument("--base", required=True)
    p.add_argument("--windows", required=True)
    p.add_argument("--hosts-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=int, default=30)
    p.set_defaults(func=cmd_swap_hosts)

    p = sub.add_parser("qc", help="成片终检：规格 + 响度/真峰值 + 12 帧抽检图 + 成片人物小窗 SyncNet 复测")
    p.add_argument("video", help="最终成片 MP4")
    p.add_argument("--windows", help="host-windows.json，用于裁出成片人物小窗复测 SyncNet")
    p.add_argument("--base-audio", help="可选：比对音轨 SHA-256 的底片路径")
    p.add_argument("--report-dir", help="终检报告与抽检图目录")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--max-sync-ms", type=int, default=40)
    p.add_argument("--target-lufs", type=float, default=DEFAULT_TARGET_LUFS, help="整体响度目标（默认 -14.5）")
    p.add_argument("--lufs-tolerance", type=float, default=DEFAULT_LUFS_TOLERANCE, help="允许偏差 LU（默认 0.5）")
    p.add_argument("--max-true-peak", type=float, default=DEFAULT_MAX_TRUE_PEAK_DB, help="真峰值上限 dBTP（默认 -1）")
    p.add_argument("--no-loudness-gate", action="store_true", help="只测响度不判定（复查旧片用）")
    p.set_defaults(func=cmd_qc)

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
