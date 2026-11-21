# video-studio

极简、硬门禁驱动的本地数字人讲解与口播视频生产工具链。

核心链路：`project.json` -> 合规检查 -> CosyVoice3 干声（默认 1.2x）-> Whisper 词级字幕对齐 -> H3 13 号无限分段数字人（样片先行）-> SyncNet 口型门禁与 24fps 整数移画 + `handcheck` 手部检测 -> 1080p 策展外壳合成与线性响度对齐（`-16.5 LUFS`）-> 成片人物小窗裁剪 SyncNet 复测与 12 帧抽检图。

所有步骤均为独立原子命令，不设重型多阶段状态机，随时可单步重跑或仅无损替换出镜段。

## 目录结构

```text
video-studio/
├── AGENTS.md                 # AI Agent 操作守则
├── bin/
│   └── video-studio          # CLI 入口（自动优先选择带 torch/cv2/scipy 的 Python）
├── data/
│   ├── use-cases/            # 22 用途目录 + 8 组行业/用途限制
│   ├── industries/           # 23 行业包 + 14 条法条出处合规原文
│   ├── licenses/             # BrewReel Apache-2.0 LICENSE & NOTICE
│   ├── recipes/              # 标准视频配方（curator-h3-explainer）
│   └── styles/               # 外壳视觉主题（museum-paper）
├── skills/
│   └── video-studio/         # AI Coding Agent 全局 Skill（SKILL.md）
├── studio/
│   ├── cli.py                # 统一命令入口 python3 -m studio <cmd>
│   ├── config.py             # 环境路径、端点与 vision python 自动解析
│   ├── project.py            # project.json 脚手架与目录规范
│   ├── compliance.py         # 用途/行业/合规与肖像授权门禁
│   ├── tts.py                # CosyVoice3 干声（HTTP / SSH 跳板）、切分出镜段、主时间轴
│   ├── subtitles.py          # 中文字幕分句、whisper.cpp / openai-whisper 锚点对齐、ASS 导出
│   ├── h3.py                 # ComfyUI-H3 与 Qwen Image Edit 接口（HTTP / SSH 跳板）
│   ├── sync_gate.py          # 波形相关度 + SyncNet + 24fps 整帧移画 + handcheck
│   ├── compose.py            # split/wide/host/full 整数帧合成、响度校准与无损换人
│   ├── qc.py                 # 成片规格、LUFS、12 帧抽检图与小窗 SyncNet 复测
│   └── tools/                # 独立工具（syncnet_check、handcheck.m、fidelity-v2、fill-slots、transient_recover）
├── tests/                    # 单元、端到端合成与门禁测试
└── projects/                 # 实际视频生产目录（gitignored）
```

## 环境准备

1. **基础依赖**：Python 3.10+（装有 `torch`、`opencv-python`、`scipy`、`Pillow`、`numpy`）、`ffmpeg` / `ffprobe`、Node.js 18+（运行填槽/保真工具）。
2. **SyncNet 权重（本地不入 git）**：将 `syncnet_v2.model` 放到 `models/syncnet_v2.model`（或通过环境变量 `SYNCNET_MODEL_PATH` 指定）。
3. **macOS 手部检测二进制（可选，推荐）**：
   ```bash
   clang -fobjc-arc -framework Vision -framework AVFoundation -framework CoreMedia -framework Foundation \
     studio/tools/handcheck.m -o bin/handcheck
   ```
4. **远端/本地推理服务端点**：支持直连 HTTP 或免隧道 SSH 跳板（`ssh:<host>:<port>`），可通过环境变量或在 `project.json` 中配置：
   ```bash
   export VIDEO_STUDIO_TTS_ENDPOINT="http://127.0.0.1:8001"        # 或 ssh:<gpu-host>:8001
   export VIDEO_STUDIO_H3_ENDPOINT="http://127.0.0.1:8190"         # 或 ssh:<gpu-host>:8190
   export VIDEO_STUDIO_QWEN_EDIT_ENDPOINT="http://127.0.0.1:8192"  # 或 ssh:<gpu-host>:8192
   ```

## 快速上手

```bash
cd video-studio

# 1. 环境自检（ffmpeg、vision python、torch/mps、SyncNet 权重、handcheck）
./bin/video-studio doctor

# 2. 查看内置用途、行业、配方与样式
./bin/video-studio catalog use-cases
./bin/video-studio catalog industries

# 3. 新建项目（默认语速 1.2x）
./bin/video-studio init my-topic --title "主题标题" --subtitle "副标题" \
  --use-case U1 --industry I1 --voice-id <voice-id>

# 4. 写完 projects/my-topic/project.json 后做合规门禁检查
./bin/video-studio check projects/my-topic

# 5. 生成干声（或把已有 WAV 放进 work/narration/ 后加 --build-only）
./bin/video-studio tts projects/my-topic

# 6. 写入 work/asr/<id>.json（兼容 openai-whisper 与 whisper.cpp）后生成词级对齐 ASS 字幕
./bin/video-studio subtitles projects/my-topic

# 7. H3 样片先行（先跑第 1 段并过 sync-gate 与 12 帧目检，才允许批量提交）
./bin/video-studio h3-prepare projects/my-topic
./bin/video-studio h3-submit projects/my-topic --pilot
./bin/video-studio h3-collect projects/my-topic --wait
./bin/video-studio sync-gate projects/my-topic/work/h3/raw/01-1.mp4 --project projects/my-topic --chapter 01

# 8. 若某段口型窗口漂移不一致，不强行移帧，换 seed 重跑 take 2
./bin/video-studio h3-submit projects/my-topic --chapters 01 --take 2

# 9. 合成 1080p 成片并跑终检（含成片人物小窗裁剪 SyncNet 复测与 12 帧抽检图）
./bin/video-studio compose projects/my-topic
./bin/video-studio qc projects/my-topic/outputs/my-topic.mp4 --windows projects/my-topic/work/host-windows.json
```

## 作为 AI Agent 全局 Skill 安装

仓库内置了即用型 Skill（`skills/video-studio/SKILL.md`），可通过软链挂到常用的 Agent Skills 目录：

```bash
ln -sfn "$(pwd)/skills/video-studio" ~/.factory/skills/video-studio
ln -sfn "$(pwd)/skills/video-studio" ~/.agents/skills/video-studio
ln -sfn "$(pwd)/skills/video-studio" ~/.claude/skills/video-studio
ln -sfn "$(pwd)/skills/video-studio" ~/.codex/skills/video-studio
```

## 运行测试

```bash
python3 -m unittest discover -s tests -v
node --test tests/test_fill_slots.mjs
```

## 开源许可与致谢

- 本项目基于 Apache License 2.0 发布，详见 `LICENSE` 与 `NOTICE`。
- `data/use-cases/` 与 `data/industries/` 中的用途目录与行业合规数据包源自 [BrewReel（精酿）](https://github.com/Finderchangchang/brewreel)，由柳伟杰（GitHub: `Finderchangchang`）创作并以 Apache-2.0 许可发布，原始许可与声明保留于 `data/licenses/`。
- 合规规则仅用于内容自查参考，不构成法律意见；对外发布内容须以监管部门及平台最新规定为准。
