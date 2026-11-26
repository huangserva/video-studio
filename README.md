# video-studio

极简、硬门禁驱动的本地数字人讲解与口播视频生产工具链。

核心链路：`project.json` -> 合规、事实与念法检查 -> CosyVoice3 逐句干声（默认 1.3x，每句 3 条，whisper 回读择优）-> 字幕对齐 -> H3 13 号无限分段数字人（样片先行）-> SyncNet 口型门禁与 24fps 整数移画 + `handcheck` 手部检测 -> 封面 + 1080p 策展外壳合成与响度对齐（`-14.5 LUFS`，真峰值 ≤ `-1 dBTP`）-> 成片人物小窗裁剪 SyncNet 复测与 12 帧抽检图。改稿时用 `redub` 只重配改过的句子。

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
│   ├── pronunciations.json   # CosyVoice 念法坑（check 会提醒 ttsText 换写法）
│   ├── recipes/              # 标准视频配方（curator-h3-explainer）
│   └── styles/               # 外壳视觉主题（museum-paper）
├── skills/
│   └── video-studio/         # AI Coding Agent 全局 Skill（SKILL.md）
├── studio/
│   ├── cli.py                # 统一命令入口 python3 -m studio <cmd>
│   ├── config.py             # 环境路径、端点与 vision python 自动解析
│   ├── project.py            # project.json 脚手架与目录规范
│   ├── compliance.py         # 用途/行业/合规与肖像授权门禁
│   ├── tts.py                # CosyVoice3 逐句干声与择优、局部重配、切分出镜段、主时间轴
│   ├── subtitles.py          # 中文字幕分句、whisper.cpp / openai-whisper 锚点对齐、ASS 导出
│   ├── h3.py                 # ComfyUI-H3 与 Qwen Image Edit 接口（HTTP / SSH 跳板）
│   ├── sync_gate.py          # 波形相关度 + SyncNet + 24fps 整帧移画 + handcheck
│   ├── compose.py            # 封面、split/wide/host/full 整数帧合成、响度校准与无损换人
│   ├── qc.py                 # 成片规格、LUFS 与真峰值、12 帧抽检图与小窗 SyncNet 复测
│   └── tools/                # 独立工具（syncnet_check、handcheck.m、fidelity-v2、fill-slots、transient_recover）
├── tests/                    # 单元、端到端合成与门禁测试
└── projects/                 # 实际视频生产目录（gitignored）
```

## 环境准备

1. **基础依赖**：Python 3.10+（装有 `torch`、`opencv-python`、`scipy`、`Pillow`、`numpy`）、`ffmpeg` / `ffprobe`、Node.js 18+（运行填槽/保真工具）。
2. **SyncNet 权重（本地不入 git）**：将 `syncnet_v2.model` 放到 `models/syncnet_v2.model`（或通过环境变量 `SYNCNET_MODEL_PATH` 指定）。
   **whisper.cpp（逐句回读择优用）**：安装 `whisper-cli`，把 `ggml-small.bin` 放到 `models/ggml-small.bin`（或用 `VIDEO_STUDIO_WHISPER_MODEL` 指定）。缺了它只能按时长挑，`doctor` 会提示。
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

# 3. 新建项目（默认语速 1.3x，默认带 0.75 秒封面）
./bin/video-studio init my-topic --title "主题标题" --subtitle "副标题" \
  --use-case U1 --industry I1 --voice-id <voice-id>

# 4. 写完 projects/my-topic/project.json 后做合规门禁检查
./bin/video-studio check projects/my-topic

# 5. 逐句生成干声：每句 3 条，whisper 回读择优，自动写 work/asr/<id>.json，
#    并给出每章出镜取前几句时的 onCameraSec 建议（已考虑 H3 帧数取整）
./bin/video-studio tts projects/my-topic

# 6. 生成字幕（alignmentModes 应全为 asr-anchored）
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

# 10. 改稿只动几句：改 project.json 里的 narration/ttsText 后局部重配，
#     报告会说明出镜段音频变没变（没变就沿用已验收的 H3，不用重新生成）
./bin/video-studio redub projects/my-topic --chapter 03
./bin/video-studio redub projects/my-topic --chapter 03 --sentence 4   # 文字没改、只想重摇一句
```

### H3 三条路线

`presenter.h3Route` 选出镜生成路线，不写就用配方里的 `hostSegment.route`（讲解片配方是 `aicg3d`）。`init` 新建的项目写的是 `native-voice`。

- `native-voice`（新项目默认，2026-11-26 起）：H3 原生出声，画面和声音同时生成，挂同一条音色参考 `presenter.voiceRef`，不配音也不锁音。用 `native` 命令：

  ```bash
  ./bin/video-studio native projects/my-topic --plan       # 分块：章内整句、每块不超过 62 字，时长 = 0.4 + 字数 × 0.2067 s 取整到 17k+5 帧
  ./bin/video-studio native projects/my-topic --pilot      # 样片：只跑第 1 块
  ./bin/video-studio native projects/my-topic              # 批量生成 + 逐块验收（回读 >= 0.75、片尾 >= 0.06 s、基频在参考 ±9 Hz、sync-gate），不过换 seed，最多 3 次
  ./bin/video-studio native projects/my-topic --assemble   # 拼人声、逐字对齐、汇总 timeline.json、拼每章出镜片
  ```

  4090 共用：只在队列里没有排队任务时提交，不清队列；跑完队列空闲才释放显存。之后照常 `subtitles` → `compose` → `qc --windows`。

- `aicg3d`：AICG3D 序列节点「数字人（锁定音频）」，turbo 8 步加 1.5 倍精修，1:1 腰上窗口。讲解片默认用它。
- `native-lock`：原生 MiniMax H3 T8 节点，`lock_source`、`Ref2VA`、20 步、不精修，默认 864×480。长椅这类整屏单镜头用它。2026-11-24 用同一条干声、同一段引导片、同 3 个 seed 对照：原生 3/3 原片窗口在 0 到 +80 ms，AICG3D 有 1 条整体晚 200 多毫秒。

原生路线的提示词按 H3 官方数字人说话版六段格式自动生成（AICG3D `prompt_guides/digital_human_audio_performance`）：`subject_definitions`、`summary`（以 `[reference generation + audio reuse]` 开头）、`retention_analysis`（`<Audio 1>: fully_copy`）、`detailed_description`、`overall_soundscape`、`non_diegetic_music`。不写台词，口型交给锁定干声。原生图没有接 Picture，所以只引用 `<Audio 1>`，人物按「首帧里的人」来描述。要写的字段：

- `presenter.subjectDescription`：首帧人物的长相、发型和服装，用英文写。
- `presenter.scenePrompt`（章节里可以用 `chapters[].scenePrompt` 覆盖）：用英文写景别、机位、场景和开场动作，用 `<Subject 1>` 指代人物。
- `presenter.sceneSummary`（可选）：一句话概括；不写就用通用说法。
- `presenter.guideVideo`（可选）：首帧引导片，比如 22 帧闭唇转头。没有就用 `waistImage` 当首帧。
- `chapters[].gestures`：每句最多一个小动作，会写成「At 秒-秒 seconds 动作」。**说话的句子上不要安排 smile、laugh、grin**，写了 `prepare` 会直接报错。2026-11-24 实测，这类词会让后半段定格成露齿笑，嘴不再跟着字开合，口型也会漂。笑只放在句间停顿里，提示词已经自动带上这一句。

开口时刻取 `voice.headPadSec`。尺寸、步数、风格写在 `presenter.h3Native`，例如 `{"width": 864, "height": 480, "steps": 20, "style": "a realistic natural live-action shot in soft daylight", "soundscape": "..."}`。

原生路线**默认挂 digicam 真人质感 LoRA**（`minimax-h3-digicam-realism-improve.safetensors`，强度 1.0，没有触发词）。2026-11-24 在 5 个真人感 LoRA 里由人工目检定版，用官方格式提示词跑，sync-gate 置信度 3.06。不挂就写 `"loras": []`；换别的就写 `"loras": [{"name": "...", "strength": 1.0, "trigger": "..."}]`，触发词会放在提示词最前面。换 LoRA 时注意两点：

- 查 ComfyUI 日志里有没有 `lora key not loaded`。键名不带 `diffusion_model.` 前缀的 LoRA 会被整个跳过，不报错，只是不起作用（facial-realism-closeup 原版就是这样）。
- 看嘴有没有张开。「影帝表演」这个 LoRA 会压住口型，最响的 8 个元音里有 7 个闭着嘴，不能用在说话镜头上。

两条路线都照样逐段跑 `sync-gate`。

### 切点写法

`chapters[].cuts[]` 的时间可以写三种：

- `"at": 6.55`：章内固定秒数。
- `"at": "onCameraEnd"`：出镜段结尾，自动取 `onCameraSec` 和 H3 实际时长里较短的那个，避免末尾定格。
- `"atSentence": 4`：第 4 句开口前 0.1 秒。局部重配后句子时长变了，切点会跟着走。

### brief.facts 写法

每条可以是一句话，也可以写成对象，`check` 会多查几项：

```json
{"claim": "聊到 25 万 token 时回答一次 24 秒", "values": ["25", "24"],
 "source": "技术报告 v3 表 4", "variant": "Nano", "unit": "MiB", "derived": false}
```

- `variant`：这条数字是哪个型号测的。文案里用到这个数字时，本句或上一句要说出型号。
- `derived: true`：自己推算的数字。口播要说「算下来」，画面上标「推算」。
- `unit`：来源单位。字幕要照来源写（MiB 不写成 MB），口播可以说「兆」。

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
