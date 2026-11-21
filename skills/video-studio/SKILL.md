---
name: video-studio
description: 本地数字人讲解片与商业带货口播视频生产线（CosyVoice3 1.2x 配音 + H3 数字人 + SyncNet 口型门禁 + 1080p 合成与小窗终检）。Use when the user asks to 用 video-studio 做视频, 做数字人讲解片, 做口播带货视频, 做产品介绍/评测/探店/促销视频, 替换数字人出镜段, 或对已有成片做变速补救与口型复测。
---

# video-studio 视频生产线

工具本体位于本 skill 目录上两级（即 `video-studio` 仓库根目录，可通过 `VIDEO_STUDIO_ROOT` 覆盖，默认从当前 `SKILL.md` 所在路径向上解析两级）。所有步骤通过原子 CLI `./bin/video-studio <command>` 驱动，不手写临时状态机，也不跳过质检门禁。

## 人工只拍 3 次板

1. **选题与规格**：确认主题、用途（`U1–U22`）、行业（`I1–I23`）、音色 ID、出镜形象（虚拟形象或已授权真人）及核心事实数字。
2. **样片验收**：定稿口播通过 `check`，第 1 段 5–10 秒 H3 出镜样片通过 `sync-gate` 指标并目检 12 帧网格图后，才批量提交其余出镜段。
3. **成片终审**：查看 `outputs/<id>.mp4` 与终检 12 帧网格图。

## 标准流程与命令

```bash
VS="${VIDEO_STUDIO_ROOT:-$(cd "$(dirname "$(readlink -f ~/.factory/skills/video-studio/SKILL.md)")/../.." && pwd)}"
PROJ="$VS/projects/<project-id>"

# 0. 检查本地环境与目录
"$VS/bin/video-studio" doctor
"$VS/bin/video-studio" catalog use-cases

# 1. 建项与合规预检
"$VS/bin/video-studio" init <project-id> --title "主标题" --subtitle "副标题" \
  --use-case U1 --industry I1 --voice-id <voice-id>
# 编辑 $PROJ/project.json：配置 endpoint（HTTP 或 ssh:<gpu-host>:<port>）、
# brief.facts、chapters（narration/ttsText/onCameraText/gestures/cuts）与 assets/waist.png
"$VS/bin/video-studio" check "$PROJ"

# 2. 1.2x 干声 + Whisper 回听对齐 + 字幕
"$VS/bin/video-studio" tts "$PROJ"
# 用本地 Whisper（openai-whisper 或 whisper-cli）对 work/narration/*.wav 跑词级 ASR，
# 写入 work/asr/<id>.json，回听检查无漏字/误读，按句尾静音谷更新 onCameraSec 与手势时间点
"$VS/bin/video-studio" tts "$PROJ" --build-only
"$VS/bin/video-studio" subtitles "$PROJ"   # 确认 alignmentModes 全为 asr-anchored

# 3. 样片先行（Pilot-first）
"$VS/bin/video-studio" h3-prepare "$PROJ"
"$VS/bin/video-studio" h3-submit "$PROJ" --pilot
"$VS/bin/video-studio" h3-collect "$PROJ" --wait --poll 10
"$VS/bin/video-studio" sync-gate "$PROJ/work/h3/raw/01-1.mp4" --project "$PROJ" --chapter 01
# 必须查看 work/h3/audit/01-1-grid.jpg 确认人物与手势无变形

# 4. 批量生成其余段、逐段过门禁、合成与终检
"$VS/bin/video-studio" h3-submit "$PROJ" --chapters 02 03
"$VS/bin/video-studio" h3-collect "$PROJ" --wait --poll 10
"$VS/bin/video-studio" sync-gate "$PROJ/work/h3/raw/02-1.mp4" --project "$PROJ" --chapter 02
# ... 对每一段逐一 sync-gate 并目检 *-grid.jpg
"$VS/bin/video-studio" compose "$PROJ"
"$VS/bin/video-studio" qc "$PROJ/outputs/<project-id>.mp4" --windows "$PROJ/work/host-windows.json"
# 必须查看 outputs/qc/<project-id>-qc-grid.jpg
```

## 不可妥协的硬规矩

1. **口播与默认 1.2 倍语速**
   - 写稿配合 `rn-vo-script`：每句说完整（对象、动作、结果全），禁压缩 AI 腔和 hype 词。
   - 字幕展示稿 `narration` 与念法稿 `ttsText` 分开存（如屏幕写 `29.9 元`，念法写 `二十九块九`；H3 `onCameraText` 也写中文读法辅助口型）。
   - 新项目 CosyVoice3 默认 **`speed: 1.2`**（支持 HTTP 或 `ssh:<gpu-host>:8001`，纯 zero-shot，不加 `instruct`），先以 1.2 倍生成干声再进 H3。
2. **合规与肖像**
   - `check` 命中封禁行业或受限组合直接拦截；所有口播数字必须登记在 `brief.facts`。
   - 真人肖像对外发布前须确认 `portraitAuthorized: true`；所有数字人片段保留左上角 `AI 生成` 标识；带货/促销内容按规定标明「广告」。
3. **H3 参考帧与手势**
   - `assets/waist.png` 必须是 **1:1 正方形腰上构图**，头高约 30%，双肘、前臂、双掌完整入画，背景干净无杂物。
   - 手势按秒绑定具体词眼，克制在胸口以下、不挡嘴、动作间隙自然回原位。
   - H3 会按内部 latent 步长对时长取整，切点 `onCameraSec` 与对应 `cuts` 需不超过 H3 实际输出视频时长，避免末尾冻帧。
4. **口型红线（`sync-gate`）**
   - H3「锁定音频」只保证音轨波形一致，不保证嘴型同步；必须逐段过 `sync-gate`：干声相关度 `>= 0.975` 且尾部不截断可听音；窗口极差 `<= 80 ms` 且不贴 `±360 ms` 才允许 24 fps 整帧移画（`-c:a copy` + SHA-256 核验，复测全窗 `<= 40 ms` 且 `confidence >= 1.5`）。
   - 窗口方向漂移不一致一律 `--take 2` 换 seed 重生成，禁止硬移。
5. **音频与终检（禁止假绿）**
   - `compose` 自动按 `ebur128` 做纯线性增益对齐 `-16.5 LUFS`；只使用真实音乐 BGM 或纯人声，严禁合成正弦波底噪。
   - 终检必须带 `--windows work/host-windows.json` 从 1080p 成片裁出每个人物小窗复测 SyncNet，并查看 12 帧抽检图。
   - 仅换数字人镜头不重算母带时，使用 `swap-hosts` 保持原音轨 `-c:a copy` 与 SHA-256 一致。

## 已有成片 1.2x 变速补救配方（不重跑 H3）

对已成品做后期 1.2 倍速补救时，`atempo=1.2` 会带来约 `40 ms` 固定延迟，必须同步提前视频 `0.04 s` 补偿并重跑小窗 `qc` 核验（全窗 `<= 40 ms`）：

```bash
ffmpeg -nostdin -y -v error -i input.mp4 \
  -filter_complex "[0:v]setpts=PTS/1.2,trim=start=0.04,setpts=PTS-STARTPTS[v];[0:a]atempo=1.2[a]" \
  -map "[v]" -map "[a]" -r 30 -c:v libx264 -preset fast -crf 17 -pix_fmt yuv420p \
  -c:a aac -b:a 192k -ar 48000 -ac 2 -movflags +faststart output-1p2x.mp4
```

随后把 `work/host-windows.json` 中的 `startSec/endSec/startFrame/endFrame/chapterOriginFrame` 按 `/ 1.2` 缩放为新文件，运行 `qc output-1p2x.mp4 --windows work/host-windows-1p2x.json` 确认通过。
