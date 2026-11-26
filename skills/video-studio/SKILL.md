---
name: video-studio
description: 本地数字人讲解片与商业带货口播视频生产线（默认 H3 原生出声：数字人画面与声音同时生成，挂音色参考，备选 CosyVoice3 逐句配音加锁音；SyncNet 口型门禁，1080p 合成，小窗终检）。Use when the user asks to 用 video-studio 做视频, 做数字人讲解片, 做口播带货视频, 做产品介绍/评测/探店/促销视频, 替换数字人出镜段, 或对已有成片做变速补救与口型复测。
---

# video-studio 视频生产线

工具本体位于本 skill 目录上两级（即 `video-studio` 仓库根目录，可通过 `VIDEO_STUDIO_ROOT` 覆盖，默认从当前 `SKILL.md` 所在路径向上解析两级）。所有步骤通过原子 CLI `./bin/video-studio <command>` 驱动，不手写临时状态机，也不跳过质检门禁。

## 人工只拍 3 次板

1. **选题与规格**：确认主题、用途（`U1–U22`）、行业（`I1–I23`）、音色 ID、出镜形象（虚拟形象或已授权真人）及核心事实数字。商单、AI / 技术产品科普的稿子先按 `sponsored-explainer` skill 写（结构骨架、商单分寸、交片核对单），再进本流程。
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

# 2. 1.3x 逐句干声（每句 3 条，whisper 回读择优）+ 字幕
"$VS/bin/video-studio" tts "$PROJ"
# 输出里看三处：lowAsrSentences（回读对不上的句子，要人听）；boundaries（每句起止秒）；
# onCamera.options（出镜取前 N 句时的 onCameraSec，已按 H3 17k+5 帧取整算好，不会截断人声）。
# 章节里写 "onCameraSentences": N 会自动回填 onCameraSec。手势时间点按 work/asr/<id>.json 的字时间戳定。
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
# 必须查看 outputs/qc/<project-id>-qc-grid.jpg 和 outputs/<project-id>-cover.png

# 5. 改稿（品牌方反馈等）：改 project.json 的 narration/ttsText 后局部重配
"$VS/bin/video-studio" redub "$PROJ" --chapter 03                 # 只重配改过字的句子
"$VS/bin/video-studio" redub "$PROJ" --chapter 03 --sentence 4    # 文字没改，只重摇这一句
# 看 hostSegments：onCameraAudioUnchanged=true 就沿用已验收 H3；false 必须重做 H3 并过 sync-gate。
# 看 cutsWithFixedSeconds：固定秒数的切点不会跟着句子走，改成 atSentence。
# 然后 subtitles → compose → qc。改动按「时间点、原来的、现在的」列表回给用户。
```

## 不可妥协的硬规矩

1. **口播与出声：默认 H3 原生出声（备选 CosyVoice 1.3 倍逐句配音）**
   - 写稿配合 `rn-vo-script`（完整句、禁 hype 词）；商单科普配合 `sponsored-explainer`。本流程的配音做法以这里为准：逐句配，不整段喂。
   - 字幕展示稿 `narration` 与念法稿 `ttsText` 分开存，**两边按 。！？ 切出的句数必须一致**（`check` 不一致直接拦）。屏幕写 `29.9 元`，念法写 `二十九块九`；H3 `onCameraText` 也写中文读法辅助口型。
   - 念法坑登记在 `data/pronunciations.json`（如 `MemT`→`Mem T`、`MemOS`→`Mem O S`），`check` 会逐条提醒；新踩到的坑补进这个文件。
   - **出镜片默认走 H3 原生出声（用户 2026-11-26 拍板）**：H3 同时生成画面和声音，不再先用 CosyVoice 配音再锁音。
     - **工作流改法**：在 lock_source 图（`build_h3_native_graph`）的基础上：
       - 删掉节点 1、2、15，`MiniMaxH3AudioConditioningT8` 去掉 `drive_audio`；
       - 设 `audio_mode: "native"`、`add_source_as_reference: false`、`prompt_primary_audio_ordinal: 0`，`length` 填整数帧数（17k+5）；
       - 用 `LoadAudio` 加载音色参考，接到 `ref_audios.ref_audio_0`。注意不挂参考时 `task_type` 要改 `auto`，否则 Ref2VA 会报错。
       - 成片的画面和声音都从 AVDecode 节点 `["14", 0]` 和 `["14", 1]` 取。
     - **音色参考**：约 10 秒，由一两句干净的整句组成，每句以完整句子收尾，句间留 0.3 s 静音；全片每块都挂同一条，保证音色统一。提示词里称它 `<Audio 1>`，并写明只取音色，不照搬台词和节奏。
     - **提示词**：用六段格式（subject_definitions / summary / retention_analysis / detailed_description / overall_soundscape / non_diegetic_music）；台词写成 `<d>[Chinese] 念法稿 </d>`，语速描述写 `at a natural, fluent documentary narration pace, neither slow nor rushed`。
     - **分块与语速**：在同一段内按整句切块，不跨段，每块不超过约 60–70 字；开场和结尾的块用整屏，中间的块用小窗。**语速只靠时长控制**：`时长 = 0.4 + 字数 × 0.2067` 秒约合每秒 4.8 字（历史片的 1 倍语速），换算成帧数再取整到 17k+5。提示词里写快写慢基本没用。H3 约 0.24 s 后开口，会把台词铺满整个时长，压得太短会截字。
     - **逐块验收**（不过就换 seed，最多 3 次）：whisper 回读相似度 >= 0.75，并按读音人工核对同音字；片尾留白 >= 0.06 s；基频落在 80–95 Hz（参考音色 86.5 Hz）以内；`sync-gate` 以这块自带的音轨为准，允许整帧移画。
     - **拼接**：按顺序拼各块自带的音轨，块间间隔段内 0.10 s、换段 0.40 s，开头留 0.75 s，结尾留 0.8 s。用 whisper 逐 token 时间对齐念法稿的每个字，得出每句起止，写进 `starts.json` 和 `asr.txt`，再交给字幕和画面。小窗取帧用 `TRACK_PAD=0`。之后照常 `compose` → `qc --windows`。
     - **参考实现**（仓库 CLI 还没并入）：`projects/long-march-01/work/native_full.py`（分块、生成、验收）和 `native_assemble.py`（拼人声、逐字对齐）。
     - **实测**（210 秒讲解片）：全流程约 1 小时 30 分，比配音加锁音的 1 小时 56 分快约 25%。成片口型最低置信度 3.40，锁音版是 2.51；读音更准，人工复核也少。代价是各块音色和音高有轻微起伏，节奏不能逐句精确控制，**交片时要提醒用户连着听几块**。
   - **（备选）CosyVoice 逐句配音加锁音**：需要精确控制节奏、纯旁白不出镜，或用户点名时才用。CosyVoice3 默认音色 **`qiuzhi2046-female-v3a`、`speed: 1.3`**（备选 `qiuzhi2046-female-v3b` 更干净，语速要给 1.43）（支持 HTTP 或 `ssh:<gpu-host>:8001`，纯 zero-shot，不加 `instruct`）。每句 3 条，去头尾静音，静音 CosyVoice 句首残音并做 4 ms 头尾淡入（句首 2 ms 内跳变 > 0.01 的条目不参与择优，报告 `edgeJumpFailures` 必须为空），按有声段均方根对齐到 `-20 dBFS`，whisper 回读后挑识别最干净的几条里最短的一条。音色参考音频要以完整句收尾、提示文本逐字一致，否则每句开头会带出半个字的接缝声。回读对上了也要提醒用户亲耳听。
2. **合规与肖像**
   - `check` 命中封禁行业或受限组合直接拦截；所有口播数字必须登记在 `brief.facts`。商单数字写成对象：`source`（来源）、`variant`（哪个型号测的，文案要说出型号）、`derived`（推算，口播说「算下来」、画面标「推算」）、`unit`（字幕按来源单位写）。
   - 真人肖像对外发布前须确认 `portraitAuthorized: true`；所有数字人片段保留左上角 `AI 生成` 标识；带货/促销内容按规定标明「广告」。
3. **H3 参考帧与手势**
   - `assets/waist.png` 必须是 **1:1 正方形腰上构图**，头高约 30%，双肘、前臂、双掌完整入画，背景干净无杂物。
   - 手势按秒绑定具体词眼，克制在胸口以下、不挡嘴、动作间隙自然回原位。
   - 整屏单镜头（如长椅转头开口）用原生路线：`presenter.h3Route: "native-lock"`，写英文 `subjectDescription` 和 `scenePrompt`（不写台词），可配 `guideVideo` 闭唇转头引导片。提示词自动按 H3 官方数字人六段格式生成，默认挂 digicam 真人质感 LoRA；说话的句子上不要安排 smile 或 laugh（会压住口型，`prepare` 会拦下）。讲解片 1:1 窗口仍走默认 `aicg3d`。两条路线都要逐段过 `sync-gate`。
   - H3 输出帧数只有 17k+5 档（24 fps），按请求时长就近取整。用 `tts` 给出的 onCameraSec 建议；出镜后第一个切点写 `"at": "onCameraEnd"`，自动取 onCameraSec 和 H3 实际时长里较短的那个，避免末尾定格。后面的切点优先写 `"atSentence": N`。
4. **口型红线（`sync-gate`）**
   - H3「锁定音频」只保证音轨波形一致，不保证嘴型同步；必须逐段过 `sync-gate`：干声相关度 `>= 0.975` 且尾部不截断可听音；窗口极差 `<= 80 ms` 且不贴 `±360 ms` 才允许 24 fps 整帧移画（`-c:a copy` + SHA-256 核验，复测全窗 `<= 40 ms` 且 `confidence >= 1.5`）。
   - 窗口方向漂移不一致一律 `--take 2` 换 seed 重生成，禁止硬移。
5. **音频与终检（禁止假绿）**
   - `compose` 先出混音：线性增益 + 限幅（`-2 dBFS`，补偿前瞻延迟），测一遍再微调，整体响度对齐 **`-14.5 LUFS`**（±0.5），真峰值不超过 **`-1 dBTP`**；`qc` 默认按这个判定。只使用真实音乐 BGM 或纯人声，严禁合成正弦波底噪；配乐要压在人声下面，宁小勿大。
   - 默认开头 0.75 秒封面（`cover`）：标题、副标题我们自己排版叠字，不让生图模型写中文；封面期间不出人声和字幕。成片第 0 帧导出为 `outputs/<id>-cover.png` 一起交付。
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
