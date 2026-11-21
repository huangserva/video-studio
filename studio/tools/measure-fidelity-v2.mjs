#!/usr/bin/env node
// 机器保真尺子 v2：候选片 vs 参考片"像不像"，把颜色算进去，并统一"平均 / 最差一帧"的判定口径。
// v1（measure-fidelity.mjs）保留不动；本文件是并行的新尺子，旧数据可两把尺子对照重量。
//
// v1 的两处漏洞（2026-09-28 实测）：
//   1) MAE 只取差值画面的亮度平均（YAVG）：满屏橙 vs 满屏青（亮度相同）→ MAE=0、闸门过、判 encoded-frame-aligned。
//      颜色整块做错也能登记成"逐帧对齐"。SSIM/PSNR 虽然算了，但不参与判定。
//   2) v1 闸门看平均 MAE，validate-learned-components 看 maxFrameMae（最差一帧）——同一体系两套口径。
//
// v2 的做法：
//   - MAE 在 RGB 三通道上算（两片都转 gbrp 后逐帧相减，node 端逐帧求均值，0–255 尺度），另报各通道均值和最差帧列表。
//   - PSNR 在 RGB 上算。SSIM 算两份：判级用 YUV（和 v1 同口径，0.70 这个门槛就是按 YUV 定的，RGB 上没有校准过的门槛），
//     RGB 的 SSIM 照常算出来写进报告，不参与判级（2026-09-28 黄总定）。
//   - 判定口径统一为"最差一帧"：闸门 = maxFrameMae ≤ 阈值（与校验器 maxFrameMae 同义）。
//     是否"完全一致"（MAE=0）用未四舍五入的原始值判断，只在输出时四舍五入——只差 1 个像素不算逐帧一致。
//   - 帧数闸门：两片帧数不同时只能比上重叠部分，闸门直接判不过、级别封顶 style-reference，
//     并报出两边帧数。带 --align-to-reference 时按候选片对齐后的帧数比较：用候选片的
//     前处理滤镜独立跑一遍只数帧（blend shortest=1 看不到候选多出的帧）；fps 重采样按
//     时长取整，与 时长×帧率 相比允许 ±1 帧误差（比较时用容差 1）。
//   - 级别：render-frame-exact（最差帧 MAE=0）> encoded-frame-aligned（最差帧 ≤2.5）
//           > visual-aligned（平均 MAE ≤12 且 YUV SSIM ≥0.70，两条都要）> style-reference。
//     带数字人（substitute）封顶 visual-aligned，同 v1。
//
// 用法（参数与 v1 兼容，另加 --align-to-reference）：
//   node measure-fidelity-v2.mjs --reference R.mp4 --candidate C.mp4 [--mode none|substitute]
//        [--presenter-box x,y,w,h] [--crop x,y,w,h] [--max-mae 2.5] [--align-to-reference]
//        [--evidence-dir DIR] [--json]
//   --align-to-reference：候选片尺寸或帧率与参考片不同时，先把候选片缩放、重采样到参考片的尺寸和帧率再比。
//                        不给这个参数而尺寸不同，直接报用法错误（退出码 2），不静默缩放。
// 退出码：0 = 闸门过，1 = 不过，2 = 用法/运行错误。

import { execFileSync, spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const FFMPEG = process.env.FFMPEG_BIN || "ffmpeg";
const FFPROBE = process.env.FFPROBE_BIN || "ffprobe";
// 项目阈值（Pluviobyte project fallback，非普适真理）——与 learned-components 校验器 maxFrameMae 一致
const THRESHOLDS = { maxFrameMae: 2.5, visualAlignedMae: 12, visualAlignedSsim: 0.7 };
const WORST_N = 5;

function arg(flag, def) {
  const i = process.argv.indexOf(flag);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : def;
}
const has = (flag) => process.argv.includes(flag);

function usage(msg) {
  console.error(msg);
  process.exit(2);
}

function parseBox(raw, flag) {
  const box = raw.split(",").map(Number);
  if (box.length !== 4 || box.some(Number.isNaN)) usage(`${flag} 格式 x,y,w,h`);
  return box;
}

function probe(file) {
  const r = spawnSync(FFPROBE, ["-v", "error", "-select_streams", "v:0",
    "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", file], { encoding: "utf8" });
  const s = JSON.parse(r.stdout || "{}").streams?.[0];
  if (!s) usage(`读不到视频流：${file}`);
  const [n, d] = s.r_frame_rate.split("/").map(Number);
  return { width: s.width, height: s.height, fps: d ? n / d : n, fpsRaw: s.r_frame_rate };
}

// 逐帧解码数实际帧数（nb_frames 元数据可能缺失或不准，-count_frames 才是真数）
function probeFrames(file) {
  const r = spawnSync(FFPROBE, ["-v", "error", "-select_streams", "v:0", "-count_frames",
    "-show_entries", "stream=nb_read_frames", "-of", "json", file], { encoding: "utf8" });
  const n = JSON.parse(r.stdout || "{}").streams?.[0]?.nb_read_frames;
  if (n === null || n === undefined) usage(`读不到帧数：${file}`);
  return Number(n);
}

// 候选片按前处理滤镜（对齐缩放/重采样等）独立跑一遍只数帧：
// blend 的 shortest=1 会把输出截到 min(参考, 候选)，候选多出来的帧在 MAE 里看不到，
// 必须独立数一遍才知道对齐后的真实帧数。fps 滤镜按输出时长取整帧数，与 时长×帧率 相比有 ±1 帧舍入误差，比较时用容差 1。
function countFramesThrough(file, filters) {
  const r = spawnSync(FFMPEG, ["-hide_banner", "-i", file, "-vf", filters, "-f", "null", "-"],
    { encoding: "utf8", maxBuffer: 64 * 1024 * 1024 });
  if (r.status !== 0) throw new Error(`ffmpeg 数候选帧失败：${(r.stderr || "").trim().slice(-300)}`);
  const m = [...`${r.stderr}\n${r.stdout}`.matchAll(/frame=\s*(\d+)/g)].map((x) => +x[1]);
  if (!m.length) throw new Error("ffmpeg 数候选帧：读不到 frame= 计数");
  return m[m.length - 1]; // 进度行可能多次出现，取最后一次
}

// 两路输入共用的前处理：对齐（只作用于候选）→ 裁剪 / 抠人 → 转 gbrp
function buildPrefilters({ refInfo, align, crop, box }) {
  const common = [];
  if (crop) common.push(`crop=${crop[2]}:${crop[3]}:${crop[0]}:${crop[1]}`);
  if (box) common.push(`drawbox=x=${box[0]}:y=${box[1]}:w=${box[2]}:h=${box[3]}:color=black:t=fill`);
  const candAlign = align ? [`scale=${refInfo.width}:${refInfo.height}:flags=bicubic`, `fps=${refInfo.fpsRaw}`] : [];
  const rgb = [...common, "format=gbrp"], yuv = [...common, "format=yuv420p"];
  return {
    ref: rgb.join(","), cand: [...candAlign, ...rgb].join(","),
    // 同一套裁剪/抠人，只是最后转 yuv420p：给判级用的 SSIM（和 v1 同口径）
    refYuv: yuv.join(","), candYuv: [...candAlign, ...yuv].join(","),
  };
}

function frameSize(refInfo, crop) {
  const w = crop ? crop[2] : refInfo.width;
  const h = crop ? crop[3] : refInfo.height;
  return { w, h, bytes: w * h * 3 };
}

// 逐帧 RGB 差：ffmpeg 输出 gbrp 的差值原始帧（G、B、R 三个平面），node 端求每帧、每通道均值
function measureRgbMae(ref, cand, pre, size, fps, maxMae) {
  return new Promise((resolve, reject) => {
    const graph = `[0:v]${pre.ref}[a];[1:v]${pre.cand}[b];[a][b]blend=all_mode=difference:shortest=1,format=gbrp[d]`;
    const p = spawn(FFMPEG, ["-hide_banner", "-loglevel", "error", "-i", ref, "-i", cand,
      "-filter_complex", graph, "-map", "[d]", "-f", "rawvideo", "-pix_fmt", "gbrp", "pipe:1"]);
    const plane = size.w * size.h;
    const frames = [];
    const chan = { g: 0, b: 0, r: 0 };
    let buf = Buffer.alloc(0);
    let err = "";
    const take = (f) => {
      let g = 0, b = 0, r = 0;
      for (let i = 0; i < plane; i++) g += f[i];
      for (let i = plane; i < 2 * plane; i++) b += f[i];
      for (let i = 2 * plane; i < 3 * plane; i++) r += f[i];
      chan.g += g / plane; chan.b += b / plane; chan.r += r / plane;
      frames.push((g + b + r) / (3 * plane));
    };
    p.stdout.on("data", (chunk) => {
      buf = buf.length ? Buffer.concat([buf, chunk]) : chunk;
      while (buf.length >= size.bytes) {
        take(buf.subarray(0, size.bytes));
        buf = buf.subarray(size.bytes);
      }
    });
    p.stderr.on("data", (d) => { err += d; });
    p.on("error", reject);
    p.on("close", (code) => {
      if (code !== 0) return reject(new Error(`ffmpeg 逐帧差失败(${code})：${err.trim()}`));
      if (!frames.length) return resolve({ averageMae: null, maxFrameMae: null, maxFrameMaeRaw: null, frames: 0, framesOverGate: 0, perChannel: null, worstFrames: [] });
      const n = frames.length;
      const round = (v) => +v.toFixed(4);
      // 原始值（不四舍五入）用于"是否=0"和闸门判断；Math.max(...frames) 帧数多时可能超参数上限，用 reduce
      const rawMax = frames.reduce((m, v) => (v > m ? v : m), -Infinity);
      const worstFrames = frames.map((mae, frame) => ({ frame, sec: round(frame / fps), mae: round(mae) }))
        .sort((x, y) => y.mae - x.mae).slice(0, WORST_N);
      resolve({
        averageMae: round(frames.reduce((s, v) => s + v, 0) / n),
        maxFrameMae: round(rawMax),
        maxFrameMaeRaw: +rawMax.toFixed(8),
        frames: n,
        framesOverGate: frames.filter((v) => v > maxMae).length,
        perChannel: { r: round(chan.r / n), g: round(chan.g / n), b: round(chan.b / n) },
        worstFrames,
      });
    });
  });
}

function ffSummary(ref, cand, pre, filter) {
  const graph = `[0:v]${pre.ref}[a];[1:v]${pre.cand}[b];[a][b]${filter}`;
  const r = spawnSync(FFMPEG, ["-hide_banner", "-nostats", "-i", ref, "-i", cand, "-filter_complex", graph, "-f", "null", "-"],
    { encoding: "utf8", maxBuffer: 64 * 1024 * 1024 });
  return (r.stderr || "") + "\n" + (r.stdout || "");
}
// pre 传 {ref, cand} 两条前处理滤镜链；链尾是 gbrp 就是 RGB SSIM，是 yuv420p 就是 YUV SSIM
function measureSsim(ref, cand, pre) {
  const m = ffSummary(ref, cand, pre, "ssim=shortest=1").match(/SSIM.*All:([0-9.]+)/);
  return m ? +Number(m[1]).toFixed(6) : null;
}
function measurePsnr(ref, cand, pre) {
  const m = ffSummary(ref, cand, pre, "psnr=shortest=1").match(/PSNR.*average:([0-9.inf]+)/);
  if (!m) return null;
  return m[1] === "inf" ? Infinity : +Number(m[1]).toFixed(3);
}

// 统一口径：逐帧对齐看最差一帧；视觉对齐要平均 MAE（RGB）和 SSIM（YUV）同时达标。
// 调用方传未四舍五入的原始 maxFrameMae（==0 才是真的逐帧一致）；帧数不一致时级别封顶 style-reference。
export function suggestFidelityV2({ averageMae, maxFrameMae, ssim, frameMismatch = false }, presenterMode, maxMae = THRESHOLDS.maxFrameMae) {
  let level;
  if (frameMismatch) level = "style-reference";
  else if (maxFrameMae === 0) level = "render-frame-exact";
  else if (maxFrameMae !== null && maxFrameMae <= maxMae) level = "encoded-frame-aligned";
  else if (averageMae !== null && averageMae <= THRESHOLDS.visualAlignedMae
    && ssim !== null && ssim >= THRESHOLDS.visualAlignedSsim) level = "visual-aligned";
  else level = "style-reference";
  if (presenterMode === "substitute") {
    const order = ["style-reference", "visual-aligned", "encoded-frame-aligned", "render-frame-exact"];
    if (order.indexOf(level) > order.indexOf("visual-aligned")) level = "visual-aligned";
  }
  return level;
}

// 对账图：参考行 / 候选行，各 6 张（每秒 2 张）
function makeContactSheet(ref, cand, pre, out, tmpDir) {
  const rowA = path.join(tmpDir, "rowA.jpg"), rowB = path.join(tmpDir, "rowB.jpg");
  const strip = (inp, filters, o) => execFileSync(FFMPEG, ["-y", "-hide_banner", "-loglevel", "error", "-i", inp,
    "-vf", `${filters},fps=2,scale=240:-1,tile=6x1`, "-frames:v", "1", o], { stdio: ["ignore", "ignore", "pipe"] });
  strip(ref, pre.ref, rowA); strip(cand, pre.cand, rowB);
  execFileSync(FFMPEG, ["-y", "-hide_banner", "-loglevel", "error", "-i", rowA, "-i", rowB,
    "-filter_complex", "[0:v][1:v]vstack", "-frames:v", "1", out], { stdio: ["ignore", "ignore", "pipe"] });
}

function writeEvidence(dir, result, ref, cand, pre, tmpDir) {
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "comparison.json"), JSON.stringify(result, null, 2) + "\n");
  const m = result.metrics;
  const worst = m.worstFrames.map((w) => `  - 第 ${w.frame} 帧（${w.sec}s）MAE ${w.mae}`).join("\n");
  fs.writeFileSync(path.join(dir, "report.md"),
    `# 保真对账报告（v2 · RGB）\n\n- 参考: \`${ref}\`\n- 候选: \`${cand}\`\n` +
    `- 模式: ${result.mode}${result.crop ? ` · 只比裁剪区 [${result.crop}]` : ""} · presenterMode=${result.presenterMode}` +
    `${result.aligned ? " · 候选已对齐到参考尺寸/帧率" : ""}\n` +
    `- RGB MAE 平均 \`${m.averageMae}\`，最差一帧 \`${m.maxFrameMae}\`（参考 ${m.refFrames} 帧 / 候选 ${m.candFrames} 帧，比了 ${m.frames} 帧，其中 ${m.framesOverGate} 帧超过闸门）\n` +
    `- 各通道平均：R \`${m.perChannel?.r}\` G \`${m.perChannel?.g}\` B \`${m.perChannel?.b}\`\n` +
    `- SSIM(YUV，判级用): \`${m.ssim}\`　SSIM(RGB，只报): \`${m.ssimRgb}\`　PSNR(RGB): \`${m.psnr}\`\n` +
    `- 闸门(最差一帧 MAE≤${result.thresholds.maxFrameMae}): ${result.gate === "pass" ? "✅ 过" : "❌ 不过"}\n` +
    `- 建议保真级: **${result.fidelitySuggest}**\n\n最差的 ${m.worstFrames.length} 帧：\n${worst}\n` +
    (result.warnings.length ? `\n警告：\n${result.warnings.map((w) => `- ${w}`).join("\n")}\n` : "") +
    `\n> ${result.thresholds.note}\n`);
  try { makeContactSheet(ref, cand, pre, path.join(dir, "contact-sheet.jpg"), tmpDir); } catch { /* 对账图非关键，失败不阻断 */ }
}

async function main() {
  const ref = arg("--reference"), cand = arg("--candidate");
  const mode = arg("--mode", "none");
  const maxMae = Number(arg("--max-mae", THRESHOLDS.maxFrameMae));
  const align = has("--align-to-reference");
  if (!ref || !cand || !fs.existsSync(ref) || !fs.existsSync(cand)) {
    usage("用法：--reference R.mp4 --candidate C.mp4 [--mode none|substitute] [--presenter-box x,y,w,h] [--crop x,y,w,h] [--align-to-reference]");
  }
  if (!["none", "substitute"].includes(mode)) usage("--mode 只能 none|substitute");
  const cropRaw = arg("--crop");
  if (cropRaw && mode === "substitute") usage("暂不支持 --crop 与 substitute 同时（裁剪后 presenter-box 坐标系会变）");
  const crop = cropRaw ? parseBox(cropRaw, "--crop") : null;
  if (crop) {
    // yuv420p 下 ffmpeg 的 crop 滤镜会把奇数宽/高悄悄取偶（101→100、77→76），而 frameSize() 仍按
    // 给定的宽高算每帧字节数：逐帧读取会帧帧错位、MAE 失真、比到的帧数变少，且不报任何错。
    // 宁可报用法错误，不静默取偶。（起点 x、y 为奇数不受影响，crop 起点不要求偶对齐。）
    const evenOf = (v) => v - (v % 2);
    if (crop[2] % 2 !== 0 || crop[3] % 2 !== 0) {
      usage(`--crop 的宽/高必须是偶数：yuv420p 下 ffmpeg 的 crop 会把奇数宽高悄悄取偶（w=${crop[2]}→${evenOf(crop[2])}，h=${crop[3]}→${evenOf(crop[3])}），而每帧字节数按你给的宽高算，帧会错位、帧数变少且不报错。请改用偶数宽高（向下取偶）：${crop[0]},${crop[1]},${evenOf(crop[2])},${evenOf(crop[3])}`);
    }
  }
  let box = null;
  if (mode === "substitute") {
    const boxRaw = arg("--presenter-box");
    if (!boxRaw) usage("substitute 模式必须给 --presenter-box x,y,w,h（人区域矩形）");
    box = parseBox(boxRaw, "--presenter-box");
  }

  const refInfo = probe(ref), candInfo = probe(cand);
  const refFrames = probeFrames(ref);
  const candFramesProbe = align ? null : probeFrames(cand); // 对齐模式下候选帧率/尺寸会变，对齐后的帧数以 blend 实际输出为准
  const warnings = [];
  const sizeDiffers = refInfo.width !== candInfo.width || refInfo.height !== candInfo.height;
  const fpsDiffers = Math.abs(refInfo.fps - candInfo.fps) > 1e-3;
  if (sizeDiffers && !align) {
    usage(`尺寸不同：参考 ${refInfo.width}x${refInfo.height}，候选 ${candInfo.width}x${candInfo.height}。加 --align-to-reference 把候选缩放到参考尺寸再比。`);
  }
  if (fpsDiffers && !align) warnings.push(`帧率不同（参考 ${refInfo.fpsRaw}，候选 ${candInfo.fpsRaw}），逐帧按序号配对，结果可能错位；建议加 --align-to-reference`);
  if (align && (sizeDiffers || fpsDiffers)) warnings.push(`候选片已从 ${candInfo.width}x${candInfo.height}@${candInfo.fpsRaw} 对齐到 ${refInfo.width}x${refInfo.height}@${refInfo.fpsRaw}（缩放本身会带来少量差值）`);

  const pre = buildPrefilters({ refInfo, align, crop, box });
  const size = frameSize(refInfo, crop);
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "fidelity-v2-"));
  try {
    const mae = await measureRgbMae(ref, cand, pre, size, refInfo.fps, maxMae);
    // 帧数闸门：blend shortest=1 只比重叠部分，帧数不同时判不过——重叠部分再像也不能算整片对齐。
    // 对齐模式：候选的帧率/尺寸会被前处理改掉，blend 输出帧数最多等于参考帧数（shortest=1），
    // 候选多出的帧看不到——必须用候选的前处理滤镜独立跑一遍数帧数（fps 重采样允许 ±1 帧舍入误差，容差 1）。
    // 不带 --align-to-reference：行为不变，直接用两片的实际帧数比较。
    let candFrames;
    if (align) {
      candFrames = countFramesThrough(cand, pre.cand);
    } else {
      candFrames = candFramesProbe;
    }
    // 容差只给对齐模式（fps 重采样有舍入误差）；不对齐时两片帧数必须完全相同
    const frameTolerance = align ? 1 : 0;
    const frameMismatch = candFrames !== null && Math.abs(refFrames - candFrames) > frameTolerance;
    if (frameMismatch) warnings.push(`帧数不一致：参考片 ${refFrames} 帧，候选片${align ? `（对齐后）${candFrames}` : ` ${candFrames}`} 帧——只比了重叠部分，闸门判不过；请检查是否截短/多帧/丢帧`);
    // SSIM 两份：判级用 YUV（0.70 门槛按 v1 的 YUV 口径定），RGB 只报不判
    const ssim = measureSsim(ref, cand, { ref: pre.refYuv, cand: pre.candYuv });
    const ssimRgb = measureSsim(ref, cand, pre);
    const psnr = measurePsnr(ref, cand, pre);
    const presenterMode = mode === "substitute" ? "substitute" : "none";
    // 闸门用原始值（未四舍五入）比较，避免编码噪声被 round 掉/放大
    const gatePass = !frameMismatch && mae.maxFrameMaeRaw !== null && mae.maxFrameMaeRaw <= maxMae;
    const result = {
      ruler: "measure-fidelity-v2",
      colorSpace: "rgb",
      mode, presenterMode, crop: cropRaw || null, aligned: align && (sizeDiffers || fpsDiffers),
      metrics: {
        averageMae: mae.averageMae, maxFrameMae: mae.maxFrameMae, maxFrameMaeRaw: mae.maxFrameMaeRaw, frames: mae.frames, framesOverGate: mae.framesOverGate,
        refFrames, candFrames,
        perChannel: mae.perChannel, worstFrames: mae.worstFrames,
        ssim, ssimSpace: "yuv", ssimRgb, psnr: psnr === Infinity ? "inf" : psnr,
      },
      gate: gatePass ? "pass" : "fail",
      // 传给建议器的 maxFrameMae 用原始值：只差 1 个像素时 round 后是 0，不能判 render-frame-exact
      fidelitySuggest: suggestFidelityV2({
        averageMae: mae.averageMae,
        maxFrameMae: mae.maxFrameMaeRaw !== null ? mae.maxFrameMaeRaw : mae.maxFrameMae,
        ssim, frameMismatch,
      }, presenterMode, maxMae),
      thresholds: {
        maxFrameMae: maxMae, visualAlignedMae: THRESHOLDS.visualAlignedMae, visualAlignedSsim: THRESHOLDS.visualAlignedSsim,
        note: (presenterMode === "substitute" ? "带人：masked 比对(抠人区域)，fidelity 封顶 visual-aligned；" : "整帧比对；")
          + "MAE 为 RGB 三通道均值，闸门看最差一帧；SSIM 判级按 YUV（同 v1 口径），RGB SSIM 只报不判",
      },
      warnings,
    };
    const evidenceDir = arg("--evidence-dir");
    if (evidenceDir) writeEvidence(evidenceDir, result, ref, cand, pre, tmpDir);
    if (has("--json")) console.log(JSON.stringify(result, null, 2));
    else {
      const m = result.metrics;
      console.log(`模式: ${mode}${presenterMode === "substitute" ? "（带数字人·抠人版）" : "（整帧）"}${cropRaw ? `　只比裁剪区 [${cropRaw}]` : ""}`);
      console.log(`  RGB MAE 平均: ${m.averageMae}  最差一帧: ${m.maxFrameMae}  (参考 ${m.refFrames} 帧 / 候选 ${m.candFrames} 帧，比了 ${m.frames} 帧，${m.framesOverGate} 帧超过闸门)`);
      if (frameMismatch) console.log(`  ❌ 帧数不一致：参考 ${m.refFrames} 帧 vs 候选 ${m.candFrames} 帧${align ? "（对齐后）" : ""}，只比了重叠部分`);
      console.log(`  各通道平均: R ${m.perChannel?.r}  G ${m.perChannel?.g}  B ${m.perChannel?.b}`);
      console.log(`  SSIM(YUV，判级用): ${m.ssim}   SSIM(RGB，只报): ${m.ssimRgb}   PSNR(RGB): ${m.psnr}`);
      if (m.worstFrames.length) console.log(`  最差帧: ${m.worstFrames.map((w) => `#${w.frame}(${w.sec}s)=${w.mae}`).join("  ")}`);
      console.log(`  闸门(最差一帧 MAE≤${maxMae}): ${gatePass ? "✅ 过" : "❌ 不过"}   建议保真级: ${result.fidelitySuggest}`);
      for (const w of warnings) console.log(`  ⚠️ ${w}`);
    }
    process.exitCode = gatePass ? 0 : 1;
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}

if (import.meta.url === `file://${process.argv[1]}` || process.argv[1]?.endsWith("measure-fidelity-v2.mjs")) {
  main().catch((e) => { console.error(e.message); process.exit(2); });
}
