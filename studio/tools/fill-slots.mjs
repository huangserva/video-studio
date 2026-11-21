// 结构化填槽（2026-07-26 黄总指出"动效内容跟口播对不上"后的正解）。
//
// 问题的根子：组件是**有语义角色的表单**（大标题 / 卡1标题 / 卡1副标题 / …，一幕可达39个位），
// 而之前的做法是"把口播稿截40个字塞进其中1个位"——剩下的位还写着原片内容，
// 于是画面讲 A、音频讲 B，两张皮（2026-07-26 黄总一眼看出）。
//
// 正解：把每一幕的**槽位清单 + 原内容 + 该幕口播 + 全片主题**交给 agent，
// 让它按"原来的语义角色和字数量级"逐位生成新内容。组件的版式/动效/几何一律不动，只换字。
import { execFileSync } from "node:child_process";

const SCHEMA = {
  type: "object",
  required: ["slots"],
  properties: {
    slots: {
      type: "array",
      items: {
        type: "object",
        required: ["name", "text"],
        properties: { name: { type: "string" }, text: { type: "string" } },
      },
    },
  },
};

const graphemes = new Intl.Segmenter("und", { granularity: "grapheme" });
const segments = (text) => Array.from(graphemes.segment(String(text ?? "")), ({ segment }) => segment);

// 估算宽度单位：中文/全角/其他非ASCII字形=1，ASCII字母/数字/标点/空格=0.5。
// 组合音标不额外占宽；按字素截断，不拆代理对或组合字。它不是字体像素测量。
// 原文宽度是硬上限，maxChars是额外的硬上限；另外非空原文还有一个 MIN_SLOT_WIDTH 下限（见下）。
function characterWidth(segment) {
  const base = segment.normalize("NFD").replace(/\p{Mark}/gu, "");
  return /^[\x00-\x7f]$/.test(base) ? 0.5 : 1;
}

// 宽度预算下限：非空原文的槽，预算宽度至少 MIN_SLOT_WIDTH。
// 为什么要下限：PR #15 去掉了旧的"至少12字符"下限后，短英文标签换成中文会被截成
// 半截——原文 "New"（宽度1.5）、新文案 "新品上市"（宽度4）会被截成 "新"，语义直接变了。
// 为什么取 4：4 个宽度单位 = 4 个汉字或 8 个英文字符，是短标签能保住意思的最小长度；
// 再低就必然退回到"一个字"这种一定变形的程度。
// maxChars 仍是额外的硬上限：若某槽的 maxChars 比下限还紧，以 maxChars 为准
// （fitText 同时按 width 和 chars 两个预算截断，谁更紧谁先生效）。
// 空原文槽位不适用下限：原片里就是空的位不该凭空多出文字。
const MIN_SLOT_WIDTH = 4;

function slotBudget(slot) {
  const width = segments(slot.default).reduce((sum, segment) => sum + characterWidth(segment), 0);
  return {
    width: width > 0 ? Math.max(width, MIN_SLOT_WIDTH) : width,
    chars: Number.isFinite(slot.maxChars) && slot.maxChars >= 0 ? Math.floor(slot.maxChars) : Infinity,
  };
}

function fitText(text, budget) {
  const parts = segments(text);
  let width = 0, chars = 0, end = 0;
  for (const part of parts) {
    const nextWidth = width + characterWidth(part);
    const nextChars = chars + Array.from(part).length;
    if (nextWidth > budget.width || nextChars > budget.chars) break;
    width = nextWidth;
    chars = nextChars;
    end++;
  }
  if (end === parts.length) return text;
  let cut = parts.slice(0, end).join("");
  // If a sentence ends mid-English-word, prefer the previous word boundary.
  if (/[a-z0-9]$/i.test(cut) && /^[a-z0-9]/i.test(parts[end] || "") && /\s/.test(cut)) {
    cut = cut.replace(/\s+\S*$/, "");
  }
  return cut.trimEnd();
}

/**
 * 为一幕生成整套新文案。
 * @param {object} p
 * @param {Array<{name:string,default:string,maxChars?:number}>} p.slots 该幕组件的文字位
 * @param {string} p.narration 这一幕对应的口播内容
 * @param {string} p.topic 全片主题（整段新文案）
 * @param {string} p.agent grok|claude|codex
 * @returns {Promise<Record<string,string>>} { 槽名: 新文案 }
 */
export async function fillSlotsForAct({ slots, narration, topic, agent = "grok", timeoutMs = 180000 }) {
  if (!Array.isArray(slots) || slots.length === 0) return {};
  const list = slots
    .map((s, i) => {
      const budget = slotBudget(s);
      return `${i + 1}. name=${s.name} 原文="${s.default ?? ""}" 宽度上限=${budget.width}${Number.isFinite(budget.chars) ? `，字符数上限=${budget.chars}` : ""}`;
    })
    .join("\n");
  const prompt = `你是视频文案改写员。有一个做好的动效版式，里面每个文字位都有固定的语义角色（大标题/卡片标题/副标题/标签等）。
现在要用这个版式讲一个**新主题**，请为每个文字位写出对应的新文案。

## 全片新主题（整段口播稿）
${String(topic).slice(0, 1200)}

## 这一幕的口播内容（画面要跟这段话说的事对上）
${String(narration).slice(0, 400)}

## 这一幕的文字位（共${slots.length}个）
${list}

## 规则
1. 逐位改写：保持每个位原来的**语义角色**，严格遵守宽度上限和字符数上限。
   宽度按中文/全角=1、英文字母/数字/半角标点/空格=0.5计算。短标签也不许超预算，宁可少写。
2. 内容必须服务于"这一幕的口播内容"，让观众看画面和听声音说的是同一件事。
3. 纯装饰性/无实义的位（如 "01"、"→"、纯符号、英文缩写标签）保持原样返回。
4. 不许编造具体数字、公司名、人名；口播稿里没有的事实不要出现在画面上。
5. 只返回文字，不解释。每个位都要给，一个都不能漏。`;

  const ADAPTERS = {
    grok: ["grok", ["-p", prompt, "--always-approve", "--output-format", "json", "--json-schema", JSON.stringify(SCHEMA)]],
    claude: ["claude", ["-p", prompt, "--dangerously-skip-permissions"]],
    codex: ["codex", ["exec", prompt, "--dangerously-bypass-approvals-and-sandbox"]],
  };
  const [cmd, args] = ADAPTERS[agent] || ADAPTERS.grok;
  let out;
  try {
    out = execFileSync(cmd, args, { encoding: "utf8", maxBuffer: 32 << 20, timeout: timeoutMs, env: { ...process.env, NO_COLOR: "1" } });
  } catch (e) {
    return { __error: `agent失败:${String(e.message).slice(0, 80)}` };
  }
  // grok 结构化输出在 structuredOutput；其他家从文本里捞 JSON
  let parsed = null;
  try {
    const j = JSON.parse(out);
    parsed = j.structuredOutput || j;
  } catch {
    const m = out.match(/\{[\s\S]*"slots"[\s\S]*\}/);
    if (m) { try { parsed = JSON.parse(m[0]); } catch { /* ignore */ } }
  }
  if (!parsed?.slots) return { __error: "agent没给出可用的槽位内容" };
  const values = {};
  for (const s of parsed.slots) {
    if (!s?.name || typeof s.text !== "string") continue;
    const orig = slots.find((x) => x.name === s.name);
    if (!orig) continue;
    values[s.name] = fitText(s.text, slotBudget(orig));
  }
  return values;
}
