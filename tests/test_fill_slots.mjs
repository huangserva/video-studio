import test from "node:test";
import assert from "node:assert/strict";
import childProcess from "node:child_process";
import { syncBuiltinESMExports } from "node:module";
import { fillSlotsForAct } from "../studio/tools/fill-slots.mjs";

async function fill(t, slots, texts, options = {}) {
  let prompt;
  t.mock.method(childProcess, "execFileSync", (_cmd, args) => {
    prompt = args[1];
    return JSON.stringify({
      structuredOutput: { slots: slots.map((s, i) => ({ name: s.name, text: texts[i] })) },
    });
  });
  syncBuiltinESMExports();
  t.after(() => {
    t.mock.restoreAll();
    syncBuiltinESMExports();
  });
  const values = await fillSlotsForAct({ slots, topic: "主题原文", narration: "口播原文", ...options });
  return { values, prompt };
}

test("short labels expand to MIN_SLOT_WIDTH=4 then truncate by width", async (t) => {
  const { values, prompt } = await fill(t, [{ name: "label", default: "原文" }], ["新的超长标签"]);
  assert.deepEqual(values, { label: "新的超长" });
  assert.match(prompt, /宽度上限=4/);
});

test("short English label keeps width 4 so Chinese replacement is not cut in half", async (t) => {
  const { values } = await fill(
    t,
    [
      { name: "a", default: "New" },
      { name: "b", default: "New" },
    ],
    ["新品上市", "新品上市限时特惠"],
  );
  assert.deepEqual(values, { a: "新品上市", b: "新品上市" });
});

test("maxChars remains hard ceiling even below MIN_SLOT_WIDTH", async (t) => {
  const { values } = await fill(t, [{ name: "ok", default: "OK", maxChars: 2 }], ["好的收到"]);
  assert.deepEqual(values, { ok: "好的" });
});

test("empty defaults do not acquire budget", async (t) => {
  const { values } = await fill(t, [{ name: "empty", default: "" }, { name: "missing" }], ["新增", "新增"]);
  assert.deepEqual(values, { empty: "", missing: "" });
});

test("empty slot list needs no agent", async () => {
  assert.deepEqual(await fillSlotsForAct({ slots: [], topic: "", narration: "" }), {});
});
