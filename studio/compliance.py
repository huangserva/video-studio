"""用途、行业限制与合规规则校验（基于 data/use-cases 与 data/industries）。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .config import DATA_DIR
from .project import load_json

MEDICAL_FORBIDDEN_TERMS = (
    "根治",
    "包治",
    "治愈率",
    "有效率",
    "药到病除",
    "代替药物",
    "抗癌",
    "防癌",
)
ABSOLUTIST_WARN_TERMS = (
    "国家级",
    "最高级",
    "最佳",
    "第一品牌",
    "100%有效",
    "稳赚不赔",
    "保本保收益",
)
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")
DERIVED_MARKERS = ("算下来", "推算", "估算", "折算", "算一下")
UNIT_TOKENS = {
    "b", "kb", "mb", "gb", "tb", "kib", "mib", "gib", "tib",
    "ms", "s", "fps", "hz", "khz", "mhz", "ghz", "w", "kw",
}
UNIT_AFTER_RE = r"\s*([A-Za-z]+)"


def list_use_cases() -> list[dict[str, Any]]:
    reg = load_json(DATA_DIR / "use-cases" / "registry.json")
    return reg["useCases"]


def list_industries() -> list[dict[str, Any]]:
    items = []
    for path in sorted((DATA_DIR / "industries").glob("I*.json"), key=lambda p: int(p.stem[1:])):
        items.append(load_json(path))
    return items


def load_pronunciations() -> list[dict[str, Any]]:
    path = DATA_DIR / "pronunciations.json"
    return load_json(path)["entries"] if path.is_file() else []


def normalize_facts(facts: list[Any]) -> list[dict[str, Any]]:
    """brief.facts 兼容两种写法：一句话字符串；或带来源、型号、是否推算、单位的对象。"""
    items: list[dict[str, Any]] = []
    for i, f in enumerate(facts or []):
        if isinstance(f, dict):
            item = dict(f)
            item["claim"] = str(f.get("claim") or f.get("text") or "")
            item["structured"] = True
        else:
            item = {"claim": str(f), "structured": False}
        nums = set(NUMBER_RE.findall(item["claim"]))
        nums.update(str(v) for v in (item.get("values") or []))
        item["numbers"] = nums
        item["n"] = i + 1
        items.append(item)
    return items


def _chapter_sentences(manifest: dict[str, Any]) -> list[tuple[str, int, str, str]]:
    """(章 id, 句序号, 本句, 上一句)。"""
    from .tts import split_sentences

    out = []
    for ch in manifest.get("chapters", []):
        sents = split_sentences(ch.get("narration", ""))
        for i, sent in enumerate(sents):
            out.append((ch.get("id", "?"), i + 1, sent, sents[i - 1] if i else ""))
    return out


def check_facts(manifest: dict[str, Any]) -> list[dict[str, str]]:
    """数字核对：登记过没有、写没写来源、型号有没有说清、推算有没有说「算下来」、单位是否按来源。"""
    warnings: list[dict[str, str]] = []
    facts = normalize_facts((manifest.get("brief") or {}).get("facts", []))
    all_narration = "\n".join(ch.get("narration", "") for ch in manifest.get("chapters", []))
    allowed = set().union(*(f["numbers"] for f in facts)) if facts else set()
    script_nums = sorted(set(NUMBER_RE.findall(all_narration)))
    unverified = [n for n in script_nums if n not in allowed]
    if unverified:
        warnings.append({
            "code": "UNVERIFIED_NUMBERS",
            "message": f"文案中出现未在 brief.facts 登记的数字：{', '.join(unverified)}",
        })

    for f in facts:
        if f["structured"] and not str(f.get("source") or "").strip():
            warnings.append({
                "code": "FACT_SOURCE_MISSING",
                "message": f"brief.facts 第 {f['n']} 条没写来源（source）：{f['claim']}",
            })

    owners: dict[str, list[dict[str, Any]]] = {}
    for f in facts:
        for n in f["numbers"]:
            owners.setdefault(n, []).append(f)

    sentences = _chapter_sentences(manifest)
    for cid, idx, sent, prev in sentences:
        context = prev + sent
        for n in sorted(set(NUMBER_RE.findall(sent))):
            fs = owners.get(n) or []
            if len(fs) != 1:
                continue  # 没登记的上面已经报了；多条事实共用同一个数字时不猜归属
            f = fs[0]
            variant = str(f.get("variant") or "").strip()
            if variant and variant not in context:
                warnings.append({
                    "code": "VARIANT_NOT_NAMED",
                    "message": f"第 {cid} 章第 {idx} 句的 {n} 来自「{variant}」，这句和上一句都没说是哪个型号：{sent}",
                })
            if f.get("derived") and not any(m in context for m in DERIVED_MARKERS):
                warnings.append({
                    "code": "DERIVED_NOT_MARKED",
                    "message": f"第 {cid} 章第 {idx} 句的 {n} 是推算出来的，口播要说「算下来」，画面上标「推算」：{sent}",
                })
            unit = str(f.get("unit") or "").strip()
            if unit:
                for m in re.finditer(re.escape(n) + UNIT_AFTER_RE, sent):
                    tok = m.group(1)
                    if tok.lower() in UNIT_TOKENS and tok != unit:
                        warnings.append({
                            "code": "UNIT_MISMATCH",
                            "message": f"第 {cid} 章第 {idx} 句写的是 {n} {tok}，来源单位是 {unit}，字幕要按来源写：{sent}",
                        })
    return warnings


def check_readings(manifest: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """念法检查：送给配音和 H3 的文本里，不该留阿拉伯数字和已知会念错的写法。另外查 narration/ttsText 句数。"""
    from .tts import split_sentences

    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    entries = load_pronunciations()
    sentence_mode = (manifest.get("voice") or {}).get("dubbing", "sentence") != "chapter"
    for ch in manifest.get("chapters", []):
        cid = ch.get("id", "?")
        narration = ch.get("narration", "")
        tts = ch.get("ttsText") or narration
        if sentence_mode:
            a, b = len(split_sentences(narration)), len(split_sentences(tts))
            if a != b:
                errors.append({
                    "code": "SENTENCE_COUNT_MISMATCH",
                    "message": f"第 {cid} 章 narration 有 {a} 句，ttsText 有 {b} 句。逐句配音要求按 。！？ 一一对应。",
                })
        for field, text in (("ttsText", tts), ("onCameraText", ch.get("onCameraText") or "")):
            if not text:
                continue
            digits = sorted(set(NUMBER_RE.findall(text)))
            if digits:
                warnings.append({
                    "code": "DIGITS_IN_READING",
                    "message": f"第 {cid} 章 {field} 里还有阿拉伯数字 {', '.join(digits)}，口播要写成中文读法。",
                })
            for e in entries:
                m = re.search(e["pattern"], text)
                if not m:
                    continue
                if e.get("level") == "listen":
                    warnings.append({
                        "code": "LISTEN_CAREFULLY",
                        "message": f"第 {cid} 章 {field} 有「{m.group(0)}」：{e.get('note', '')}",
                    })
                else:
                    warnings.append({
                        "code": "PRONUNCIATION_HINT",
                        "message": f"第 {cid} 章 {field} 里是「{m.group(0).strip('。！？，、；')}」：{e.get('note', '')}。建议写成「{e['say']}」。",
                    })
    return errors, warnings


def check_project_compliance(manifest: dict[str, Any]) -> dict[str, Any]:
    use_case_id = manifest.get("useCase", "")
    industry_id = manifest.get("industry", "")

    uc_reg = load_json(DATA_DIR / "use-cases" / "registry.json")
    uc_map = {u["id"]: u for u in uc_reg["useCases"]}
    restrictions = load_json(DATA_DIR / "use-cases" / "industry-restrictions.json")["restrictions"]
    rules_doc = load_json(DATA_DIR / "industries" / "compliance-rules.json")
    rules_map = {r["id"]: r for r in rules_doc["rules"]}

    ind_path = DATA_DIR / "industries" / f"{industry_id}.json"
    industry = load_json(ind_path) if ind_path.is_file() else None
    use_case = uc_map.get(use_case_id)

    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    human_review: list[dict[str, str]] = []

    if not use_case:
        errors.append({"code": "UNKNOWN_USE_CASE", "message": f"未登记的用途 ID: {use_case_id}"})
    if not industry:
        errors.append({"code": "UNKNOWN_INDUSTRY", "message": f"未登记的行业 ID: {industry_id}"})
    else:
        if industry.get("availability") == "blocked":
            errors.append({
                "code": "INDUSTRY_BLOCKED",
                "message": f"行业 {industry_id}（{industry['name']}）当前关闭：{industry.get('availabilityReason', '')}",
            })
        for r in restrictions:
            if r["industryId"] == industry_id and r["useCaseId"] == use_case_id:
                errors.append({
                    "code": "RESTRICTED_PAIR",
                    "message": f"行业 {industry_id} 与用途 {use_case_id} 组合被拦截（{r['kind']}）：{r['reason']}",
                })

        for rule_id in industry.get("complianceRuleRefs", []):
            rule = rules_map.get(rule_id)
            if rule:
                human_review.append({
                    "ruleId": rule["id"],
                    "scope": rule["scope"],
                    "legalBasis": rule["legalBasis"],
                    "text": rule["text"],
                })

    presenter = manifest.get("presenter") or {}
    if presenter.get("enabled"):
        if not presenter.get("portraitAuthorized"):
            warnings.append({
                "code": "PORTRAIT_AUTH_UNCONFIRMED",
                "message": "启用了数字人出镜，但 portraitAuthorized 为 false。对外发布前必须确认真人肖像授权。",
            })
        if not (presenter.get("aiLabel") or "").strip():
            errors.append({
                "code": "MISSING_AI_LABEL",
                "message": "启用了 AI 生成数字人，但未设置 aiLabel（例如 'AI 生成'）。",
            })

    all_narration = "\n".join(ch.get("narration", "") for ch in manifest.get("chapters", []))
    if industry_id != "I15":
        for term in MEDICAL_FORBIDDEN_TERMS:
            if term in all_narration:
                errors.append({
                    "code": "MEDICAL_CLAIM_OUTSIDE_MEDICAL",
                    "message": f"非医疗行业文案出现医疗/功效用语「{term}」（对应 C3《广告法》第十七条）。",
                })
    for term in ABSOLUTIST_WARN_TERMS:
        if term in all_narration:
            warnings.append({
                "code": "RISKY_CLAIM_TERM",
                "message": f"文案出现高风险绝对化/收益承诺词「{term}」，请人工复核。",
            })

    warnings.extend(check_facts(manifest))
    reading_errors, reading_warnings = check_readings(manifest)
    errors.extend(reading_errors)
    warnings.extend(reading_warnings)
    if any(isinstance(f, dict) and f.get("derived") for f in (manifest.get("brief") or {}).get("facts", [])):
        human_review.append({
            "ruleId": "DERIVED-ON-SCREEN",
            "scope": "画面",
            "legalBasis": "sponsored-explainer 商单分寸第 1 条",
            "text": "自己推算的数字，画面上要标「推算」。",
        })

    status = "blocked" if errors else ("warn" if warnings else "pass")
    return {
        "status": status,
        "useCase": {"id": use_case_id, "name": use_case["name"]} if use_case else None,
        "industry": {"id": industry_id, "name": industry["name"]} if industry else None,
        "errors": errors,
        "warnings": warnings,
        "humanReviewChecklist": human_review,
    }
