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


def list_use_cases() -> list[dict[str, Any]]:
    reg = load_json(DATA_DIR / "use-cases" / "registry.json")
    return reg["useCases"]


def list_industries() -> list[dict[str, Any]]:
    items = []
    for path in sorted((DATA_DIR / "industries").glob("I*.json"), key=lambda p: int(p.stem[1:])):
        items.append(load_json(path))
    return items


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

    brief = manifest.get("brief") or {}
    facts_text = " ".join(str(x) for x in brief.get("facts", []))
    if facts_text:
        allowed_nums = set(NUMBER_RE.findall(facts_text))
        script_nums = sorted(set(NUMBER_RE.findall(all_narration)))
        unverified = [n for n in script_nums if n not in allowed_nums]
        if unverified:
            warnings.append({
                "code": "UNVERIFIED_NUMBERS",
                "message": f"文案中出现未在 brief.facts 登记的数字：{', '.join(unverified)}",
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
