"""红旗症状（Red Flag）安全检查。

健康域的安全前置检查：在意图识别之前用关键词快速筛查危险信号，
命中后强制把请求推入就医预警链路（urgency=CRITICAL → MedicalAlertAgent），
保证急症信号不会被普通健康建议延误。

设计原则：漏检是致命缺陷，宁可多拦不可漏拦；
但通过否定/假设语境过滤降低误报，避免正常咨询被误判为急症。
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


RED_FLAG_KEYWORDS: List[str] = [
    "胸痛", "胸闷", "呼吸困难", "昏迷", "抽搐", "大出血", "吐血", "咯血",
    "黑便", "剧烈头痛", "偏瘫", "高烧不退", "意识模糊", "休克",
    # 口语化/变体表达（评测驱动的扩充：覆盖用户日常说法）
    "胸口闷", "胸口闷闷", "胸口有点闷", "胸口疼", "胸口堵", "气不够用",
    "闷得难受", "闷得慌", "胸口发闷",
    "喘不上气", "喘不过气", "喘气费劲", "说不利索", "说话含糊",
    "晕倒", "晕了过去", "眼前一黑", "晕厥",
    "吐了血", "吐过血", "一口血", "咳血",
    "柏油样", "柏油一样", "黑得像柏油",
    "半边身子", "一侧无力", "身子没力气", "嘴角歪",
    "烧了三天", "高烧三天", "高烧一直不退", "一直高烧", "一直退不下来",
    "出血特别多", "血流不止", "止不住血",
    "头炸裂", "炸裂一样疼",
    "纽扣电池", "误吞电池", "吞下电池",
]

# 否定前缀：紧邻红旗词时视为"未发生"，如"没有胸痛""没昏迷"
_NEGATION_PREFIXES: List[str] = ["没有", "没出现", "没发生", "并未", "没有过", "未", "不是", "不算", "没"]

# 假设/知识咨询标记：出现时视为提问而非症状自述。
# 只收录明确的知识咨询信号，避免把真急症（如“胸痛是怎么回事”）误放。
_QUESTION_MARKERS: List[str] = [
    "是什么", "什么是", "什么病", "哪种病",
    "怎么预防", "如何预防", "预防",
    "会不会", "是否会",
    "上次", "医生说是", "医生诊断",
]

_STRONG_RED_FLAG_KEYWORDS: List[str] = [
    "呼吸困难", "喘不上气", "喘不过气", "气不够用", "昏迷", "抽搐", "大出血",
    "吐血", "咯血", "黑便", "柏油样", "柏油一样", "黑得像柏油", "偏瘫",
    "半边身子", "一侧无力", "嘴角歪", "说话含糊", "休克", "血流不止",
    "头炸裂", "炸裂一样疼", "纽扣电池", "误吞电池", "吞下电池",
]


@dataclass
class SafetyCheckResult:
    """红旗症状检测结果。"""
    risk_level:      str                 # "high" / "low"
    matched_flags:   List[str] = field(default_factory=list)
    action:          str = "no_escalate"  # must_escalate / ask_clarifying / no_escalate
    source:          str = "rule"         # rule / llm / fallback
    reason:          str = ""
    llm_failed:      bool = False
    error: Optional[str] = None

    @property
    def is_high_risk(self) -> bool:
        return self.risk_level == "high" or self.action == "must_escalate"

    @property
    def should_ask_clarifying(self) -> bool:
        return self.action == "ask_clarifying"


# 子句分隔符：否定判断只在同一子句内生效，
# 避免“最近没有运动，胸口闷闷的”中前句的“没有”误取消后句症状
_CLAUSE_SEPARATORS = "，。？！、；,.;!?\n"


def _is_negated(text: str, start: int) -> bool:
    """判断命中位置所在子句内、前 5 字是否存在否定前缀。"""
    clause_start = 0
    for i in range(start - 1, -1, -1):
        if text[i] in _CLAUSE_SEPARATORS:
            clause_start = i + 1
            break
    window = text[max(clause_start, start - 5):start]
    return any(neg in window for neg in _NEGATION_PREFIXES)


def detect_red_flags(text: str) -> SafetyCheckResult:
    """
    检测文本中是否包含红旗症状关键词。

    三层判断：
      1. 关键词/口语变体子串匹配（保召回，漏检是致命缺陷）
      2. 否定语境过滤："没有胸痛"等紧邻否定不触发
      3. 全局假设语境过滤：纯知识咨询/预防类提问不触发
    """
    text = text or ""
    matched = [
        kw for kw in RED_FLAG_KEYWORDS
        if any(not _is_negated(text, m.start())
               for m in re.finditer(re.escape(kw), text))
    ]
    if any(marker in text for marker in _QUESTION_MARKERS):
        return SafetyCheckResult(
            risk_level="low",
            matched_flags=matched,
            action="no_escalate",
            reason="规则检测到知识咨询/预防/既往转述语境",
        )
    strong = [kw for kw in matched if kw in _STRONG_RED_FLAG_KEYWORDS]
    if strong:
        return SafetyCheckResult(
            risk_level="high",
            matched_flags=matched,
            action="must_escalate",
            reason="命中强红旗关键词，规则直接升级",
        )
    return SafetyCheckResult(
        risk_level="high" if matched else "low",
        matched_flags=matched,
        action="must_escalate" if matched else "no_escalate",
        reason="命中红旗关键词" if matched else "未命中红旗关键词",
    )


def detect_red_flag_candidates(text: str) -> SafetyCheckResult:
    """只做红旗候选召回和否定过滤，不用知识咨询标记直接放行。"""
    text = text or ""
    matched = [
        kw for kw in RED_FLAG_KEYWORDS
        if any(not _is_negated(text, m.start())
               for m in re.finditer(re.escape(kw), text))
    ]
    question_like = any(marker in text for marker in _QUESTION_MARKERS)
    strong = [kw for kw in matched if kw in _STRONG_RED_FLAG_KEYWORDS]
    if strong and not question_like:
        action = "must_escalate"
        reason = "命中强红旗关键词，规则直接升级"
    elif matched:
        action = "ask_clarifying"
        reason = "命中红旗候选词，需语义复核"
    else:
        action = "no_escalate"
        reason = "未命中红旗候选词"
    return SafetyCheckResult(
        risk_level="high" if action == "must_escalate" else "low",
        matched_flags=matched,
        action=action,
        reason=reason,
    )


class SafetyClassifier:
    """
    两阶段安全分流：规则召回候选风险，再用 LLM 复核弱红旗语境。

    LLM 只做安全分流，不做诊断：must_escalate / ask_clarifying / no_escalate。
    """

    PROMPT = """你是健康助手的安全分流审核器，只判断是否需要升级就医，不做诊断。

用户消息: {message}
规则命中的红旗候选词: {flags}

请根据语义判断用户是否正在描述本人或身边人当前发生的危险信号。

分类标准:
- must_escalate: 当前正在发生或刚发生红旗症状，应立即就医/急救。
- ask_clarifying: 有风险词，但是否当前发生、严重程度或主体不清楚，应先追问并提示如有加重及时就医。
- no_escalate: 只是知识咨询、预防、否定、假设、既往转述，未描述当前急症。

只返回 JSON，例如:
{{"action":"must_escalate","reason":"用户自述当前胸闷并伴随呼吸不适"}}
"""

    def __init__(self, client: Any, model: str):
        self._client = client
        self._model = model

    async def classify(self, text: str) -> SafetyCheckResult:
        rule = detect_red_flag_candidates(text)
        if rule.action in ("must_escalate", "no_escalate"):
            return rule
        return await self._llm_review(text, rule)

    async def _llm_review(self, text: str, rule: SafetyCheckResult) -> SafetyCheckResult:
        import json

        prompt = self.PROMPT.format(
            message=_clean_text(text),
            flags=", ".join(rule.matched_flags) or "无",
        )
        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=256,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[s:e])
            action = str(data.get("action") or "ask_clarifying")
            if action not in {"must_escalate", "ask_clarifying", "no_escalate"}:
                action = "ask_clarifying"
            return SafetyCheckResult(
                risk_level="high" if action == "must_escalate" else "low",
                matched_flags=rule.matched_flags,
                action=action,
                source="llm",
                reason=str(data.get("reason") or "LLM 完成语义安全复核"),
            )
        except Exception as ex:
            return SafetyCheckResult(
                risk_level="low",
                matched_flags=rule.matched_flags,
                action="ask_clarifying",
                source="fallback",
                reason="LLM 安全复核失败，弱红旗候选降级为追问",
                llm_failed=True,
                error=str(ex),
            )


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.encode("utf-8", errors="ignore").decode("utf-8")
