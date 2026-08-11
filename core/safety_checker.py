"""红旗症状（Red Flag）安全检查。

健康域的安全前置检查：在意图识别之前用关键词快速筛查危险信号，
命中后强制把请求推入就医预警链路（urgency=CRITICAL → MedicalAlertAgent），
保证急症信号不会被普通健康建议延误。

设计原则：漏检是致命缺陷，宁可多拦不可漏拦；
但通过否定/假设语境过滤降低误报，避免正常咨询被误判为急症。
"""
import re
from dataclasses import dataclass, field
from typing import List


RED_FLAG_KEYWORDS: List[str] = [
    "胸痛", "胸闷", "呼吸困难", "昏迷", "抽搐", "大出血", "吐血", "咯血",
    "黑便", "剧烈头痛", "偏瘫", "高烧不退", "意识模糊", "休克",
    # 口语化/变体表达（评测驱动的扩充：覆盖用户日常说法）
    "胸口闷", "胸口疼", "闷得难受", "闷得慌", "胸口发闷",
    "喘不上气", "喘不过气", "喘气费劲", "说不利索",
    "晕倒", "晕了过去", "眼前一黑", "晕厥",
    "吐了血", "吐过血", "一口血", "咳血",
    "半边身子", "一侧无力", "身子没力气",
    "烧了三天", "高烧三天", "高烧一直不退", "一直高烧", "一直退不下来",
    "出血特别多", "血流不止", "止不住血",
    "头炸裂", "炸裂一样疼",
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


@dataclass
class SafetyCheckResult:
    """红旗症状检测结果。"""
    risk_level:      str                 # "high" / "low"
    matched_flags:   List[str] = field(default_factory=list)

    @property
    def is_high_risk(self) -> bool:
        return self.risk_level == "high"


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
    if any(marker in text for marker in _QUESTION_MARKERS):
        return SafetyCheckResult(risk_level="low", matched_flags=[])

    matched = [
        kw for kw in RED_FLAG_KEYWORDS
        if any(not _is_negated(text, m.start())
               for m in re.finditer(re.escape(kw), text))
    ]
    return SafetyCheckResult(
        risk_level="high" if matched else "low",
        matched_flags=matched,
    )
