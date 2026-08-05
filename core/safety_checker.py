"""
红旗症状（Red Flag）安全检查。

健康域的安全前置检查：在意图识别之前用关键词快速筛查危险信号，
命中后强制把请求推入就医预警链路（urgency=CRITICAL → MedicalAlertAgent），
保证急症信号不会被普通健康建议延误。

词表参考 rag-health-assistant 的 safety_tool，并补充常见急症表达。
"""
from dataclasses import dataclass, field
from typing import List


RED_FLAG_KEYWORDS: List[str] = [
    "胸痛", "胸闷", "呼吸困难", "昏迷", "抽搐", "大出血", "吐血", "咯血",
    "黑便", "剧烈头痛", "偏瘫", "高烧不退", "意识模糊", "休克",
]


@dataclass
class SafetyCheckResult:
    """红旗症状检测结果。"""
    risk_level:      str                 # "high" / "low"
    matched_flags:   List[str] = field(default_factory=list)

    @property
    def is_high_risk(self) -> bool:
        return self.risk_level == "high"


def detect_red_flags(text: str) -> SafetyCheckResult:
    """检测文本中是否包含红旗症状关键词。"""
    text = text or ""
    matched = [kw for kw in RED_FLAG_KEYWORDS if kw in text]
    return SafetyCheckResult(
        risk_level="high" if matched else "low",
        matched_flags=matched,
    )
