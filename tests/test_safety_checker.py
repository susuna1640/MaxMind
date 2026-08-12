"""红旗症状安全检查（core/safety_checker.py）单元测试。

覆盖：命中、未命中、空输入、多关键词命中。
"""
from types import SimpleNamespace

from core.safety_checker import (
    RED_FLAG_KEYWORDS,
    SafetyCheckResult,
    SafetyClassifier,
    detect_red_flag_candidates,
    detect_red_flags,
)


class TestDetectRedFlags:
    def test_high_risk_on_chest_pain(self):
        r = detect_red_flags("我突然胸痛，还有点胸闷")
        assert r.is_high_risk
        assert "胸痛" in r.matched_flags
        assert "胸闷" in r.matched_flags

    def test_high_risk_on_breathing_difficulty(self):
        assert detect_red_flags("呼吸困难喘不上气").is_high_risk

    def test_low_risk_on_normal_consult(self):
        r = detect_red_flags("最近总觉得很累，应该怎么调理？")
        assert not r.is_high_risk
        assert r.matched_flags == []

    def test_empty_and_none_input(self):
        assert not detect_red_flags("").is_high_risk
        assert not detect_red_flags(None).is_high_risk

    def test_keyword_coverage_all_flags_detectable(self):
        # 词表中每个关键词单独出现时都必须进入红旗候选集。
        for kw in RED_FLAG_KEYWORDS:
            r = detect_red_flag_candidates(f"我感觉{kw}怎么办")
            assert kw in r.matched_flags, f"关键词 {kw} 未被检出"

    def test_candidate_review_keeps_question_context_for_llm(self):
        r = detect_red_flag_candidates("怎么预防中暑导致的昏迷")
        assert r.action == "ask_clarifying"
        assert "昏迷" in r.matched_flags

    def test_result_dataclass_fields(self):
        r = detect_red_flags("昏迷")
        assert r.risk_level == "high"
        assert isinstance(r, SafetyCheckResult)


class TestContextFiltering:
    """语境过滤：否定/知识咨询/转述语境不应误报（优化后的行为固化）。"""

    def test_negation_not_triggered(self):
        assert not detect_red_flags("我没有胸痛，只是有点累").is_high_risk
        assert not detect_red_flags("他有点晕，没昏迷").is_high_risk

    def test_knowledge_question_not_triggered(self):
        assert not detect_red_flags("胸痛一般是什么病引起的？").is_high_risk
        assert not detect_red_flags("怎么预防中暑导致的昏迷").is_high_risk

    def test_past_narration_not_triggered(self):
        assert not detect_red_flags("我妈上次胸闷，医生说是胃食管反流").is_high_risk

    def test_real_symptom_still_caught(self):
        # 症状自述即使带问号也必须拦截，宁多拦不漏检
        assert detect_red_flags("我胸痛得厉害，是怎么回事？要不要去医院").is_high_risk
        assert detect_red_flags("最近没有运动，胸口闷闷的").is_high_risk
        assert detect_red_flags("胸口有点闷但还能正常说话，我想先做个呼吸练习").is_high_risk

    def test_hard_red_flag_variants(self):
        assert detect_red_flags("胸口堵得慌，感觉气不够用").is_high_risk
        assert detect_red_flags("突然嘴角歪了，说话也含糊").is_high_risk
        assert detect_red_flags("便便黑得像柏油一样，还头晕").is_high_risk
        assert detect_red_flags("孩子把纽扣电池吞下去了").is_high_risk


class TestSafetyClassifier:
    def _classifier_with_reply(self, text: str) -> SafetyClassifier:
        async def create(**kwargs):
            return SimpleNamespace(content=[SimpleNamespace(text=text)])
        client = SimpleNamespace(messages=SimpleNamespace(create=create))
        return SafetyClassifier(client, model="fake-model")

    async def test_llm_review_can_escalate_weak_flag(self):
        clf = self._classifier_with_reply(
            '{"action":"must_escalate","reason":"当前胸闷自述"}'
        )
        result = await clf.classify("胸口有点闷，是怎么回事")
        assert result.is_high_risk
        assert result.source == "llm"

    async def test_llm_review_can_release_knowledge_question(self):
        clf = self._classifier_with_reply(
            '{"action":"no_escalate","reason":"知识咨询"}'
        )
        result = await clf.classify("胸痛一般是什么病引起的？")
        assert not result.is_high_risk
        assert result.action == "no_escalate"

    async def test_llm_failure_falls_back_to_clarifying(self):
        async def create(**kwargs):
            raise RuntimeError("boom")
        client = SimpleNamespace(messages=SimpleNamespace(create=create))
        result = await SafetyClassifier(client, model="fake-model").classify("胸口有点闷")
        assert result.action == "ask_clarifying"
        assert result.source == "fallback"
        assert result.llm_failed
