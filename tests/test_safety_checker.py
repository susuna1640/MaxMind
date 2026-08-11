"""红旗症状安全检查（core/safety_checker.py）单元测试。

覆盖：命中、未命中、空输入、多关键词命中。
"""
from core.safety_checker import RED_FLAG_KEYWORDS, SafetyCheckResult, detect_red_flags


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
        # 词表中每个关键词单独出现时都必须命中
        for kw in RED_FLAG_KEYWORDS:
            r = detect_red_flags(f"我感觉{kw}怎么办")
            assert r.is_high_risk, f"关键词 {kw} 未被检出"

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
