"""三路融合意图识别器（core/intent_recognizer.py）单元测试。

全部为离线测试：同步策略直接测，异步链路通过 monkeypatch 替换 LLM 调用。
"""
import pytest

from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel


@pytest.fixture
def rec():
    # 不传 base_url → embedding 路启用（本地 n-gram 向量），构造无网络开销
    return IntentRecognizer(api_key="test-key")


@pytest.fixture
def rec_no_emb():
    # 传 base_url → embedding 路禁用，权重转移到 LLM + Pattern
    return IntentRecognizer(api_key="test-key", base_url="http://localhost:9999")


class TestPatternRecognize:
    def test_escalation_keywords(self, rec):
        assert rec._pattern_recognize("我突然胸痛喘不上气")["intent"] == IntentCategory.ESCALATION

    def test_calculate_keywords(self, rec):
        assert rec._pattern_recognize("帮我算一下BMI")["intent"] == IntentCategory.CALCULATE

    def test_no_match_returns_other(self, rec):
        r = rec._pattern_recognize("量子力学是什么")
        assert r["intent"] == IntentCategory.OTHER
        assert r["confidence"] == 0.0

    def test_case_insensitive(self, rec):
        assert rec._pattern_recognize("帮我算bmi")["intent"] == IntentCategory.CALCULATE


class TestUrgency:
    def test_red_flag_is_critical(self, rec):
        assert rec._urgency("我胸痛得厉害", IntentCategory.CONSULT) == UrgencyLevel.CRITICAL

    def test_escalation_intent_always_critical(self, rec):
        assert rec._urgency("身体不太舒服", IntentCategory.ESCALATION) == UrgencyLevel.CRITICAL

    def test_normal_is_low(self, rec):
        assert rec._urgency("吃什么养胃", IntentCategory.NUTRITION) == UrgencyLevel.LOW

    def test_high_urgency_keywords(self, rec):
        assert rec._urgency("请立刻回复我", IntentCategory.CONSULT) == UrgencyLevel.HIGH


class TestVote:
    def test_llm_dominates(self, rec):
        llm = {"intent": IntentCategory.NUTRITION, "confidence": 0.9}
        emb = {"intent": IntentCategory.CONSULT, "confidence": 0.6}
        pat = {"intent": IntentCategory.CONSULT, "confidence": 0.5}
        assert rec._vote(llm, emb, pat) == IntentCategory.NUTRITION

    def test_low_confidence_degrades_to_other(self, rec):
        llm = {"intent": IntentCategory.NUTRITION, "confidence": 0.3}
        emb = {"intent": IntentCategory.OTHER, "confidence": 0.0}
        pat = {"intent": IntentCategory.OTHER, "confidence": 0.0}
        assert rec._vote(llm, emb, pat) == IntentCategory.OTHER

    def test_llm_failure_falls_back_to_embedding(self, rec):
        llm = {"intent": IntentCategory.OTHER, "confidence": 0.0, "failed": True}
        emb = {"intent": IntentCategory.FITNESS, "confidence": 0.8}
        pat = {"intent": IntentCategory.OTHER, "confidence": 0.0}
        assert rec._vote(llm, emb, pat) == IntentCategory.FITNESS

    def test_llm_failure_falls_back_to_pattern(self, rec_no_emb):
        llm = {"intent": IntentCategory.OTHER, "confidence": 0.0, "failed": True}
        emb = {"intent": IntentCategory.OTHER, "confidence": 0.0}
        pat = {"intent": IntentCategory.CALCULATE, "confidence": 0.5}
        assert rec_no_emb._vote(llm, emb, pat) == IntentCategory.CALCULATE

    def test_all_failed_returns_other(self, rec):
        failed = {"intent": IntentCategory.OTHER, "confidence": 0.0}
        assert rec._vote({**failed, "failed": True}, failed, failed) == IntentCategory.OTHER


class TestLocalEmbedding:
    def test_deterministic(self):
        v1 = IntentRecognizer._local_embedding("湿气重怎么调理")
        v2 = IntentRecognizer._local_embedding("湿气重怎么调理")
        assert v1 == v2

    def test_dimension(self):
        assert len(IntentRecognizer._local_embedding("测试", dims=128)) == 128

    def test_similar_texts_score_higher(self, rec):
        from core.intent_recognizer import _cosine
        a = IntentRecognizer._local_embedding("晚上睡不着怎么调理")
        b = IntentRecognizer._local_embedding("晚上总是睡不着怎么办")
        c = IntentRecognizer._local_embedding("帮我算一下BMI指数")
        assert _cosine(a, b) > _cosine(a, c)


class TestRecognizePipeline:
    async def test_full_pipeline_with_mocked_llm(self, rec, monkeypatch):
        async def fake_llm(message, history):
            return {"intent": IntentCategory.NUTRITION, "confidence": 0.9, "reasoning": "mock"}

        async def fake_entities(message):
            return {"symptom": [], "body_part": ["胃"], "age": [], "height": [], "weight": [], "goal": []}

        monkeypatch.setattr(rec, "_llm_recognize", fake_llm)
        monkeypatch.setattr(rec, "_extract_entities", fake_entities)

        result = await rec.recognize("吃什么可以养胃？")
        assert result.intent == IntentCategory.NUTRITION
        assert result.entities["body_part"] == ["胃"]
        assert result.latency_ms >= 0

    async def test_cache_hit(self, rec, monkeypatch):
        calls = {"n": 0}

        async def fake_llm(message, history):
            calls["n"] += 1
            return {"intent": IntentCategory.GREETING, "confidence": 0.9, "reasoning": ""}

        async def fake_entities(message):
            return {}

        monkeypatch.setattr(rec, "_llm_recognize", fake_llm)
        monkeypatch.setattr(rec, "_extract_entities", fake_entities)

        await rec.recognize("你好")
        await rec.recognize("你好")
        assert calls["n"] == 1
        assert rec.cache_stats["hits"] == 1
        assert rec.cache_stats["hit_rate"] == 0.5

    def test_learn_appends_template(self, rec):
        from core.intent_recognizer import _TEMPLATES
        before = len(_TEMPLATES[IntentCategory.CONSULT])
        rec.learn("我脾虚平时要注意什么", IntentCategory.CONSULT)
        assert len(_TEMPLATES[IntentCategory.CONSULT]) == before + 1
        # 重复学习不会重复添加
        rec.learn("我脾虚平时要注意什么", IntentCategory.CONSULT)
        assert len(_TEMPLATES[IntentCategory.CONSULT]) == before + 1


class TestCleanText:
    def test_strips_surrogate_chars(self):
        dirty = "你好\ud800世界"
        assert IntentRecognizer._clean_text(dirty) == "你好世界"

    def test_none_and_non_str(self):
        assert IntentRecognizer._clean_text(None) == ""
        assert IntentRecognizer._clean_text(123) == "123"
