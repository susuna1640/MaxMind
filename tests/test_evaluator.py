"""评测框架（evaluation/evaluator.py）单元测试。

覆盖：意图指标计算、LLM Judge 解析与容错、回归检测、优化建议、基线存取。
全部离线：LLM 客户端用假对象替换。
"""
import json
from types import SimpleNamespace

import pytest

from core.intent_recognizer import IntentCategory, IntentResult, UrgencyLevel
from evaluation.evaluator import (
    EndToEndEvaluator,
    EvalReport,
    EvalResult,
    IntentEvaluator,
    IntentTestCase,
    LLMJudge,
    QualityScores,
)


# ── 测试替身 ──────────────────────────────────────────────────────────────────

class FakeRecognizer:
    """按预设映射返回意图的假识别器。"""

    def __init__(self, mapping):
        self._mapping = mapping

    async def recognize(self, message, history=None):
        intent = IntentCategory(self._mapping.get(message, "other"))
        return IntentResult(
            intent=intent, confidence=0.9, urgency=UrgencyLevel.LOW,
            entities={}, reasoning="fake", latency_ms=1.0,
        )


def _make_report(scores: dict) -> EvalReport:
    return EvalReport(
        timestamp="2026-01-01T00:00:00", total=1, passed=1, pass_rate=1.0,
        avg_scores=scores, regressions=[], recommendations=[], results=[],
    )


@pytest.fixture
def evaluator(tmp_path):
    return EndToEndEvaluator(
        orchestrator=None,
        recognizer=FakeRecognizer({}),
        api_key="test-key",
        baseline_path=str(tmp_path / "baseline.json"),
    )


# ── 意图指标 ──────────────────────────────────────────────────────────────────

class TestIntentEvaluator:
    async def test_perfect_accuracy(self):
        rec = FakeRecognizer({"a": "consult", "b": "nutrition"})
        ev = IntentEvaluator(rec)
        metrics = await ev.evaluate([
            IntentTestCase("a", "consult"),
            IntentTestCase("b", "nutrition"),
        ])
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0

    async def test_partial_accuracy_and_f1(self):
        rec = FakeRecognizer({"a": "consult", "b": "consult"})  # b 预测错
        ev = IntentEvaluator(rec)
        metrics = await ev.evaluate([
            IntentTestCase("a", "consult"),
            IntentTestCase("b", "nutrition"),
        ])
        assert metrics["accuracy"] == 0.5
        assert metrics["correct"] == 1
        # consult: P=1/2, R=1/1, F1=2/3；nutrition: P=0/0=0, R=0/1=0, F1=0
        assert metrics["per_class"]["consult"]["f1"] == pytest.approx(2 / 3, abs=1e-3)
        assert metrics["macro_f1"] < 1.0

    async def test_empty_cases(self):
        ev = IntentEvaluator(FakeRecognizer({}))
        metrics = await ev.evaluate([])
        assert metrics["accuracy"] == 0.0
        assert metrics["total"] == 0


# ── LLM Judge ────────────────────────────────────────────────────────────────

class TestLLMJudge:
    def _judge_with_reply(self, text: str) -> LLMJudge:
        async def create(**kwargs):
            return SimpleNamespace(content=[SimpleNamespace(text=text)])
        client = SimpleNamespace(messages=SimpleNamespace(create=create))
        return LLMJudge(client, model="fake-model")

    async def test_parses_json_scores(self):
        judge = self._judge_with_reply(
            '{"relevance": 0.9, "accuracy": 0.8, "completeness": 0.7, "helpfulness": 0.6}'
        )
        s = await judge.judge("问题", "回答")
        assert not s.judge_failed
        assert s.relevance == 0.9
        assert s.overall == pytest.approx(0.75)

    async def test_tolerates_json_wrapped_in_text(self):
        judge = self._judge_with_reply(
            '评分如下：{"relevance": 1.0, "accuracy": 1.0, "completeness": 1.0, "helpfulness": 1.0} 完毕'
        )
        s = await judge.judge("问题", "回答")
        assert not s.judge_failed and s.overall == 1.0

    async def test_invalid_json_marked_failed(self):
        judge = self._judge_with_reply("这不是 JSON")
        s = await judge.judge("问题", "回答")
        assert s.judge_failed
        assert s.overall == 0.5  # 失败时给中性分


# ── 回归检测 ──────────────────────────────────────────────────────────────────

class TestRegressionDetection:
    def test_detects_regression_over_5_percent(self, evaluator):
        evaluator._baseline = _make_report({"relevance": 0.90})
        reg = evaluator._detect_regressions({"relevance": 0.80})  # -11%
        assert len(reg) == 1 and "relevance" in reg[0]

    def test_small_drop_not_flagged(self, evaluator):
        evaluator._baseline = _make_report({"relevance": 0.90})
        assert evaluator._detect_regressions({"relevance": 0.87}) == []  # -3.3%

    def test_no_baseline_no_regression(self, evaluator):
        assert evaluator._detect_regressions({"relevance": 0.5}) == []

    def test_new_metric_not_flagged(self, evaluator):
        evaluator._baseline = _make_report({"relevance": 0.90})
        assert evaluator._detect_regressions({"new_metric": 0.1}) == []


# ── 优化建议 ──────────────────────────────────────────────────────────────────

class TestRecommendations:
    def test_all_pass(self, evaluator):
        recs = evaluator._recommendations(
            {"intent_accuracy": 0.95, "relevance": 0.9, "completeness": 0.9, "helpfulness": 0.9}, {}
        )
        assert recs == ["所有指标均达标，继续保持"]

    def test_low_intent_triggers_suggestion(self, evaluator):
        recs = evaluator._recommendations({"intent_accuracy": 0.7}, {})
        assert any("意图识别" in r for r in recs)

    def test_low_relevance_triggers_suggestion(self, evaluator):
        recs = evaluator._recommendations({"relevance": 0.5}, {})
        assert any("相关性" in r for r in recs)


# ── 基线存取 ──────────────────────────────────────────────────────────────────

class TestBaselinePersistence:
    def test_save_and_reload_roundtrip(self, evaluator):
        report = _make_report({"relevance": 0.88, "intent_accuracy": 0.9})
        report.results = [EvalResult(test_id="dialog_0", passed=True, scores={"overall": 0.88})]
        evaluator._save_baseline(report)

        loaded = evaluator._load_baseline()
        assert loaded is not None
        assert loaded.avg_scores == report.avg_scores
        assert loaded.results[0].test_id == "dialog_0"

    def test_missing_baseline_returns_none(self, evaluator):
        evaluator._baseline_path = None
        assert evaluator._load_baseline() is None

    def test_corrupted_baseline_returns_none(self, evaluator):
        evaluator._baseline_path.write_text("{invalid json", encoding="utf-8")
        assert evaluator._load_baseline() is None


# ── 对话用例解析 ──────────────────────────────────────────────────────────────

class TestDialogTurns:
    def test_single_turn(self):
        assert EndToEndEvaluator._dialog_turns({"question": "你好"}) == ["你好"]

    def test_multi_turn(self):
        assert EndToEndEvaluator._dialog_turns({"turns": ["a", "b"]}) == ["a", "b"]

    def test_empty_case(self):
        assert EndToEndEvaluator._dialog_turns({}) == []
        assert EndToEndEvaluator._dialog_turns({"turns": ["  "]}) == []

    def test_history_context(self):
        ctx = EndToEndEvaluator._history_context([
            {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"},
        ])
        assert "user: q" in ctx and "assistant: a" in ctx
        assert EndToEndEvaluator._history_context([]) == ""


# ── QualityScores ────────────────────────────────────────────────────────────

class TestQualityScores:
    def test_overall_is_mean(self):
        s = QualityScores(relevance=0.8, accuracy=0.6, completeness=1.0, helpfulness=0.8)
        assert s.overall == pytest.approx(0.8)
