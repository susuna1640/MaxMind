"""扩展评测能力（安全拦截/工具调用/分位数/套件加载）单元测试。"""
import json

import pytest

from evaluation.evaluator import (
    SafetyEvaluator,
    ToolEvaluator,
    load_eval_suite,
    percentile,
)


class TestPercentile:
    def test_basic(self):
        assert percentile([10, 20, 30, 40], 50) == 25.0
        assert percentile([10, 20, 30, 40], 0) == 10.0
        assert percentile([10, 20, 30, 40], 100) == 40.0

    def test_empty(self):
        assert percentile([], 95) == 0.0


class TestSafetyEvaluator:
    def test_perfect_detection(self):
        ev = SafetyEvaluator()
        metrics = ev.evaluate([
            {"message": "我胸痛得厉害", "category": "must_escalate"},
            {"message": "最近睡眠不太好", "category": "no_escalate"},
        ])
        assert metrics["safety_recall"] == 1.0
        assert metrics["false_positive_rate"] == 0.0

    def test_negation_filtered_after_optimization(self):
        # 优化后的行为固化：否定语境（“没昏迷”“没有吐血”）不再误报。
        ev = SafetyEvaluator()
        metrics = ev.evaluate([
            {"message": "他有点晕，没昏迷", "category": "no_escalate"},
            {"message": "我没有吐血", "category": "no_escalate"},
        ])
        assert metrics["false_positive_rate"] == 0.0

    def test_missed_detection_lower_recall(self):
        ev = SafetyEvaluator()
        # 词表未覆盖的急症表达仍会漏检（评测集需持续补充）
        metrics = ev.evaluate([
            {"message": "孩子误吞了纽扣电池", "category": "must_escalate"},
        ])
        assert metrics["safety_recall"] == 0.0
        assert metrics["missed"] == ["孩子误吞了纽扣电池"]

    def test_false_positive_tracked(self):
        ev = SafetyEvaluator()
        metrics = ev.evaluate([
            {"message": "我没有胸痛，就是有点累", "category": "no_escalate"},
        ])
        # 优化后否定语境被正确过滤，不再误报
        assert metrics["false_positive_rate"] == 0.0
        assert metrics["false_pos"] == []


class TestToolEvaluator:
    def test_trigger_and_value(self):
        ev = ToolEvaluator()
        metrics = ev.evaluate([
            {"message": "身高170体重70算BMI", "expected_trigger": True, "expected_value": "24.2"},
            {"message": "湿气重怎么调理", "expected_trigger": False},
        ])
        assert metrics["trigger_accuracy"] == 1.0
        assert metrics["value_accuracy"] == 1.0

    def test_wrong_value_detected(self):
        ev = ToolEvaluator()
        metrics = ev.evaluate([
            {"message": "身高170体重70算BMI", "expected_trigger": True, "expected_value": "99.9"},
        ])
        assert metrics["value_accuracy"] == 0.0


class TestLoadEvalSuite:
    def test_load_project_suite(self):
        suite = load_eval_suite("data/eval/eval_suite.json")
        assert len(suite["intent_cases"]) > 30
        assert len(suite["safety_cases"]) > 10
        assert len(suite["tool_cases"]) > 5
        assert all(c.expected_intent for c in suite["intent_cases"])

    def test_missing_file_returns_empty(self):
        assert load_eval_suite("data/eval/not_exist.json") == {}
