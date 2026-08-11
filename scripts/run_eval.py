"""离线评测 CLI：不依赖后端服务，直接运行意图/安全/工具三维评测。

用法：
  python scripts/run_eval.py                # 跑全部三维（意图需要 LLM API）
  python scripts/run_eval.py --skip-intent  # 只跑安全 + 工具（纯离线，秒级）

结果落盘到 data/eval/run_YYYYMMDD_HHMMSS.json，用于优化前后对比。
"""
import argparse
import asyncio
import json
import os
import sys
import pathlib
from datetime import datetime

from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from core.intent_recognizer import IntentRecognizer           # noqa: E402
from evaluation.evaluator import (                            # noqa: E402
    IntentEvaluator, SafetyEvaluator, ToolEvaluator, load_eval_suite,
)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-intent", action="store_true", help="跳过意图评测（不调用 LLM）")
    parser.add_argument("--suite", default=os.getenv("EVAL_SUITE_PATH", "./data/eval/eval_suite.json"))
    args = parser.parse_args()

    suite = load_eval_suite(args.suite)
    if not suite:
        print(f"评测套件不存在: {args.suite}")
        return

    report: dict = {"timestamp": datetime.now().isoformat(), "metrics": {}, "details": {}}

    # ── 安全拦截（离线）──────────────────────────────────────────────────────
    safety = SafetyEvaluator().evaluate(suite["safety_cases"])
    report["metrics"]["safety_recall"] = safety["safety_recall"]
    report["metrics"]["safety_false_positive_rate"] = safety["false_positive_rate"]
    report["details"]["safety"] = {k: v for k, v in safety.items() if k != "cases"}

    # ── 计算工具（离线）──────────────────────────────────────────────────────
    tool = ToolEvaluator().evaluate(suite["tool_cases"])
    report["metrics"]["tool_trigger_accuracy"] = tool["trigger_accuracy"]
    report["metrics"]["tool_value_accuracy"] = tool["value_accuracy"]
    report["details"]["tool"] = {k: v for k, v in tool.items() if k != "cases"}

    # ── 意图识别（需 LLM）────────────────────────────────────────────────────
    if not args.skip_intent:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("未设置 ANTHROPIC_API_KEY，跳过意图评测")
        else:
            recognizer = IntentRecognizer(
                api_key=api_key,
                base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            )
            intent = await IntentEvaluator(recognizer).evaluate(suite["intent_cases"])
            report["metrics"]["intent_accuracy"] = intent["accuracy"]
            report["metrics"]["intent_macro_f1"] = intent["macro_f1"]
            report["metrics"]["intent_latency_p50_ms"] = intent["latency_p50_ms"]
            report["metrics"]["intent_latency_p95_ms"] = intent["latency_p95_ms"]
            wrong = [c for c in intent["cases"] if c["predicted"] != c["expected"]]
            report["details"]["intent"] = {
                "per_class": intent["per_class"],
                "wrong_cases": [
                    {"message": c["message"], "expected": c["expected"], "predicted": c["predicted"]}
                    for c in wrong
                ],
            }

    # ── 输出与落盘 ───────────────────────────────────────────────────────────
    print("=" * 56)
    print(f"评测报告 {report['timestamp']}")
    print("=" * 56)
    for k, v in report["metrics"].items():
        print(f"  {k:32s} = {v}")
    if safety["missed"]:
        print(f"\n安全漏检 ({len(safety['missed'])} 条):")
        for m in safety["missed"]:
            print(f"  - {m}")
    if safety["false_pos"]:
        print(f"\n安全误报 ({len(safety['false_pos'])} 条):")
        for m in safety["false_pos"]:
            print(f"  - {m}")
    if report["details"].get("intent", {}).get("wrong_cases"):
        print(f"\n意图错分 ({len(report['details']['intent']['wrong_cases'])} 条):")
        for c in report["details"]["intent"]["wrong_cases"]:
            print(f"  - {c['message']} | 期望 {c['expected']} → 预测 {c['predicted']}")

    out = ROOT / "data" / "eval" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已保存: {out}")


if __name__ == "__main__":
    asyncio.run(main())
