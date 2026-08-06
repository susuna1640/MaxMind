"""
亮点：标准 MCP Server —— 把 HealthMind 的工具能力按 Model Context Protocol 导出

运行方式（stdio 传输，可被 Claude Desktop / Cursor 等标准 MCP 客户端接入）：
    python mcp_server.py

导出工具：
  - bmi_calculator              BMI 计算
  - bmr_tdee_calculator         基础代谢与每日总消耗（Mifflin-St Jeor）
  - water_intake_estimator      每日饮水量估算
  - sleep_schedule_planner      睡眠作息规划
  - exercise_heart_rate_estimator  运动心率区间估算
  - knowledge_search            RAG 知识库检索（连 ChromaDB）

注意：项目本地 mcp/ 目录与官方 mcp SDK 包同名，导入 SDK 时需临时
把项目根目录移出 sys.path，避免命名遮蔽；本地模块改用文件路径加载。
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

_ROOT = str(Path(__file__).parent.resolve())

# ── 先导入官方 MCP SDK（临时屏蔽项目根目录，避免本地 mcp/ 遮蔽）───────────────
_saved_path = sys.path[:]
sys.path = [p for p in sys.path if str(Path(p or ".").resolve()) != _ROOT]
sys.modules.pop("mcp", None)
from mcp.server.fastmcp import FastMCP  # noqa: E402
sys.path = _saved_path


def _load_local(name: str, rel_path: str):
    """按文件路径加载本地模块（绕开 mcp 包名冲突）。"""
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_kb_mod = _load_local("hm_knowledge_base", os.path.join("mcp", "knowledge_base.py"))
_calc_mod = _load_local("hm_health_calculators", os.path.join("tools", "health_calculators.py"))
_calculators = _calc_mod.health_calculators

mcp = FastMCP("HealthMind")


# ── 确定性健康计算工具（纯函数复用 compute_*）────────────────────────────────

@mcp.tool()
def bmi_calculator(height_cm: float, weight_kg: float) -> str:
    """计算 BMI 与体重区间。参数：身高(cm)、体重(kg)。"""
    return json.dumps(_calculators.compute_bmi(height_cm, weight_kg), ensure_ascii=False)


@mcp.tool()
def bmr_tdee_calculator(height_cm: float, weight_kg: float, age: float,
                        sex: str, activity: str = "sedentary") -> str:
    """Mifflin-St Jeor 公式计算基础代谢 BMR 与每日总消耗 TDEE。
    sex: 男/女；activity: sedentary/light/moderate/active。"""
    return json.dumps(
        _calculators.compute_bmr_tdee(height_cm, weight_kg, age, sex, activity),
        ensure_ascii=False,
    )


@mcp.tool()
def water_intake_estimator(weight_kg: float) -> str:
    """按体重估算每日建议饮水量(ml)。"""
    return json.dumps(_calculators.compute_water(weight_kg), ensure_ascii=False)


@mcp.tool()
def sleep_schedule_planner(wake_time: str) -> str:
    """按起床时间反推建议入睡时间（7.5h / 9h）。wake_time 格式 HH:MM，如 07:00。"""
    hour, minute = wake_time.split(":")
    return json.dumps(_calculators.compute_sleep(int(hour), int(minute)), ensure_ascii=False)


@mcp.tool()
def exercise_heart_rate_estimator(age: float) -> str:
    """按年龄估算最大心率与温和有氧目标心率区间。"""
    return json.dumps(_calculators.compute_heart_rate(age), ensure_ascii=False)


# ── RAG 知识库检索（连接 ChromaDB）────────────────────────────────────────────

_kb = None


def _get_kb():
    global _kb
    if _kb is None:
        _kb = _kb_mod.KnowledgeBase(
            chroma_host=os.getenv("CHROMA_HOST", "localhost"),
            chroma_port=int(os.getenv("CHROMA_PORT", "8001")),
            chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma"),
        )
    return _kb


@mcp.tool()
async def knowledge_search(query: str, top_k: int = 3) -> str:
    """在 HealthMind 健康知识库中做语义检索，返回最相关的文档片段。"""
    results = _get_kb().search(query, top_k=top_k)
    return json.dumps(results, ensure_ascii=False, default=str)


if __name__ == "__main__":
    mcp.run()  # 默认 stdio 传输
