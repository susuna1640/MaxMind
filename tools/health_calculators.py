"""
亮点：确定性健康计算工具（移植自 rag-health-assistant 的 health_tool_service）

包含四个纯计算工具，全部为确定性逻辑（不依赖 LLM），结果可复现：
  1. BMI 计算 —— 身高 + 体重 → BMI 与体重区间
  2. 饮水量估算 —— 体重 → 每日建议饮水范围
  3. 睡眠作息规划 —— 起床时间 → 反推建议入睡时间（7.5h / 9h）
  4. 运动心率估算 —— 年龄 → 温和有氧目标心率区间

这些工具通过 MCPToolManager 注册后，在 /chat 链路中按意图/关键词触发（方案 A），
计算结果注入 Agent 上下文，保证回复中的数值来自工具而非模型编造。
"""
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


@dataclass
class HealthToolResult:
    """单个健康工具的计算结果。"""
    name:   str
    title:  str
    result: str
    data:   Dict[str, Any]


class HealthCalculators:
    """四个健康计算工具的统一入口。"""

    DISCLAIMER = "以上结果用于健康养生科普和日常参考，不能替代医生或专业营养师的个体化评估。"

    # ── 触发与执行 ────────────────────────────────────────────────────────────

    def run_tools(self, text: str, user_profile: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """根据文本关键词自动触发匹配的计算工具，返回所有命中的结果。"""
        profile = user_profile if isinstance(user_profile, dict) else {}
        text = text or ""
        results: List[HealthToolResult] = []

        bmi = self._maybe_calculate_bmi(text)
        if bmi:
            results.append(bmi)

        water = self._maybe_calculate_water(text)
        if water:
            results.append(water)

        sleep = self._maybe_plan_sleep(text)
        if sleep:
            results.append(sleep)

        exercise = self._maybe_calculate_exercise(text, profile)
        if exercise:
            results.append(exercise)

        return [asdict(item) for item in results]

    def should_trigger(self, text: str) -> bool:
        """粗筛：文本是否包含任何健康计算相关的关键词。"""
        text = (text or "").lower()
        keywords = (
            "bmi", "体重指数", "身高", "体重",
            "喝水", "饮水", "补水", "水量", "多少水",
            "几点睡", "几点起", "起床", "入睡", "作息",
            "运动心率", "有氧", "燃脂", "快走", "跑步",
        )
        return any(kw in text for kw in keywords)

    # ── 结果格式化 ────────────────────────────────────────────────────────────

    def format_for_prompt(self, results: List[Dict[str, Any]]) -> str:
        """把计算结果格式化为注入 Agent 上下文的文本。"""
        if not results:
            return ""
        lines = ["[健康计算工具结果]"]
        for item in results:
            lines.append(f"- {item['title']}：{item['result']}")
        lines.append(f"请直接引用以上工具结果作答，不要自行重新计算。{self.DISCLAIMER}")
        return "\n".join(lines)

    # ── 单项工具实现 ──────────────────────────────────────────────────────────

    def _maybe_calculate_bmi(self, text: str) -> Optional[HealthToolResult]:
        if not any(kw in text for kw in ("BMI", "bmi", "体重指数", "肥胖", "减重", "超重")):
            return None

        height_cm = self._extract_height_cm(text)
        weight_kg = self._extract_weight_kg(text)
        if not height_cm or not weight_kg:
            return None

        height_m = height_cm / 100
        bmi = round(weight_kg / (height_m * height_m), 1)
        if bmi < 18.5:
            category = "偏瘦"
        elif bmi < 24:
            category = "正常范围"
        elif bmi < 28:
            category = "超重"
        else:
            category = "肥胖"

        return HealthToolResult(
            name="bmi_calculator",
            title="BMI 计算工具",
            result=f"身高 {height_cm:g} cm、体重 {weight_kg:g} kg，对应 BMI 约 {bmi}，属于{category}。",
            data={"height_cm": height_cm, "weight_kg": weight_kg, "bmi": bmi, "category": category},
        )

    def _maybe_calculate_water(self, text: str) -> Optional[HealthToolResult]:
        if not any(kw in text for kw in ("喝水", "饮水", "补水", "水量", "多少水", "喝多少")):
            return None

        weight_kg = self._extract_weight_kg(text)
        if not weight_kg:
            return None

        low = round(weight_kg * 30)
        high = round(weight_kg * 35)
        return HealthToolResult(
            name="water_intake_estimator",
            title="饮水量估算工具",
            result=f"按体重 {weight_kg:g} kg 估算，每日温水摄入可参考 {low}-{high} ml，分多次饮用更稳妥。",
            data={"weight_kg": weight_kg, "daily_water_ml_low": low, "daily_water_ml_high": high},
        )

    def _maybe_plan_sleep(self, text: str) -> Optional[HealthToolResult]:
        if not any(kw in text for kw in ("几点睡", "几点起", "起床", "入睡", "睡眠计划", "作息")):
            return None

        wake_time = self._extract_time(text)
        if not wake_time:
            return None

        base = datetime(2000, 1, 2, wake_time[0], wake_time[1])
        bedtime_75 = (base - timedelta(hours=7, minutes=30)).strftime("%H:%M")
        bedtime_90 = (base - timedelta(hours=9)).strftime("%H:%M")
        wake_label = f"{wake_time[0]:02d}:{wake_time[1]:02d}"

        return HealthToolResult(
            name="sleep_schedule_planner",
            title="睡眠作息工具",
            result=f"如果计划 {wake_label} 起床，可参考 {bedtime_75} 入睡获得约 7.5 小时睡眠，或 {bedtime_90} 入睡获得约 9 小时睡眠。",
            data={"wake_time": wake_label, "bedtime_7_5h": bedtime_75, "bedtime_9h": bedtime_90},
        )

    def _maybe_calculate_exercise(self, text: str, profile: Dict[str, Any]) -> Optional[HealthToolResult]:
        if not any(kw in text for kw in ("运动心率", "有氧", "燃脂", "心率", "快走", "跑步")):
            return None

        age = self._extract_age(text) or self._to_number(profile.get("age"))
        if not age or age <= 0:
            return None

        max_hr = max(220 - age, 80)
        low = round(max_hr * 0.5)
        high = round(max_hr * 0.7)
        return HealthToolResult(
            name="exercise_heart_rate_estimator",
            title="运动心率工具",
            result=f"按年龄 {age:g} 岁估算，温和有氧运动目标心率可参考 {low}-{high} 次/分钟，运动中应以能说话但略喘为宜。",
            data={"age": age, "max_heart_rate_estimate": max_hr, "target_low": low, "target_high": high},
        )

    # ── 数值抽取 ──────────────────────────────────────────────────────────────

    def _extract_height_cm(self, text: str) -> Optional[float]:
        patterns = [
            r"身高\s*(\d+(?:\.\d+)?)\s*cm",
            r"身高\s*(\d+(?:\.\d+)?)\s*厘米",
            r"(\d+(?:\.\d+)?)\s*cm",
            r"(\d+(?:\.\d+)?)\s*厘米",
            r"身高\s*(1\.\d{1,2})\s*米",
            r"(1\.\d{1,2})\s*m",
            r"身高\s*[:：]?\s*(1[2-9]\d|2[0-2]\d)(?!\d)",  # 裸数字：身高170
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            value = float(match.group(1))
            return value * 100 if value < 3 else value
        return None

    def _extract_weight_kg(self, text: str) -> Optional[float]:
        patterns = [
            r"体重\s*(\d+(?:\.\d+)?)\s*kg",
            r"体重\s*(\d+(?:\.\d+)?)\s*公斤",
            r"体重\s*(\d+(?:\.\d+)?)\s*千克",
            r"(\d+(?:\.\d+)?)\s*kg",
            r"(\d+(?:\.\d+)?)\s*公斤",
            r"(\d+(?:\.\d+)?)\s*千克",
            r"体重\s*[:：]?\s*(\d{2,3}(?:\.\d+)?)(?!\d)",  # 裸数字：体重70
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None

    def _extract_age(self, text: str) -> Optional[float]:
        patterns = [
            r"年龄\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*岁",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return float(match.group(1))
        return None

    def _extract_time(self, text: str) -> Optional[tuple]:
        patterns = [
            r"(\d{1,2})[:：](\d{1,2})",
            r"(\d{1,2})\s*点\s*(\d{1,2})?\s*分?",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            hour = int(match.group(1))
            minute = int(match.group(2) or 0)
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return hour, minute
        return None

    @staticmethod
    def _to_number(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(str(value).strip())
        except ValueError:
            return None


# 全局单例，供 MCP 工具注册时引用
health_calculators = HealthCalculators()
