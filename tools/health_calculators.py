"""
亮点：确定性健康计算工具（移植自 rag-health-assistant 的 health_tool_service）

包含五个纯计算工具，全部为确定性逻辑（不依赖 LLM），结果可复现：
  1. BMI 计算 —— 身高 + 体重 → BMI 与体重区间
  2. 饮水量估算 —— 体重 → 每日建议饮水范围
  3. 睡眠作息规划 —— 起床时间 → 反推建议入睡时间（7.5h / 9h）
  4. 运动心率估算 —— 年龄 → 温和有氧目标心率区间
  5. BMR/TDEE 计算 —— Mifflin-St Jeor 公式 → 基础代谢与每日总消耗

这些工具通过 MCPToolManager 注册后，在 /chat 链路中按意图/关键词触发（方案 A），
计算结果注入 Agent 上下文，保证回复中的数值来自工具而非模型编造。
纯计算方法（compute_*）同时被标准 MCP Server（mcp_server.py）复用导出。
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

        bmr = self._maybe_calculate_bmr_tdee(text, profile)
        if bmr:
            results.append(bmr)

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
            "喝水", "饮水", "补水", "水量", "多少水", "升水", "毫升水",
            "几点睡", "几点起", "起床", "入睡", "作息",
            "运动心率", "有氧", "燃脂", "快走", "跑步",
            "基础代谢", "代谢", "tdee", "消耗热量", "热量缺口",
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

    def _maybe_calculate_bmr_tdee(self, text: str, profile: Dict[str, Any]) -> Optional[HealthToolResult]:
        if not any(kw in text for kw in ("基础代谢", "代谢", "tdee", "TDEE", "消耗热量", "热量缺口", "消耗多少卡")):
            return None

        height_cm = self._extract_height_cm(text) or self._to_number(profile.get("height_cm") or profile.get("height"))
        weight_kg = self._extract_weight_kg(text) or self._to_number(profile.get("weight_kg") or profile.get("weight"))
        age       = self._extract_age(text) or self._to_number(profile.get("age"))
        if not height_cm or not weight_kg or not age:
            return None

        sex = self._extract_sex(text) or str(profile.get("sex") or profile.get("gender") or "")
        sex = "男" if "男" in sex else ("女" if "女" in sex else "")
        if sex not in ("男", "女"):
            return None

        if any(kw in text for kw in ("高强度", "天天练", "重体力")):
            activity = "active"
        elif any(kw in text for kw in ("中等运动", "规律运动", "每周三", "每周运动")):
            activity = "moderate"
        elif any(kw in text for kw in ("轻度运动", "偶尔运动", "每周一", "散步")):
            activity = "light"
        else:
            activity = "sedentary"

        data = self.compute_bmr_tdee(height_cm, weight_kg, age, sex, activity)
        return HealthToolResult(
            name="bmr_tdee_calculator",
            title="基础代谢与每日总消耗工具",
            result=(f"按身高 {height_cm:g} cm、体重 {weight_kg:g} kg、{age:g} 岁、{sex}性估算（Mifflin-St Jeor 公式）："
                    f"基础代谢 BMR 约 {data['bmr_kcal']} kcal/天，{data['activity_label']}下每日总消耗 TDEE 约 {data['tdee_kcal']} kcal/天。"),
            data=data,
        )

    def _maybe_calculate_water(self, text: str) -> Optional[HealthToolResult]:
        if not any(kw in text for kw in ("喝水", "饮水", "补水", "水量", "多少水", "喝多少", "升水", "毫升水")):
            return None

        weight_kg = self._extract_weight_kg(text)
        if not weight_kg:
            return None

        data = self.compute_water(weight_kg)
        return HealthToolResult(
            name="water_intake_estimator",
            title="饮水量估算工具",
            result=f"按体重 {weight_kg:g} kg 估算，每日温水摄入可参考 {data['daily_water_ml_low']}-{data['daily_water_ml_high']} ml，分多次饮用更稳妥。",
            data={"weight_kg": weight_kg, **data},
        )

    def _maybe_plan_sleep(self, text: str) -> Optional[HealthToolResult]:
        if not any(kw in text for kw in ("几点睡", "几点起", "起床", "入睡", "睡眠计划", "作息")):
            return None

        wake_time = self._extract_time(text)
        if not wake_time:
            return None

        data = self.compute_sleep(wake_time[0], wake_time[1])
        return HealthToolResult(
            name="sleep_schedule_planner",
            title="睡眠作息工具",
            result=f"如果计划 {data['wake_time']} 起床，可参考 {data['bedtime_7_5h']} 入睡获得约 7.5 小时睡眠，或 {data['bedtime_9h']} 入睡获得约 9 小时睡眠。",
            data=data,
        )

    def _maybe_calculate_exercise(self, text: str, profile: Dict[str, Any]) -> Optional[HealthToolResult]:
        if not any(kw in text for kw in ("运动心率", "有氧", "燃脂", "心率", "快走", "跑步")):
            return None

        age = self._extract_age(text) or self._to_number(profile.get("age"))
        if not age or age <= 0:
            return None

        data = self.compute_heart_rate(age)
        return HealthToolResult(
            name="exercise_heart_rate_estimator",
            title="运动心率工具",
            result=f"按年龄 {age:g} 岁估算，温和有氧运动目标心率可参考 {data['target_low']}-{data['target_high']} 次/分钟，运动中应以能说话但略喘为宜。",
            data={"age": age, **data},
        )

    # ── 纯计算方法（供 MCP Server 复用导出）───────────────────────────────

    ACTIVITY_FACTORS = {
        "sedentary": (1.2,   "久坐少动"),
        "light":     (1.375, "轻度活动（每周 1-3 次）"),
        "moderate":  (1.55,  "中等活动（每周 3-5 次）"),
        "active":    (1.725, "高强度（每周 6-7 次）"),
    }

    @staticmethod
    def compute_bmi(height_cm: float, weight_kg: float) -> Dict[str, Any]:
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
        return {"height_cm": height_cm, "weight_kg": weight_kg, "bmi": bmi, "category": category}

    @staticmethod
    def compute_water(weight_kg: float) -> Dict[str, Any]:
        return {"daily_water_ml_low": round(weight_kg * 30), "daily_water_ml_high": round(weight_kg * 35)}

    @staticmethod
    def compute_sleep(wake_hour: int, wake_minute: int) -> Dict[str, Any]:
        base = datetime(2000, 1, 2, wake_hour, wake_minute)
        return {
            "wake_time": f"{wake_hour:02d}:{wake_minute:02d}",
            "bedtime_7_5h": (base - timedelta(hours=7, minutes=30)).strftime("%H:%M"),
            "bedtime_9h": (base - timedelta(hours=9)).strftime("%H:%M"),
        }

    @staticmethod
    def compute_heart_rate(age: float) -> Dict[str, Any]:
        max_hr = max(220 - age, 80)
        return {
            "max_heart_rate_estimate": max_hr,
            "target_low": round(max_hr * 0.5),
            "target_high": round(max_hr * 0.7),
        }

    @classmethod
    def compute_bmr_tdee(cls, height_cm: float, weight_kg: float, age: float,
                         sex: str, activity: str = "sedentary") -> Dict[str, Any]:
        """Mifflin-St Jeor 公式：男 +5，女 -161；TDEE = BMR × 活动系数。"""
        if sex not in ("男", "女"):
            raise ValueError("sex 必须是 男 或 女")
        factor, label = cls.ACTIVITY_FACTORS.get(activity, cls.ACTIVITY_FACTORS["sedentary"])
        base = 10 * weight_kg + 6.25 * height_cm - 5 * age
        bmr = round(base + (5 if sex == "男" else -161))
        return {
            "height_cm": height_cm, "weight_kg": weight_kg, "age": age, "sex": sex,
            "activity": activity, "activity_label": label,
            "bmr_kcal": bmr, "tdee_kcal": round(bmr * factor),
            "formula": "Mifflin-St Jeor",
        }

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

    def _extract_sex(self, text: str) -> Optional[str]:
        if "女" in text:
            return "女"
        if "男" in text:
            return "男"
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
