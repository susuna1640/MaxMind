"""确定性健康计算工具（tools/health_calculators.py）单元测试。

覆盖：
  1. 五个纯计算函数的数值正确性与边界
  2. 自然语言数值抽取（身高/体重/时间/年龄/性别）
  3. run_tools 关键词触发链路
"""
import pytest

from tools.health_calculators import HealthCalculators


@pytest.fixture
def calc():
    return HealthCalculators()


class TestComputeBMI:
    def test_normal(self):
        # 注意：分档采用中国卫生行业标准（WS/T 428）：24≤BMI<28 超重，
        # 而非 WHO 标准（25 超重），故 24.2 判为超重
        r = HealthCalculators.compute_bmi(170, 70)
        assert r["bmi"] == 24.2
        assert r["category"] == "超重"

    def test_normal_range(self):
        assert HealthCalculators.compute_bmi(170, 65)["category"] == "正常范围"  # 22.5

    def test_category_boundaries(self):
        # 18.5 / 24 / 28 三个分档边界
        assert HealthCalculators.compute_bmi(170, 52.0)["category"] == "偏瘦"      # 18.0
        assert HealthCalculators.compute_bmi(170, 69.0)["category"] == "正常范围"  # 23.87
        assert HealthCalculators.compute_bmi(170, 69.4)["category"] == "超重"      # 24.0
        assert HealthCalculators.compute_bmi(170, 80.0)["category"] == "超重"      # 27.7
        assert HealthCalculators.compute_bmi(170, 82.0)["category"] == "肥胖"      # 28.4


class TestComputeWater:
    def test_range_formula(self):
        r = HealthCalculators.compute_water(60)
        assert r["daily_water_ml_low"] == 1800
        assert r["daily_water_ml_high"] == 2100


class TestComputeSleep:
    def test_normal_wake_time(self):
        r = HealthCalculators.compute_sleep(6, 30)
        assert r["wake_time"] == "06:30"
        assert r["bedtime_7_5h"] == "23:00"
        assert r["bedtime_9h"] == "21:30"

    def test_cross_midnight(self):
        # 凌晨起床，入睡时间应正确回退到前一天晚上
        r = HealthCalculators.compute_sleep(5, 0)
        assert r["bedtime_7_5h"] == "21:30"
        assert r["bedtime_9h"] == "20:00"


class TestComputeHeartRate:
    def test_typical_age(self):
        r = HealthCalculators.compute_heart_rate(30)
        assert r["max_heart_rate_estimate"] == 190
        assert r["target_low"] == 95
        assert r["target_high"] == 133

    def test_elderly_lower_bound(self):
        # 超高年龄时最大心率有 80 的下限保护
        r = HealthCalculators.compute_heart_rate(150)
        assert r["max_heart_rate_estimate"] == 80


class TestComputeBmrTdee:
    def test_male_known_value(self):
        # Mifflin-St Jeor: 10*70 + 6.25*170 - 5*30 + 5 = 1617.5 → 1618
        r = HealthCalculators.compute_bmr_tdee(170, 70, 30, "男")
        assert r["bmr_kcal"] == 1618
        assert r["tdee_kcal"] == 1942   # 1618 × 1.2

    def test_female_offset(self):
        # 女性 -161: 10*55 + 6.25*160 - 5*25 - 161 = 1264
        r = HealthCalculators.compute_bmr_tdee(160, 55, 25, "女")
        assert r["bmr_kcal"] == 1264
        assert r["formula"] == "Mifflin-St Jeor"

    def test_activity_factor(self):
        sedentary = HealthCalculators.compute_bmr_tdee(170, 70, 30, "男", "sedentary")
        moderate  = HealthCalculators.compute_bmr_tdee(170, 70, 30, "男", "moderate")
        assert moderate["tdee_kcal"] > sedentary["tdee_kcal"]

    def test_invalid_sex_raises(self):
        with pytest.raises(ValueError):
            HealthCalculators.compute_bmr_tdee(170, 70, 30, "未知")


class TestExtractionAndTrigger:
    def test_run_tools_bmi_from_natural_language(self, calc):
        results = calc.run_tools("身高170体重70，帮我算一下BMI")
        assert any(r["name"] == "bmi_calculator" for r in results)
        bmi_result = next(r for r in results if r["name"] == "bmi_calculator")
        assert bmi_result["data"]["bmi"] == 24.2

    def test_run_tools_no_match_returns_empty(self, calc):
        assert calc.run_tools("今天天气怎么样") == []

    def test_height_unit_variants(self, calc):
        assert calc._extract_height_cm("身高175cm") == 175
        assert calc._extract_height_cm("身高1.75米") == 175.0
        assert calc._extract_height_cm("身高170") == 170

    def test_weight_unit_variants(self, calc):
        assert calc._extract_weight_kg("体重70kg") == 70
        assert calc._extract_weight_kg("我70公斤") == 70
        assert calc._extract_weight_kg("体重70") == 70

    def test_time_extraction(self, calc):
        assert calc._extract_time("我早上6:30起床") == (6, 30)
        assert calc._extract_time("6点30起床") == (6, 30)
        assert calc._extract_time("没有时间") is None

    def test_should_trigger_keywords(self, calc):
        assert calc.should_trigger("帮我算BMI")
        assert calc.should_trigger("每天喝多少水")
        assert not calc.should_trigger("养肝吃什么好")

    def test_format_for_prompt(self, calc):
        results = calc.run_tools("身高170体重70算BMI")
        text = calc.format_for_prompt(results)
        assert "[健康计算工具结果]" in text
        assert "BMI" in text
        assert calc.format_for_prompt([]) == ""
