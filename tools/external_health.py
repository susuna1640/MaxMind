"""
亮点：外部环境健康工具（真实外部 API 依赖）

数据源：Open-Meteo（免费、无需 API Key）
  - 空气质量: air-quality-api.open-meteo.com（PM2.5 / PM10 / US AQI）
  - 实时天气: api.open-meteo.com（温度 / 体感 / 湿度 / 风速）

为什么值得放进来：
  外部 API 会超时、会挂——正好让 MCPToolManager 的熔断器、超时控制、
  fallback 降级在真实场景下发挥作用（区别于本地确定性计算工具）。
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

AIR_URL     = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_S   = 10.0   # Open-Meteo 偶发慢响应（单请求可达 6-8s），超时留足余量
MAX_RETRIES = 2      # 失败重试次数，抵御外部 API 抖动，避免误熔断

# 常见城市经纬度（无外部地理编码依赖，查不到时默认北京）
CITIES: Dict[str, Tuple[float, float]] = {
    "北京": (39.90, 116.41), "上海": (31.23, 121.47), "广州": (23.13, 113.26),
    "深圳": (22.54, 114.06), "杭州": (30.27, 120.16), "成都": (30.57, 104.07),
    "武汉": (30.59, 114.31), "西安": (34.34, 108.94), "南京": (32.06, 118.80),
    "重庆": (29.56, 106.55), "长沙": (28.23, 112.94), "天津": (39.08, 117.20),
}
DEFAULT_CITY = "北京"

TRIGGER_KEYWORDS = (
    "空气质量", "空气", "雾霾", "pm2.5", "aqi", "污染",
    "天气", "户外", "适合跑步", "适合运动", "出门运动",
)


def should_trigger(text: str) -> bool:
    return any(kw in (text or "").lower() for kw in TRIGGER_KEYWORDS)


def extract_city(text: str) -> str:
    for city in CITIES:
        if city in (text or ""):
            return city
    return DEFAULT_CITY


def _aqi_level(aqi: Optional[float]) -> Tuple[str, str]:
    """US AQI 分级 → (等级, 户外活动建议)。"""
    if aqi is None:
        return "未知", "空气质量数据缺失，户外活动请以自身感受为准。"
    if aqi <= 50:
        return "优", "空气清新，适合户外活动与运动。"
    if aqi <= 100:
        return "良", "可以正常户外活动，敏感人群适当减少长时间剧烈运动。"
    if aqi <= 150:
        return "轻度污染", "建议缩短户外剧烈运动时间，敏感人群改室内训练。"
    if aqi <= 200:
        return "中度污染", "不建议户外跑步，优先室内运动并佩戴口罩。"
    return "重度污染", "应避免户外活动，运动请全部安排在室内。"


async def fetch_env_report(city: str) -> Dict[str, Any]:
    """并行拉取空气质量与天气，合并为一份环境健康报告。"""
    lat, lon = CITIES.get(city, CITIES[DEFAULT_CITY])
    params_air = {
        "latitude": lat, "longitude": lon,
        "current": "pm2_5,pm10,us_aqi",
    }
    params_weather = {
        "latitude": lat, "longitude": lon,
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m",
    }

    # trust_env=False：直连外部 API，不读 HTTP(S)_PROXY 环境变量。
    # 本机代理（如 127.0.0.1:7897）时断时续，走代理会让请求超时/抖动导致工具熔断
    last_exc: Optional[Exception] = None
    async with httpx.AsyncClient(timeout=TIMEOUT_S, trust_env=False) as client:
        for attempt in range(MAX_RETRIES + 1):
            try:
                air_task = client.get(AIR_URL, params=params_air)
                weather_task = client.get(WEATHER_URL, params=params_weather)
                air_resp, weather_resp = await asyncio.gather(air_task, weather_task)
                air_resp.raise_for_status()
                weather_resp.raise_for_status()
                break
            except Exception as ex:
                last_exc = ex
                logger.warning(f"外部环境 API 第 {attempt + 1} 次请求失败: {ex}")
                if attempt == MAX_RETRIES:
                    raise
                await asyncio.sleep(0.5)

    air_cur = air_resp.json().get("current", {})
    wx_cur = weather_resp.json().get("current", {})

    aqi = air_cur.get("us_aqi")
    level, advice = _aqi_level(aqi)
    report = {
        "city": city,
        "aqi": aqi,
        "aqi_level": level,
        "pm2_5": air_cur.get("pm2_5"),
        "pm10": air_cur.get("pm10"),
        "temperature_c": wx_cur.get("temperature_2m"),
        "feels_like_c": wx_cur.get("apparent_temperature"),
        "humidity_pct": wx_cur.get("relative_humidity_2m"),
        "wind_kmh": wx_cur.get("wind_speed_10m"),
        "outdoor_advice": advice,
        "source": "Open-Meteo (免费开放数据)",
    }
    return report


def format_report(report: Dict[str, Any]) -> str:
    """格式化为注入 Agent 上下文的文本。"""
    return (
        f"城市 {report['city']}：空气质量 {report['aqi_level']}"
        f"（US AQI {report['aqi']}，PM2.5 {report['pm2_5']}，PM10 {report['pm10']}）；"
        f"气温 {report['temperature_c']}°C（体感 {report['feels_like_c']}°C），"
        f"湿度 {report['humidity_pct']}%，风速 {report['wind_kmh']} km/h。"
        f"户外活动建议：{report['outdoor_advice']}（数据来源：{report['source']}）"
    )
