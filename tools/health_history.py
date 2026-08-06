"""
亮点：BMI 历史趋势工具（记忆系统 ↔ 工具调用的打通）

机制：
  - 每次 /chat 链路算出 BMI 时，自动把记录写入 Redis（按 user_id 隔离）
  - 用户询问"体重/BMI 趋势、历史、变化"时，生成趋势摘要注入 Agent 上下文

存储结构：Redis List，key = healthmind:bmi_history:{user_id}，
          元素为 JSON 记录（时间倒序），最多保留 MAX_RECORDS 条。
"""
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis

logger = logging.getLogger(__name__)

KEY_PREFIX  = "healthmind:bmi_history"
MAX_RECORDS = 50


class BmiHistoryStore:
    """基于 Redis 的 BMI 历史记录存取。"""

    def __init__(self, redis_url: Optional[str] = None):
        url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._redis = redis.from_url(url, decode_responses=True)

    def _key(self, user_id: str) -> str:
        return f"{KEY_PREFIX}:{user_id}"

    # ── 写入 ──────────────────────────────────────────────────────────────────

    def record(self, user_id: str, height_cm: float, weight_kg: float, bmi: float) -> bool:
        """追加一条 BMI 记录；同一天的重复记录会被覆盖而不是堆积。"""
        try:
            record = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "height_cm": height_cm,
                "weight_kg": weight_kg,
                "bmi": bmi,
            }
            key = self._key(user_id)
            existing = self._read(key)
            # 同一天只保留最新一条（反复测试不堆积）
            if existing and existing[-1]["date"][:10] == record["date"][:10]:
                existing[-1] = record
                self._redis.delete(key)
                for item in reversed(existing):
                    self._redis.rpush(key, json.dumps(item, ensure_ascii=False))
            else:
                self._redis.rpush(key, json.dumps(record, ensure_ascii=False))
                self._redis.ltrim(key, -MAX_RECORDS, -1)
            return True
        except Exception as ex:
            logger.warning(f"BMI 历史记录写入失败: {ex}")
            return False

    # ── 读取与摘要 ────────────────────────────────────────────────────────────

    def history(self, user_id: str) -> List[Dict[str, Any]]:
        """返回按时间正序（旧 → 新）的记录列表。"""
        try:
            return self._read(self._key(user_id))
        except Exception as ex:
            logger.warning(f"BMI 历史记录读取失败: {ex}")
            return []

    def summarize(self, user_id: str) -> Optional[Dict[str, Any]]:
        """生成趋势摘要；无记录时返回 None。"""
        records = self.history(user_id)
        if not records:
            return None

        first, latest = records[0], records[-1]
        diff = round(latest["bmi"] - first["bmi"], 1)
        weight_diff = round(latest["weight_kg"] - first["weight_kg"], 1)
        if diff > 0.5:
            trend = "上升"
        elif diff < -0.5:
            trend = "下降"
        else:
            trend = "基本平稳"

        return {
            "user_id": user_id,
            "record_count": len(records),
            "first": first,
            "latest": latest,
            "bmi_change": diff,
            "weight_change_kg": weight_diff,
            "trend": trend,
            "summary_text": (
                f"共记录 {len(records)} 次：首次（{first['date']}）BMI {first['bmi']} / 体重 {first['weight_kg']:g} kg，"
                f"最近（{latest['date']}）BMI {latest['bmi']} / 体重 {latest['weight_kg']:g} kg，"
                f"BMI 变化 {diff:+}，整体趋势{trend}。"
            ),
            "records": records[-10:],
        }

    def _read(self, key: str) -> List[Dict[str, Any]]:
        items = self._redis.lrange(key, 0, -1)
        records = []
        for raw in items:
            try:
                records.append(json.loads(raw))
            except (TypeError, ValueError):
                continue
        return records
