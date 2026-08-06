"""
亮点：端到端意图识别

三路融合策略：
  1. LLM 语义理解（权重 70%）—— 主力，理解复杂语义和上下文
  2. Embedding 向量相似度（权重 20%）—— 快速匹配常见表达
  3. 关键词模式匹配（权重 10%）—— 零延迟兜底

三路结果通过加权投票合并，置信度低于阈值时降级为 OTHER。
LLM 和 Embedding 并行调用，不串行等待。
"""
import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)


class IntentCategory(Enum):
    CONSULT    = "consult"     # 一般健康咨询（症状调理、体质疑问）
    NUTRITION  = "nutrition"   # 饮食/营养（养胃、食疗、减脂饮食）
    FITNESS    = "fitness"     # 运动/睡眠（锻炼、作息、失眠调理）
    CALCULATE  = "calculate"   # 健康计算（BMI、饮水量、作息、运动心率）
    PROFILE    = "profile"     # 记录/更新个人健康档案
    ESCALATION = "escalation"  # 急症信号/就医预警
    GREETING   = "greeting"    # 问候
    OTHER      = "other"


class UrgencyLevel(Enum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


@dataclass
class IntentResult:
    intent:     IntentCategory
    confidence: float
    urgency:    UrgencyLevel
    entities:   Dict[str, List[str]]   # 从消息中提取的实体
    reasoning:  str
    latency_ms: float


# ── Few-shot 模板（同时用于 LLM 示例和 Embedding 匹配）────────────────────────
_TEMPLATES: Dict[IntentCategory, List[str]] = {
    IntentCategory.CONSULT:    ["湿气重应该怎么调理？", "最近总觉得很累是什么原因？", "体质偏寒平时要注意什么？"],
    IntentCategory.NUTRITION:  ["吃什么可以养胃？", "减脂期间应该怎么吃？", "养肝吃什么食物好？"],
    IntentCategory.FITNESS:    ["晚上睡不着怎么调理？", "每天跑步多少合适？", "如何制定减脂运动计划？"],
    IntentCategory.CALCULATE:  ["身高170体重70，帮我算一下BMI", "体重60kg每天喝多少水合适？", "早上7点起床，晚上几点睡比较好？"],
    IntentCategory.PROFILE:    ["记一下我今年30岁，身高170", "把我对海鲜过敏记到健康档案里", "更新我的体重为65公斤"],
    IntentCategory.ESCALATION: ["我胸口疼得厉害还喘不上气", "高烧三天一直不退", "突然剧烈头痛还呕吐"],
    IntentCategory.GREETING:   ["你好", "嗨，有人吗", "早上好"],
}

# 紧急关键词：CRITICAL 由红旗症状（危险信号）承担，命中即触发就医预警
_URGENCY_KEYWORDS = {
    UrgencyLevel.CRITICAL: ["胸痛", "胸闷", "呼吸困难", "昏迷", "抽搐", "大出血", "吐血",
                            "咯血", "黑便", "剧烈头痛", "偏瘫", "高烧不退", "意识模糊", "休克"],
    UrgencyLevel.HIGH:     ["紧急", "emergency", "urgent", "asap", "立刻", "马上就医"],
    UrgencyLevel.MEDIUM:   ["今天", "马上", "尽快", "hurry", "now"],
}


def _cosine(a: List[float], b: List[float]) -> float:
    """纯 Python 余弦相似度，不依赖 numpy。"""
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class IntentRecognizer:
    """
    端到端意图识别器。

    初始化时不加载任何本地模型，所有 AI 能力通过 Anthropic API 调用。
    模板 Embedding 在首次请求时懒加载并缓存，后续复用。
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        confidence_threshold: float = 0.5,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        from mcp.tool_manager import build_llm_client  # 延迟导入避免循环依赖
        self.client    = build_llm_client(api_key, base_url)
        self.model     = model
        self.threshold = confidence_threshold
        # 第三方兼容 API（如 DeepSeek）通常不支持 Embedding，禁用该策略。
        # 官方 Anthropic SDK 当前没有 embeddings 资源，因此下面会使用稳定的
        # 本地字符 n-gram 向量作为轻量兜底，保证三路融合链路真实可跑。
        self._embedding_enabled = not bool(base_url)

        self._tpl_embeddings: Dict[IntentCategory, List[List[float]]] = {}
        self._cache: Dict[str, IntentResult] = {}
        self.cache_hits   = 0
        self.cache_misses = 0

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    async def recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> IntentResult:
        """
        识别用户意图。

        history 格式：[{"role": "user"/"assistant", "content": "..."}]
        """
        # ★ 意图识别入口
        print(f"[FLOW]   \u251c\u2500 意图识别开始: \"{message}\"")
        key = self._cache_key(message)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1

        t0 = time.monotonic()

        # LLM 和 Embedding 并行（Embedding 不可用时跳过）
        llm_task = asyncio.create_task(self._llm_recognize(message, history))
        emb_task = asyncio.create_task(self._embedding_recognize(message)) if self._embedding_enabled else None
        pat      = self._pattern_recognize(message)

        if emb_task:
            llm, emb = await asyncio.gather(llm_task, emb_task)
        else:
            llm = await llm_task
            emb = {"intent": IntentCategory.OTHER, "confidence": 0.0}

        # ★ 三路结果投票合并
        print(f"[FLOW]   \u251c\u2500 三路结果:")
        print(f"[FLOW]   \u2502   LLM:     {llm.get('intent', 'OTHER')} (置信度 {llm.get('confidence', 0):.2f})")
        print(f"[FLOW]   \u2502   Embedding: {emb.get('intent', 'OTHER')} (置信度 {emb.get('confidence', 0):.2f})")
        print(f"[FLOW]   \u2502   Pattern:   {pat.get('intent', 'OTHER')} (置信度 {pat.get('confidence', 0):.2f})")
        intent = self._vote(llm, emb, pat)
        entities = await self._extract_entities(message)
        urgency  = self._urgency(message, intent)
        print(f"[FLOW]   \u2514\u2500 投票结果: intent={intent.value}, urgency={urgency.name}, 耗时 {(time.monotonic()-t0)*1000:.0f}ms")

        result = IntentResult(
            intent=intent,
            confidence=llm["confidence"],
            urgency=urgency,
            entities=entities,
            reasoning=llm.get("reasoning", ""),
            latency_ms=(time.monotonic() - t0) * 1000,
        )

        # LRU 缓存
        if len(self._cache) >= 1000:
            for k in list(self._cache)[:500]:
                del self._cache[k]
        self._cache[key] = result
        return result

    def learn(self, message: str, correct: IntentCategory) -> None:
        """在线学习：将纠正样本加入模板，清除对应 Embedding 缓存。"""
        tpls = _TEMPLATES.setdefault(correct, [])
        if message not in tpls:
            tpls.append(message)
            self._tpl_embeddings.pop(correct, None)  # 下次重新计算
            logger.info(f"学习新样本 → {correct.value}: {message[:40]}")

    # ── 三路识别策略 ──────────────────────────────────────────────────────────

    async def _llm_recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        """策略 1：LLM 语义理解（Few-shot + 上下文）。"""
        message = self._clean_text(message)
        # 构建 Few-shot 示例
        examples = "\n".join(
            f'  消息: "{t}" → 意图: {cat.value}'
            for cat, tpls in _TEMPLATES.items()
            for t in tpls[:1]  # 每类取 1 条，控制 prompt 长度
        )
        # 最近 3 轮对话上下文
        ctx = ""
        if history:
            ctx = "\n最近对话:\n" + "\n".join(
                f"  {self._clean_text(m.get('role', 'user'))}: {self._clean_text(m.get('content', ''))}"
                for m in history[-3:]
            )

        prompt = f"""你是健康管理意图分析专家。根据示例判断用户意图，返回 JSON。

示例:
{examples}

{ctx}
用户消息: "{message}"

返回格式（仅 JSON，不要其他文字）:
{{"intent": "<意图值>", "confidence": <0-1>, "reasoning": "<一句话说明>"}}

可选意图: {", ".join(c.value for c in IntentCategory)}"""
        prompt = self._clean_text(prompt)

        print(f"[FLOW]   \u2502   \u251c\u2500 [意图识别 Prompt] (发送给 LLM):")
        for line in prompt.split('\n')[:8]:
            print(f"[FLOW]   \u2502   \u2502   {line}")
        if len(prompt.split('\n')) > 8:
            print(f"[FLOW]   \u2502   \u2502   ... (共 {len(prompt.split(chr(10)))} 行)")

        try:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=256,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text
            print(f"[FLOW]   \u2502   \u2514\u2500 [LLM 返回]: {raw[:100]}")
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[s:e])
            try:
                data["intent"] = IntentCategory(data["intent"])
            except ValueError:
                data["intent"] = IntentCategory.OTHER
            return data
        except Exception as ex:
            logger.warning(f"LLM 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0, "reasoning": "LLM 失败", "failed": True}

    async def _embedding_recognize(self, message: str) -> Dict[str, Any]:
        """策略 2：Embedding 向量相似度匹配。"""
        try:
            await self._load_template_embeddings()
            msg_vec = await self._embed_text(message)

            best_cat, best_score = IntentCategory.OTHER, 0.0
            for cat, vecs in self._tpl_embeddings.items():
                score = max(_cosine(msg_vec, v) for v in vecs)
                if score > best_score:
                    best_score, best_cat = score, cat

            return {"intent": best_cat, "confidence": best_score}
        except Exception as ex:
            logger.warning(f"Embedding 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0}

    def _pattern_recognize(self, message: str) -> Dict[str, Any]:
        """策略 3：关键词模式匹配（同步，零延迟兜底）。"""
        msg = message.lower()
        patterns = {
            IntentCategory.ESCALATION: ["胸痛", "胸闷", "呼吸困难", "昏迷", "抽搐", "大出血",
                                        "吐血", "咯血", "黑便", "剧烈头痛", "偏瘫", "高烧不退",
                                        "意识模糊", "急诊"],
            IntentCategory.CALCULATE:  ["bmi", "BMI", "体重指数", "喝多少水", "饮水量", "几点睡",
                                        "入睡时间", "心率", "卡路里"],
            IntentCategory.NUTRITION:  ["养胃", "养肝", "祛湿", "湿气", "食疗", "减肥吃", "减脂吃",
                                        "怎么吃", "吃什么", "营养", "食谱", "忌口"],
            IntentCategory.FITNESS:    ["失眠", "睡不着", "作息", "熬夜", "跑步", "锻炼", "健身",
                                        "运动计划", "减脂运动", "睡眠"],
            IntentCategory.PROFILE:    ["记一下", "记住我", "健康档案", "我的身高", "我的体重",
                                        "过敏史", "慢性病史", "更新我的"],
            IntentCategory.CONSULT:    ["怎么调理", "怎么办", "什么原因", "需要注意什么", "体质"],
            IntentCategory.GREETING:   ["你好", "嗨", "hello", "hi"],
        }
        best_cat, best_score = IntentCategory.OTHER, 0.0
        for cat, kws in patterns.items():
            hits = sum(1 for kw in kws if kw in msg)
            if hits:
                score = hits / len(kws)
                if score > best_score:
                    best_score, best_cat = score, cat
        return {"intent": best_cat, "confidence": best_score}

    # ── 投票合并 ──────────────────────────────────────────────────────────────

    def _vote(self, llm: Dict, emb: Dict, pat: Dict) -> IntentCategory:
        """加权投票。embedding 不可用时权重自动转移到 LLM 和 Pattern。"""
        if llm.get("failed"):
            if emb.get("intent") != IntentCategory.OTHER and emb.get("confidence", 0.0) > 0:
                return emb["intent"]
            if pat.get("intent") != IntentCategory.OTHER and pat.get("confidence", 0.0) > 0:
                return pat["intent"]
            return IntentCategory.OTHER

        if self._embedding_enabled:
            weights = [(llm, 0.7), (emb, 0.2), (pat, 0.1)]
        else:
            weights = [(llm, 0.85), (pat, 0.15)]
        scores: Dict[IntentCategory, float] = {}
        for result, w in weights:
            cat  = result.get("intent", IntentCategory.OTHER)
            conf = result.get("confidence", 0.0)
            scores[cat] = scores.get(cat, 0.0) + w * conf

        best = max(scores, key=scores.get)  # type: ignore
        return best if scores[best] >= self.threshold else IntentCategory.OTHER

    # ── 实体提取 ──────────────────────────────────────────────────────────────

    async def _extract_entities(self, message: str) -> Dict[str, List[str]]:
        """用 LLM 从消息中提取结构化实体。"""
        message = self._clean_text(message)
        prompt = f"""从健康咨询消息中提取实体，返回 JSON（字段值为列表，没有则为空列表）:
消息: "{message}"
格式: {{"symptom":[],"body_part":[],"age":[],"height":[],"weight":[],"goal":[]}}"""
        prompt = self._clean_text(prompt)
        try:
            resp = await self.client.messages.create(
                model=self.model, max_tokens=256, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text
            s, e = raw.find("{"), raw.rfind("}") + 1
            return json.loads(raw[s:e])
        except Exception:
            return {"symptom": [], "body_part": [], "age": [], "height": [], "weight": [], "goal": []}

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    async def _load_template_embeddings(self) -> None:
        """懒加载所有模板的 Embedding（只在首次调用时执行）。"""
        missing = [cat for cat in _TEMPLATES if cat not in self._tpl_embeddings]
        if not missing:
            return

        all_texts = [t for cat in missing for t in _TEMPLATES[cat]]
        vecs = [await self._embed_text(text) for text in all_texts]
        idx = 0
        for cat in missing:
            n = len(_TEMPLATES[cat])
            self._tpl_embeddings[cat] = vecs[idx: idx + n]
            idx += n

    async def _embed_text(self, text: str) -> List[float]:
        """
        生成文本向量。

        如果未来接入的官方/兼容客户端提供 embeddings.create，会优先使用远端向量；
        当前 Anthropic SDK 没有该资源时，退化为字符 n-gram 哈希向量。这样不会因为
        Embedding 服务缺失导致三路融合中断。
        """
        embeddings = getattr(self.client, "embeddings", None)
        if embeddings is not None:
            try:
                resp = await embeddings.create(model="voyage-3-lite", input=[text])
                return list(resp.data[0].embedding)
            except Exception as ex:
                logger.warning(f"远端 Embedding 失败，使用本地向量兜底: {ex}")

        return self._local_embedding(text)

    @staticmethod
    def _local_embedding(text: str, dims: int = 256) -> List[float]:
        """稳定的字符 n-gram 哈希向量，用于无远端 Embedding 时的语义近似匹配。"""
        normalized = text.lower().strip()
        vec = [0.0] * dims
        tokens = set()
        for n in (1, 2, 3):
            if len(normalized) >= n:
                tokens.update(normalized[i:i + n] for i in range(len(normalized) - n + 1))
        if not tokens:
            tokens.add(normalized)

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        return vec

    def _urgency(self, message: str, intent: IntentCategory) -> UrgencyLevel:
        msg = message.lower()
        for level, kws in _URGENCY_KEYWORDS.items():
            if any(kw.lower() in msg for kw in kws):
                return level
        # 就医预警类意图一律按最高紧急度处理，确保路由到就医预警链路
        if intent == IntentCategory.ESCALATION:
            return UrgencyLevel.CRITICAL
        return UrgencyLevel.LOW

    def _cache_key(self, message: str) -> str:
        return self._clean_text(message)[:200]

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 HTTP 客户端编码 prompt 时崩溃。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @property
    def cache_stats(self) -> Dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        return {
            "size": len(self._cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / total if total else 0.0,
        }
