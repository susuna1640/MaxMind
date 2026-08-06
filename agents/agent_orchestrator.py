"""
亮点：多 Agent 路由与编排

核心问题：多 Agent 情况下如何做 Routing？

路由策略（三层决策）：
  1. 意图路由 —— 根据 IntentCategory 直接映射到专属 Agent
  2. 性能路由 —— 同类 Agent 有多个时，选成功率最高、延迟最低的
  3. 降级路由 —— 专属 Agent 不可用时，自动降级到 HealthAgent

并行协作：
  - 复杂问题（如"减脂 = 饮食 + 运动"）可同时派发给多个 Agent
  - 结果由 Orchestrator 合并后返回

升级机制：
  - 红旗症状/急症信号 → 路由到就医预警 Agent，优先建议线下就医
  - Agent 置信度低或处理失败 → 降级到 HealthAgent 兜底
"""
import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel
from mcp.tool_manager import build_llm_client

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

class AgentType(Enum):
    HEALTH     = "health"      # 通用健康顾问（默认）
    NUTRITION  = "nutrition"   # 饮食/营养顾问
    FITNESS    = "fitness"     # 运动/睡眠顾问
    ESCALATION = "escalation"  # 就医预警（红旗症状/急症）


@dataclass
class AgentStats:
    """Agent 运行时统计，供 Monitor 和路由决策使用。"""
    total:     int   = 0
    success:   int   = 0
    total_ms:  float = 0.0
    monitor_penalty: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.total if self.total else 0.0

    def routing_score(self) -> float:
        """路由评分：成功率高、延迟低的 Agent 得分高。"""
        latency_score = 1.0 / (1.0 + self.avg_ms / 1000)
        base_score = self.success_rate * 0.7 + latency_score * 0.3
        return base_score * max(0.0, 1.0 - self.monitor_penalty)


@dataclass
class AgentResponse:
    agent_type:  AgentType
    content:     str
    success:     bool
    confidence:  float = 1.0
    latency_ms:  float = 0.0
    escalate:    bool  = False   # 是否需要升级


@dataclass
class Request:
    message:     str
    user_id:     str
    conv_id:     str
    context:     str = ""        # 来自 MemoryManager 的格式化上下文
    history:     Optional[List[Dict[str, str]]] = None  # 对话历史，传给意图识别
    intent:      Optional[IntentCategory] = None
    urgency:     Optional[UrgencyLevel]   = None
    request_id:  str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    # 混合式工具调用的门控信息（由 /chat 层按意图+快通道结果计算）：
    tools_allow: Optional[List[str]] = None   # 慢通道工具白名单，None/空 = 不启用 function calling
    tools_used:  List[str] = field(default_factory=list)  # 快通道已执行的工具，避免重复调用
    tool_context: Optional[Dict[str, Any]] = None  # 请求级上下文（user_id/档案），透传给工具 handler


@dataclass
class OrchestratorResult:
    request_id:  str
    response:    str
    agent_type:  AgentType
    intent:      Optional[IntentCategory]
    escalated:   bool  = False
    latency_ms:  float = 0.0


# ── 基础 Agent ────────────────────────────────────────────────────────────────

class BaseAgent:
    """所有 Agent 的基类，封装 LLM 调用和统计。"""

    agent_type: AgentType
    system_prompt: str

    MAX_TOOL_ROUNDS = 3   # function calling 循环上限，防止 LLM 反复调工具

    def __init__(self, client: AsyncAnthropic, model: str, skill_manager: Optional[Any] = None,
                 tool_manager: Optional[Any] = None):
        self._client = client
        self._model  = model
        self._skill_manager = skill_manager
        self._tool_manager  = tool_manager   # MCPToolManager，供工具循环执行 tool_use
        self._after_tool_call = None         # 工具执行后钩子（如 BMI 落盘），由 /chat 层注入
        self.stats   = AgentStats()

    async def handle(self, req: Request) -> AgentResponse:
        t0 = time.monotonic()
        self.stats.total += 1
        try:
            content = await self._call_llm(req)
            ms = (time.monotonic() - t0) * 1000
            self.stats.success += 1
            self.stats.total_ms += ms
            escalate = self._needs_escalation(content)
            return AgentResponse(
                agent_type=self.agent_type,
                content=content,
                success=True,
                latency_ms=ms,
                escalate=escalate,
            )
        except Exception as ex:
            ms = (time.monotonic() - t0) * 1000
            self.stats.total_ms += ms
            logger.error(f"{self.agent_type.value} 处理失败: {ex}")
            return AgentResponse(
                agent_type=self.agent_type,
                content="抱歉，处理您的请求时出现问题，请稍后重试。",
                success=False,
                latency_ms=ms,
            )

    async def _call_llm(self, req: Request) -> str:
        def _clean(s: str) -> str:
            return s.encode("utf-8", errors="ignore").decode("utf-8")

        messages = []
        if req.context:
            messages.append({"role": "user", "content": f"[背景信息]\n{_clean(req.context)}"})
            messages.append({"role": "assistant", "content": "好的，我已了解背景信息。"})
        messages.append({"role": "user", "content": _clean(req.message)})

        system_prompt = self._build_system_prompt(req)

        # ★ 显示发给 LLM 的完整内容
        print(f"[FLOW]   \u2502   \u251c\u2500 [System Prompt] ({self.agent_type.value}Agent):")
        for line in system_prompt.split('\n')[:5]:
            print(f"[FLOW]   \u2502   \u2502   {line}")
        if len(system_prompt.split('\n')) > 5:
            print(f"[FLOW]   \u2502   \u2502   ... (共 {len(system_prompt.split(chr(10)))} 行)")
        print(f"[FLOW]   │   ├─ [Messages] (共 {len(messages)} 条):")
        for i, msg in enumerate(messages):
            if isinstance(msg['content'], str):
                content_preview = msg['content'][:80].replace('\n', ' ')
            else:
                content_preview = "(结构化内容)"
            print(f"[FLOW]   │   │   [{i+1}] {msg['role']}: {content_preview}...")
        
        # 慢通道：白名单非空且工具管理器可用时，启用 function calling 循环
        tools = None
        if req.tools_allow and self._tool_manager is not None:
            tools = self._tool_manager.to_llm_tools(names=req.tools_allow, exclude=set(req.tools_used))
            if not tools:
                tools = None
        if tools:
            print(f"[FLOW]   │   ├─ [Function Calling] 开放工具: {[t['name'] for t in tools]}")
        
        kwargs: Dict[str, Any] = dict(
            model=self._model,
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
        )
        if tools:
            kwargs["tools"] = tools
        
        resp = await self._client.messages.create(**kwargs)
        
        # 工具循环：stop_reason=tool_use → 执行工具 → 结果回传 → 再调 LLM
        rounds = 0
        while resp.stop_reason == "tool_use" and tools and rounds < self.MAX_TOOL_ROUNDS:
            rounds += 1
            tool_blocks = [b for b in resp.content if b.type == "tool_use"]
            messages.append({"role": "assistant", "content": resp.content})
            tool_results = []
            for block in tool_blocks:
                print(f"[FLOW]   │   ├─ [工具调用·第{rounds}轮] {block.name}({json.dumps(block.input, ensure_ascii=False)})")
                result = await self._tool_manager.call(block.name, block.input, req.tool_context)
                payload = result.data if result.success else {"error": result.error or "工具调用失败"}
                if result.success and self._after_tool_call is not None:
                    try:
                        self._after_tool_call(block.name, block.input, result.data)
                    except Exception as hook_ex:
                        logger.warning(f"工具后置钩子异常: {hook_ex}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": _clean(json.dumps(payload, ensure_ascii=False, default=str)),
                    "is_error": not result.success,
                })
            messages.append({"role": "user", "content": tool_results})
            # kwargs["messages"] 与 messages 是同一 list 引用，原地追加后直接重发
            resp = await self._client.messages.create(**kwargs)
        
        result_text = "".join(b.text for b in resp.content if b.type == "text")
        if not result_text:
            result_text = resp.content[0].text if resp.content else "抱歉，暂时无法回答。"
        print(f"[FLOW]   \u2502   \u2514\u2500 [Agent LLM 返回] ({len(result_text)} 字):")
        for line in result_text.split('\n')[:4]:
            print(f"[FLOW]   \u2502       {line}")
        if len(result_text.split('\n')) > 4:
            print(f"[FLOW]   \u2502       ... (共 {len(result_text.split(chr(10)))} 行)")
        return result_text

    def _build_system_prompt(self, req: Request) -> str:
        """把动态加载的 Skills 拼入 system prompt，让业务规则随请求生效。"""
        if self._skill_manager is None:
            return self.system_prompt
        skill_prompt = self._skill_manager.prompt_for(req.message, self.agent_type.value)
        if not skill_prompt:
            return self.system_prompt
        return f"{self.system_prompt}\n\n[动态 Skills]\n{skill_prompt}"

    def _needs_escalation(self, content: str) -> bool:
        """检测 Agent 是否建议就医（简单关键词检测）。"""
        keywords = ["建议就医", "尽快就医", "及时就医", "立即就医", "急诊", "拨打 120", "拨打120", "医院就诊"]
        return any(kw in content for kw in keywords)


class HealthAgent(BaseAgent):
    agent_type    = AgentType.HEALTH
    system_prompt = (
        "你是 HealthMind 个性化健康顾问。友好、专业地回答健康养生问题，"
        "结合用户健康档案给出个性化建议。"
        "安全边界：不做疾病诊断、不推荐处方药、不替代医生；"
        "涉及疾病判断或持续不适时，明确建议就医。"
        "传统养生内容（如食疗、穴位）属于经验参考，需提示仅供参考。"
    )


class NutritionAgent(BaseAgent):
    agent_type    = AgentType.NUTRITION
    system_prompt = (
        "你是饮食营养顾问。专注于：养胃食疗、营养搭配、减脂饮食、体质调理饮食。"
        "给出具体可执行的饮食建议（食物、分量、频次）。"
        "涉及糖尿病、肾病等慢性病的饮食调整时，说明需要医生或营养师个体化评估。"
        "传统养生食疗属于经验参考，需提示仅供参考。"
    )


class FitnessAgent(BaseAgent):
    agent_type    = AgentType.FITNESS
    system_prompt = (
        "你是运动与睡眠顾问。专注于：运动计划、运动心率、作息调理、失眠改善。"
        "提供清晰的步骤化方案（频率、强度、时长）。"
        "涉及心血管不适、关节伤痛等情况时，说明需要先咨询医生再运动。"
    )


class MedicalAlertAgent(BaseAgent):
    agent_type    = AgentType.ESCALATION
    system_prompt = (
        "你是健康紧急响应助手。用户描述了可能的急症或危险信号。"
        "回复必须以就医提示开头：建议立即就医或拨打急救电话（120），"
        "简要说明为什么这些症状需要专业医疗评估，"
        "在就医前给出安全的临时注意事项（如保持静止、有人陪同）。"
        "严禁给出诊断结论、用药建议或任何可能延误就医的调理方案。"
    )


# ── 编排器 ────────────────────────────────────────────────────────────────────

class AgentOrchestrator:
    """
    多 Agent 编排器。

    路由逻辑（三层）：
      1. 意图 → Agent 类型映射
      2. 同类多实例时按 routing_score() 选最优
      3. 专属 Agent 失败时降级到 HealthAgent
    """

    # 意图 → Agent 类型的静态映射（路由表）
    _INTENT_ROUTING: Dict[IntentCategory, AgentType] = {
        IntentCategory.NUTRITION:  AgentType.NUTRITION,
        IntentCategory.FITNESS:    AgentType.FITNESS,
        IntentCategory.ESCALATION: AgentType.ESCALATION,
        # CONSULT/CALCULATE/PROFILE/GREETING/OTHER → HEALTH（默认）
    }

    def __init__(
        self,
        api_key:  str,
        base_url: Optional[str] = None,
        model:    str = "claude-3-5-sonnet-20241022",
        skill_manager: Optional[Any] = None,
        agent_tools_map: Optional[Dict[str, List[str]]] = None,
    ):
        client = build_llm_client(api_key, base_url)

        self._intent_recognizer = IntentRecognizer(api_key=api_key, base_url=base_url, model=model)
        self._skill_manager = skill_manager
        # Agent 类型 → 工具白名单：慢通道的二次过滤（治理层）
        self._agent_tools_map = agent_tools_map or {}

        # Agent 池：每种类型可有多个实例（水平扩展）
        self._pool: Dict[AgentType, List[BaseAgent]] = {
            AgentType.HEALTH:     [HealthAgent(client, model, skill_manager)],
            AgentType.NUTRITION:  [NutritionAgent(client, model, skill_manager)],
            AgentType.FITNESS:    [FitnessAgent(client, model, skill_manager)],
            AgentType.ESCALATION: [MedicalAlertAgent(client, model, skill_manager)],
        }

    def set_skill_manager(self, skill_manager: Optional[Any]) -> None:
        """更新 SkillManager 引用，供运行时重载或测试替换使用。"""
        self._skill_manager = skill_manager
        for agents in self._pool.values():
            for agent in agents:
                agent._skill_manager = skill_manager

    def set_tool_manager(self, tool_manager: Optional[Any]) -> None:
        """注入 MCPToolManager，启用 Agent 层 function calling 慢通道。"""
        for agents in self._pool.values():
            for agent in agents:
                agent._tool_manager = tool_manager

    # ── 主入口 ────────────────────────────────────────────────────────────────

    async def run(self, req: Request) -> OrchestratorResult:
        """
        处理一次请求的完整流程：
          意图识别 → 路由选 Agent → 执行 → 检查升级 → 返回结果
        """
        # ★ 编排器主入口
        print(f"[FLOW] Step 2/4: 编排器收到请求: \"{req.message}\"")
        t0 = time.monotonic()

        # 1. 意图识别（如果调用方已识别则跳过）
        if req.intent is None:
            intent_result = await self._intent_recognizer.recognize(req.message, history=req.history)
            req.intent  = intent_result.intent
            req.urgency = intent_result.urgency
            print(f"[FLOW]   \u251c\u2500 意图识别完成: intent={req.intent.value}, urgency={req.urgency.name}")

        # 复杂问题自动并行协作，例如同一句同时涉及减脂目标和饮食/运动安排。
        collaboration = self._collaboration_targets(req)
        if len(collaboration) > 1:
            return await self.run_parallel(req, collaboration)

        # 2. 路由：选择 Agent 类型
        agent_type = self._route(req.intent, req.urgency)
        print(f"[FLOW]   \u251c\u2500 路由决策: {req.intent.value} → {agent_type.value}Agent")

        # 3. 执行（含降级）
        print(f"[FLOW]   \u251c\u2500 调用 {agent_type.value}Agent 处理...")
        response = await self._execute(req, agent_type)
        print(f"[FLOW]   \u2514\u2500 Agent 返回: success={response.success}, 耗时 {response.latency_ms:.0f}ms")

        # 4. 升级检查（健康域：红旗症状/急症 → 就医预警）
        escalated = False
        if response.escalate or req.urgency == UrgencyLevel.CRITICAL or req.intent == IntentCategory.ESCALATION:
            escalated = True
            logger.warning(f"请求 {req.request_id} 触发就医预警: urgency={req.urgency}")
            # 生产环境：此处可推送就医提醒、创建随访工单

        return OrchestratorResult(
            request_id=req.request_id,
            response=response.content,
            agent_type=response.agent_type,
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    async def run_parallel(self, req: Request, agent_types: List[AgentType]) -> OrchestratorResult:
        """
        并行派发给多个 Agent，合并结果。
        适用于复杂问题（如同时涉及技术和账单）。
        """
        t0 = time.monotonic()
        tasks = [self._execute(req, at) for at in agent_types]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # 合并：拼接所有成功响应
        parts = []
        for r in responses:
            if isinstance(r, AgentResponse) and r.success:
                parts.append(f"[{r.agent_type.value}]\n{r.content}")

        combined = "\n\n".join(parts) if parts else "抱歉，所有 Agent 均处理失败。"
        escalated = any(isinstance(r, AgentResponse) and r.escalate for r in responses)

        return OrchestratorResult(
            request_id=req.request_id,
            response=combined,
            agent_type=agent_types[0],
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    # ── 路由逻辑 ──────────────────────────────────────────────────────────────

    def _route(self, intent: Optional[IntentCategory], urgency: Optional[UrgencyLevel]) -> AgentType:
        """
        三层路由决策：
          1. 意图映射
          2. 紧急度覆盖（CRITICAL 直接升级）
          3. 默认 GENERAL
        """
        if urgency == UrgencyLevel.CRITICAL:
            return AgentType.ESCALATION

        if intent and intent in self._INTENT_ROUTING:
            target = self._INTENT_ROUTING[intent]
            # 如果目标类型有可用实例则使用，否则降级
            if target in self._pool and self._pool[target]:
                return target

        return AgentType.HEALTH

    def _collaboration_targets(self, req: Request) -> List[AgentType]:
        """
        判断是否需要多个 Agent 并行协作。

        意图识别通常只返回一个主意图；这里用领域关键词补充检测复合问题，
        例如"减脂"同时涉及饮食和运动，需要两个 Agent 并行处理。
        """
        msg = req.message.lower()
        targets: List[AgentType] = []

        nutrition_kws = ["饮食", "吃什么", "食谱", "养胃", "食疗", "营养", "忌口"]
        fitness_kws = ["运动", "锻炼", "跑步", "健身", "睡眠", "失眠", "作息"]
        combined_kws = ["减脂", "减肥", "瘦身", "增肌"]  # 天然需要饮食 + 运动协作

        if req.intent == IntentCategory.NUTRITION or any(kw in msg for kw in nutrition_kws):
            targets.append(AgentType.NUTRITION)
        if req.intent == IntentCategory.FITNESS or any(kw in msg for kw in fitness_kws):
            targets.append(AgentType.FITNESS)
        # 复合目标（减脂等）同时命中两侧关键词才触发协作，避免普通单域问题被拆分
        if any(kw in msg for kw in combined_kws) and req.intent in (IntentCategory.NUTRITION, IntentCategory.FITNESS, IntentCategory.CONSULT):
            if AgentType.NUTRITION not in targets:
                targets.append(AgentType.NUTRITION)
            if AgentType.FITNESS not in targets:
                targets.append(AgentType.FITNESS)

        # 保持顺序去重，并只返回当前有实例的 Agent 类型。
        deduped = list(dict.fromkeys(targets))
        return [agent_type for agent_type in deduped if self._pool.get(agent_type)]

    def _best_agent(self, agent_type: AgentType) -> Optional[BaseAgent]:
        """
        性能路由：从同类 Agent 中选 routing_score() 最高的。
        这是"基于在线表现动态调整路由"的核心。
        """
        agents = self._pool.get(agent_type, [])
        if not agents:
            return None
        return max(agents, key=lambda a: a.stats.routing_score())

    async def _execute(self, req: Request, agent_type: AgentType) -> AgentResponse:
        """执行 Agent，失败时降级到 HealthAgent。"""
        agent = self._best_agent(agent_type)
        if agent is None:
            agent = self._best_agent(AgentType.HEALTH)
        if agent is None:
            return AgentResponse(
                agent_type=AgentType.HEALTH,
                content="服务暂时不可用，请稍后重试。",
                success=False,
            )

        # 慢通道二次过滤：按实际执行的 Agent 类型收紧工具白名单
        # 用 replace 复制而非原地修改，避免并行协作时多 Agent 共享 req 相互污染
        whitelist = self._agent_tools_map.get(agent_type.value)
        if req.tools_allow is not None and whitelist is not None:
            req = replace(req, tools_allow=[t for t in req.tools_allow if t in whitelist])

        response = await agent.handle(req)

        # 专属 Agent 失败时降级到 HealthAgent；就医预警不降级，保证安全链路不被吞掉
        if not response.success and agent_type not in (AgentType.HEALTH, AgentType.ESCALATION):
            logger.warning(f"{agent_type.value} 失败，降级到 HealthAgent")
            fallback = self._best_agent(AgentType.HEALTH)
            if fallback:
                response = await fallback.handle(req)

        return response

    # ── 统计（供 Monitor 读取）────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        result = {}
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                result[key] = {
                    "total":        agent.stats.total,
                    "success_rate": round(agent.stats.success_rate, 3),
                    "avg_ms":       round(agent.stats.avg_ms, 1),
                    "monitor_penalty": round(agent.stats.monitor_penalty, 3),
                    "routing_score": round(agent.stats.routing_score(), 3),
                }
        return result

    def update_routing_penalties(self, penalties: Dict[str, float]) -> None:
        """
        接收 Monitor 的在线表现反馈，动态调整路由惩罚项。

        penalties 的 key 使用 get_stats() 中的 agent key，例如 technical_0。
        """
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                penalty = penalties.get(key, 0.0)
                agent.stats.monitor_penalty = min(max(penalty, 0.0), 0.9)
