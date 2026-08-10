"""
HealthMind 个性化健康管理助手 — FastAPI 入口

启动时打印品牌 Banner。
所有核心组件在 lifespan 中初始化，通过环境变量配置。
"""
import asyncio
import logging
import os
import pathlib
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

# 将项目根目录加入 sys.path，确保无论从哪里执行都能找到 agents/core/memory 等模块
# 这一行必须在所有项目内部 import 之前执行
_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from mcp.document_parser import DocumentParseError, SUPPORTED_BINARY_SUFFIXES, parse_document

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BANNER = r"""

   ╔══════════════════════════╗
   ║   HealthMind  v3.0       ║
   ║   个性化健康管理助手      ║
   ╚══════════════════════════╝

"""

# ── 全局组件（lifespan 中初始化）─────────────────────────────────────────────
_orchestrator = None
_memory       = None
_tool_manager = None
_monitor      = None
_evaluator    = None
_skill_manager = None
_bmi_store    = None

# ── 混合式工具调用的慢通道配置 ───────────────────────────────────────────────────
# function calling 总开关：关闭后回退纯关键词快通道（降级开关）
ENABLE_FUNCTION_CALLING = os.getenv("ENABLE_FUNCTION_CALLING", "1") not in ("0", "false", "False")
# 按 Agent 类型的工具白名单：编排层治理“哪些工具对哪个 Agent 开放”
AGENT_TOOL_WHITELIST: Dict[str, List[str]] = {
    "health":     ["health_calculators", "bmi_trend_history", "external_health_api"],
    "nutrition":  ["health_calculators", "bmi_trend_history"],
    "fitness":    ["health_calculators", "bmi_trend_history", "external_health_api"],
    "escalation": [],   # 就医预警场景不给工具，避免延误就医
}
# 不开放给 LLM 自主调用的工具：knowledge_search 已由 RAG 链路注入；detect_red_flags 在 /chat 前置执行
_INTERNAL_TOOLS = {"knowledge_search", "detect_red_flags"}
# 当前请求的 user_id 持有器：供慢通道工具后置钩子（BMI 落盘）使用
req_user_id_holder: Dict[str, str] = {"uid": ""}


def _anthropic_cfg() -> Dict[str, Any]:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("未设置 ANTHROPIC_API_KEY")
    cfg: Dict[str, Any] = {
        "api_key":  key,
        "model":    os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
    }
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip()
    if base_url:
        cfg["base_url"] = base_url
    return cfg


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator, _memory, _tool_manager, _monitor, _evaluator, _skill_manager, _bmi_store

    print(BANNER, flush=True)

    from agents.agent_orchestrator import AgentOrchestrator, Request
    from core.intent_recognizer import IntentRecognizer
    from core.safety_checker import detect_red_flags
    from evaluation.evaluator import EndToEndEvaluator
    from mcp.knowledge_base import KnowledgeBase
    from mcp.tool_manager import MCPToolManager, Tool
    from memory.conversation_memory import MemoryManager
    from monitor.performance_monitor import PerformanceMonitor
    from core.skill_loader import SkillManager
    from tools.health_calculators import health_calculators
    from tools.health_history import BmiHistoryStore
    from tools import external_health

    cfg = _anthropic_cfg()
    logger.info(f"模型: {cfg['model']}  base_url: {cfg.get('base_url', '(官方)')}")

    # 意图识别器（Orchestrator 内部也会创建，这里单独暴露给 Evaluator）
    recognizer = IntentRecognizer(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    # Skills：启动时从目录加载健康咨询规范，并在 Agent 调用 LLM 时动态注入。
    skills_dir = os.getenv("HEALTHMIND_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills"))
    _skill_manager = SkillManager(
        root_dir=skills_dir,
        max_prompt_chars=int(os.getenv("HEALTHMIND_SKILLS_MAX_PROMPT_CHARS", "5000")),
    )
    _skill_manager.load()

    # Agent 编排器（agent_tools_map：慢通道按 Agent 类型收紧工具白名单）
    _orchestrator = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=_skill_manager,
        agent_tools_map=AGENT_TOOL_WHITELIST,
    )

    # 记忆管理器（Redis 工作记忆 + ChromaDB 情景记忆/用户画像）
    _memory = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    # MCP 工具管理器 + RAG 知识库（基于 ChromaDB 的真实检索）
    _tool_manager = MCPToolManager(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )
    kb = KnowledgeBase(
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
    )
    logger.info(f"知识库已加载: {kb.doc_count} 个文档片段")

    def knowledge_fallback(params: Dict[str, Any], context: Optional[Dict[str, Any]], error: str):
        query = params.get("query", "")
        return [{
            "title": "知识库降级结果",
            "content": f"知识库暂时不可用，未能完成对“{query}”的语义检索。请稍后重试。",
            "score": 0.0,
            "fallback": True,
            "error": error,
        }]

    _tool_manager.register(Tool(
        name="knowledge_search",
        description="搜索知识库（基于 ChromaDB 向量检索）",
        handler=kb.search_handler,
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
        cache_ttl=300.0,
        supports_rerank=True,
        fallback=knowledge_fallback,
    ))

    # 健康计算工具：BMI、饮水量、睡眠作息、运动心率（确定性计算，不依赖 LLM）
    async def health_calc_handler(params: Dict[str, Any], context: Optional[Dict[str, Any]]):
        # 慢通道：用户档案从请求上下文取（LLM 无需生成）；快通道仍可从 params 传入
        profile = (context or {}).get("user_profile") or params.get("user_profile")
        return health_calculators.run_tools(
            str(params.get("text") or ""),
            profile,
        )

    _tool_manager.register(Tool(
        name="health_calculators",
        description=(
            "确定性健康计算：BMI、饮水量估算、睡眠作息规划、运动心率估算。"
            "text 传入用户原话（内含身高体重等数值），系统自动识别并计算。"
        ),
        handler=health_calc_handler,
        schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "user_profile": {"type": "object"},
            },
            "required": ["text"],
        },
        cache_ttl=60.0,
    ))

    # 红旗症状检测：识别危险信号，决定是否优先就医提示
    async def red_flag_handler(params: Dict[str, Any], context: Optional[Dict[str, Any]]):
        from dataclasses import asdict
        return asdict(detect_red_flags(str(params.get("text") or "")))

    _tool_manager.register(Tool(
        name="detect_red_flags",
        description="红旗症状检测：识别危险信号，决定是否优先就医提示",
        handler=red_flag_handler,
        schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        cache_ttl=0.0,
    ))

    # BMI 历史趋势工具：Redis 存取按用户隔离的 BMI 记录（记忆系统 ↔ 工具打通）
    _bmi_store = BmiHistoryStore(redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"))

    async def bmi_trend_handler(params: Dict[str, Any], context: Optional[Dict[str, Any]]):
        # 优先用请求上下文的 user_id，避免依赖 LLM 生成的参数
        uid = (context or {}).get("user_id") or str(params.get("user_id") or "u1001")
        summary = _bmi_store.summarize(uid)
        return summary or {
            "record_count": 0,
            "summary_text": "暂无 BMI 历史记录，可先说「身高170 体重70 算一下BMI」建立首条记录。",
        }

    _tool_manager.register(Tool(
        name="bmi_trend_history",
        description="BMI 历史趋势：读取当前用户的 BMI 历史记录并生成趋势摘要（用户问体重/BMI变化、趋势时调用）",
        handler=bmi_trend_handler,
        schema={
            "type": "object",
            "properties": {},
        },
        cache_ttl=0.0,
    ))

    # 外部环境健康工具：真实外部 API（Open-Meteo），展示熔断/超时/降级保护
    def _resolve_env_city(params: Dict[str, Any], context: Optional[Dict[str, Any]]) -> str:
        """城市兜底顺序：LLM/用户显式指定 → 用户档案常住城市 → 默认北京。"""
        city = str(params.get("city") or "").strip()
        if city:
            return city
        profile = (context or {}).get("user_profile") or {}
        city = str(profile.get("default_city") or "").strip()
        return city or external_health.DEFAULT_CITY

    async def env_health_handler(params: Dict[str, Any], context: Optional[Dict[str, Any]]):
        city = _resolve_env_city(params, context)
        report = await external_health.fetch_env_report(city)
        report["summary_text"] = external_health.format_report(report)
        return report

    def env_health_fallback(params: Dict[str, Any], context: Optional[Dict[str, Any]], error: str):
        city = _resolve_env_city(params, context)
        return {
            "city": city, "fallback": True, "error": error,
            "summary_text": f"{city}的实时空气与天气数据暂时无法获取（外部 API 不可用），户外活动请依据自身实际环境判断。",
        }

    _tool_manager.register(Tool(
        name="external_health_api",
        description=(
            "外部环境健康：实时空气质量与天气（Open-Meteo），给出户外活动建议。"
            "用户询问能否户外运动/跑步、空气质量、天气时，直接调用本工具获取实时数据，"
            "不要反问用户；city 可选（北京/上海/广州/深圳/杭州/成都/武汉/西安/南京/重庆/长沙/天津），"
            "用户未说明城市时省略该参数，系统自动按其档案中的常住城市查询（无记录时默认北京）。"
        ),
        handler=env_health_handler,
        schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
        },
        cache_ttl=300.0,
        timeout_s=12.0,
        fallback=env_health_fallback,
    ))

    # 慢通道接线：把工具管理器注入编排器，Agent 层启用 function calling
    _orchestrator.set_tool_manager(_tool_manager)

    def _after_tool_call(tool_name: str, params: Dict[str, Any], data: Any):
        """慢通道工具执行后钩子：BMI 算出同样自动落盘历史（与快通道行为一致）。"""
        if tool_name != "health_calculators" or not isinstance(data, list):
            return
        for item in data:
            if item.get("name") != "bmi_calculator" or _bmi_store is None:
                continue
            d = item.get("data", {})
            if _bmi_store.record(req_user_id_holder["uid"], d.get("height_cm", 0), d.get("weight_kg", 0), d.get("bmi", 0)):
                print(f"[FLOW]   │   ├─ [BMI 历史·慢通道] 已为用户 {req_user_id_holder['uid']} 落盘一条记录")

    for agents in _orchestrator._pool.values():
        for agent in agents:
            agent._after_tool_call = _after_tool_call

    # 性能监控（可选启动 Prometheus）
    prom_port = int(os.getenv("PROMETHEUS_PORT", "0")) or None
    _monitor = PerformanceMonitor(
        orchestrator=_orchestrator,
        tool_manager=_tool_manager,
        interval_s=float(os.getenv("MONITOR_INTERVAL", "10")),
        webhook_url=os.getenv("ALERT_WEBHOOK_URL") or None,
        prometheus_port=prom_port,
    )
    await _monitor.start()

    # 评测器
    _evaluator = EndToEndEvaluator(
        orchestrator=_orchestrator,
        recognizer=recognizer,
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        baseline_path=os.getenv("EVAL_BASELINE_PATH", "/app/data/eval/baseline.json"),
    )

    logger.info("HealthMind 已就绪")
    yield

    await _monitor.stop()
    logger.info("HealthMind 已关闭")


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="HealthMind 个性化健康管理助手",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:     str
    user_id:     str = "anonymous"
    conv_id:     Optional[str] = None


class ChatResponse(BaseModel):
    conv_id:     str
    response:    str
    intent:      str
    agent_type:  str
    escalated:   bool
    latency_ms:  float
    knowledge_used: bool = False
    red_flag:    bool = False   # 是否命中红旗症状（就医预警）


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return {"status": "ok", "agents": _orchestrator.get_stats()}


@app.get("/skills", tags=["Skills"])
async def skills_summary():
    """查看当前已加载的 Skills，便于确认热加载结果和排查解析错误。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    return _skill_manager.summary()


@app.post("/skills/reload", tags=["Skills"])
async def reload_skills():
    """运行时重新扫描 Skill 目录，不需要重启服务。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    _skill_manager.reload()
    if _orchestrator is not None:
        _orchestrator.set_skill_manager(_skill_manager)
    return _skill_manager.summary()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    主对话接口。完整流程：
      安全前置检查 → 记忆读取 → 健康计算/知识库注入 → Agent 路由执行 → 记忆写入
    """
    print(f"\n{'='*60}")
    print(f"[FLOW] 收到请求: user={req.user_id}, message=\"{req.message}\"")
    print(f"{'='*60}")
    if _orchestrator is None or _memory is None:
        raise HTTPException(503, "服务未就绪")

    from agents.agent_orchestrator import Request as OrcReq
    from core.intent_recognizer import IntentCategory, UrgencyLevel
    from core.safety_checker import detect_red_flags
    from memory.conversation_memory import MsgRole

    conv_id = req.conv_id or str(uuid.uuid4())

    # 0. 安全前置检查：红旗症状命中则强制进入就医预警链路
    safety = detect_red_flags(req.message)
    if safety.is_high_risk:
        print(f"[FLOW] Step 0/4: 红旗症状检测命中: {safety.matched_flags}")

    # 1. 读取记忆上下文
    print("[FLOW] Step 1/4: 读取三级记忆上下文...")
    mem_ctx = await _memory.get_context(req.user_id, conv_id, query=req.message)

    # 2. 构建编排请求（含对话历史，用于意图识别上下文）
    history = [
        {"role": m.role.value, "content": m.content}
        for m in mem_ctx.recent_messages[-5:]
    ] if mem_ctx.recent_messages else None

    knowledge_text, knowledge_used = await _build_knowledge_context(req.message)
    health_tool_text, health_tools_used = await _build_health_tool_context(req.message, mem_ctx.user_profile, req.user_id)
    env_tool_text, env_tools_used = await _build_env_context(req.message, mem_ctx.user_profile)
    fast_tools_used = health_tools_used + env_tools_used
    context_parts = [mem_ctx.to_prompt_text()]
    if safety.is_high_risk:
        context_parts.append(
            f"[安全提示] 检测到红旗症状: {', '.join(safety.matched_flags)}。"
            "请优先建议用户立即就医，不要给出可能延误就医的调理方案。"
        )
    if knowledge_text:
        context_parts.append(knowledge_text)
    if health_tool_text:
        context_parts.append(health_tool_text)
    if env_tool_text:
        context_parts.append(env_tool_text)
    full_context = "\n\n".join(part for part in context_parts if part)

    orch_req = OrcReq(
        message=req.message,
        user_id=req.user_id,
        conv_id=conv_id,
        context=full_context,
        history=history,
        tools_allow=_decide_tools_allow(safety.is_high_risk, fast_tools_used),
        tools_used=fast_tools_used,
        tool_context={"user_id": req.user_id, "user_profile": mem_ctx.user_profile or {}},
    )
    # 慢通道后置钩子（BMI 落盘）需要当前请求的 user_id
    req_user_id_holder["uid"] = req.user_id
    # 红旗症状：跳过常规意图识别，直接强制路由到就医预警 Agent
    if safety.is_high_risk:
        orch_req.intent  = IntentCategory.ESCALATION
        orch_req.urgency = UrgencyLevel.CRITICAL

    # 3. 执行（进入编排器：意图识别 → Agent 路由 → LLM 调用）
    print(f"[FLOW] Step 3/4: 进入编排器...")
    print(f"[FLOW]   \u251c\u2500 [完整 Context] (发给 Agent 的背景信息):")
    if full_context:
        for line in full_context.split('\n')[:6]:
            print(f"[FLOW]   \u2502   {line}")
        if len(full_context.split('\n')) > 6:
            print(f"[FLOW]   \u2502   ... (共 {len(full_context.split(chr(10)))} 行)")
    else:
        print(f"[FLOW]   \u2502   (空)")
    result = await _orchestrator.run(orch_req)

    # 4. 写入记忆
    print(f"[FLOW] Step 4/4: 写入记忆 (intent={result.intent.value}, agent={result.agent_type.value})")
    print(f"[FLOW]   \u251c\u2500 [Agent 最终回复]:")
    for line in result.response.split('\n')[:4]:
        print(f"[FLOW]   \u2502   {line}")
    if len(result.response.split('\n')) > 4:
        print(f"[FLOW]   \u2502   ... (共 {len(result.response.split(chr(10)))} 行)")
    print(f"[FLOW]   \u2514\u2500 总耗时: {result.latency_ms:.0f}ms, 升级: {result.escalated}")
    await _memory.add_message(req.user_id, conv_id, MsgRole.USER, req.message)
    await _memory.add_message(req.user_id, conv_id, MsgRole.ASSISTANT, result.response)

    # 5. 异步更新用户画像（不阻塞响应）
    asyncio.create_task(_memory.update_profile(req.user_id, conv_id))

    return ChatResponse(
        conv_id=conv_id,
        response=result.response,
        intent=result.intent.value if result.intent else "other",
        agent_type=result.agent_type.value,
        escalated=result.escalated,
        latency_ms=round(result.latency_ms, 1),
        knowledge_used=knowledge_used,
        red_flag=safety.is_high_risk,
    )


async def _build_knowledge_context(message: str, top_k: int = 3) -> tuple[str, bool]:
    """
    为 /chat 主链路构建 RAG 知识上下文。

    这里复用 MCPToolManager 的查询改写、并行召回、重排、fallback 能力。
    """
    if _tool_manager is None:
        return "", False
    if not _should_use_knowledge(message):
        return "", False
    try:
        result = await _tool_manager.search_with_rewrite("knowledge_search", message, top_k=top_k)
        if not result.success or not isinstance(result.data, list) or not result.data:
            return "", False

        parts = ["[知识库检索结果]"]
        used = False
        for i, item in enumerate(result.data[:top_k], start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "未命名文档"))
            content = str(item.get("content", "")).strip()
            score = item.get("score", "")
            if not content:
                continue
            used = True
            parts.append(f"{i}. 标题: {title}\n   相关度: {score}\n   内容: {content[:600]}")

        if not used:
            return "", False
        parts.append("请优先依据以上知识库内容回答；如果知识库内容不足，再结合通用健康知识说明。")
        return "\n".join(parts), True
    except Exception as ex:
        logger.warning(f"构建知识库上下文失败: {ex}")
        return "", False


async def _build_health_tool_context(message: str, user_profile: Dict[str, Any], user_id: str = "") -> tuple:
    """
    快通道：健康计算工具由 /chat 按关键词触发，结果注入 Agent 上下文。

    复用 MCPToolManager 的熔断、缓存、参数校验能力，
    保证回复中的 BMI/饮水量/心率等数值来自确定性工具而非模型编造。
    同时负责：BMI 算出后自动落盘历史记录；用户问趋势时读取摘要。
    返回 (注入文本, 已执行工具名列表)——后者用于慢通道去重。
    """
    if _tool_manager is None:
        return "", []
    from tools.health_calculators import health_calculators

    parts: List[str] = []
    used: List[str] = []

    if health_calculators.should_trigger(message):
        try:
            result = await _tool_manager.call(
                "health_calculators",
                {"text": message, "user_profile": user_profile or {}},
            )
            if result.success and isinstance(result.data, list) and result.data:
                print(f"[FLOW]   ├─ [健康计算工具·快通道] 命中 {len(result.data)} 个工具: {[r['name'] for r in result.data]}")
                parts.append(health_calculators.format_for_prompt(result.data))
                used.append("health_calculators")

                # BMI 命中 → 自动写入 Redis 历史（记忆系统 ↔ 工具打通）
                for item in result.data:
                    if item.get("name") == "bmi_calculator" and user_id and _bmi_store is not None:
                        d = item.get("data", {})
                        if _bmi_store.record(user_id, d.get("height_cm", 0), d.get("weight_kg", 0), d.get("bmi", 0)):
                            print(f"[FLOW]   ├─ [BMI 历史] 已为用户 {user_id} 落盘一条记录")
        except Exception as ex:
            logger.warning(f"健康计算工具调用失败: {ex}")

    # 趋势问询触发：「我体重/BMI 最近变化怎么样」→ 读 Redis 历史生成摘要
    lowered = (message or "").lower()
    if any(kw in lowered for kw in ("趋势", "历史", "变化", "记录")) and ("bmi" in lowered or "体重" in lowered):
        try:
            trend = await _tool_manager.call("bmi_trend_history", {}, context={"user_id": user_id or "u1001"})
            if trend.success and isinstance(trend.data, dict):
                print("[FLOW]   ├─ [BMI 趋势工具·快通道] 注入历史摘要")
                parts.append(f"[BMI 历史趋势]\n{trend.data.get('summary_text', '')}")
                used.append("bmi_trend_history")
        except Exception as ex:
            logger.warning(f"BMI 趋势工具调用失败: {ex}")

    return "\n".join(p for p in parts if p), used


async def _build_env_context(message: str, user_profile: Dict[str, Any] = None) -> tuple:
    """
    快通道：外部环境工具按空气/天气/户外关键词触发，注入实时 AQI 与天气。

    走真实外部 API（Open-Meteo），由 MCPToolManager 的超时、熔断、
    fallback 提供保护；降级时返回提示文本而非报错。
    关键词未命中的长尾问法由慢通道（function calling）兜底。
    城市兜底顺序：消息显式提及 → 用户档案常住城市 → 默认北京。
    返回 (注入文本, 已执行工具名列表)。
    """
    if _tool_manager is None:
        return "", []
    from tools import external_health
    if not external_health.should_trigger(message):
        return "", []
    profile_city = str((user_profile or {}).get("default_city") or "").strip()
    city = external_health.extract_city(message, fallback=profile_city or external_health.DEFAULT_CITY)
    try:
        result = await _tool_manager.call("external_health_api", {"city": city})
        if not result.success or not isinstance(result.data, dict):
            return "", []
        print(f"[FLOW]   ├─ [外部环境工具·快通道] city={city}, fallback={result.data.get('fallback', False)}, error={result.data.get('error') or result.error}")
        text = (
            f"[外部环境数据]\n{result.data.get('summary_text', '')}\n"
            "回答户外运动相关问题时请结合以上实时数据给出建议。"
        )
        return text, ["external_health_api"]
    except Exception as ex:
        logger.warning(f"外部环境工具调用失败: {ex}")
        return "", []


def _decide_tools_allow(safety_high_risk: bool, tools_used: List[str]) -> Optional[List[str]]:
    """
    慢通道门控：按开关/安全等级/快通道结果计算给 Agent 的工具白名单。
    返回 None 表示本次请求不启用 function calling。
    注：白名单是候选集，最终还会按路由出的 Agent 类型二次过滤。
    """
    if not ENABLE_FUNCTION_CALLING or safety_high_risk:
        return None
    allow = [n for n in ("health_calculators", "bmi_trend_history", "external_health_api")
             if n not in _INTERNAL_TOOLS and n not in tools_used]
    return allow or None


def _should_use_knowledge(message: str) -> bool:
    """跳过纯寒暄，健康类问题才检索知识库，避免无关 RAG 干扰回复。"""
    msg = (message or "").strip().lower()
    if not msg:
        return False
    greetings = {"你好", "您好", "嗨", "hi", "hello", "hey", "早上好", "晚上好"}
    if msg in greetings:
        return False
    health_keywords = [
        "养胃", "养肝", "祛湿", "湿气", "失眠", "睡眠", "作息", "运动", "锻炼",
        "减肥", "减脂", "饮食", "营养", "食疗", "体质", "调理", "养生", "心率",
        "症状", "不适", "疲劳", "季节", "秋", "冬", "春", "夏",
        "sleep", "diet", "exercise", "health",
    ]
    return len(msg) >= 4 or any(kw in msg for kw in health_keywords)


@app.get("/monitor")
async def monitor_summary():
    """实时监控摘要：Agent 成功率、工具统计、告警、优化建议。"""
    if _monitor is None:
        raise HTTPException(503, "服务未就绪")
    return _monitor.summary()


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus 指标入口。"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/search")
async def search(query: str, top_k: int = 5):
    """
    演示检索优化链路：查询改写 → 并行召回 → 重排 → Top-K。
    展示 MCP 工具调用的核心亮点。
    """
    if _tool_manager is None:
        raise HTTPException(503, "服务未就绪")
    result = await _tool_manager.search_with_rewrite("knowledge_search", query, top_k=top_k)
    return {"query": query, "results": result.data, "reranked": result.reranked}


class DocInput(BaseModel):
    """单篇文档输入。"""
    title:   str
    content: str


class BatchDocInput(BaseModel):
    """批量文档导入请求体。"""
    documents: List[DocInput]


class EvalIntentInput(BaseModel):
    """意图识别评测用例。"""
    message: str
    expected_intent: str
    context: Optional[Dict[str, Any]] = None


class EvalDialogInput(BaseModel):
    """对话质量评测用例。question 单轮，turns 多轮。"""
    question: Optional[str] = None
    turns: Optional[List[str]] = None
    user_id: Optional[str] = None
    conv_id: Optional[str] = None


class EvalRunInput(BaseModel):
    """评测请求。为空时使用内置默认用例。"""
    intent_cases: Optional[List[EvalIntentInput]] = None
    dialog_cases: Optional[List[EvalDialogInput]] = None


@app.post("/knowledge/add", tags=["知识库"])
async def add_knowledge(body: BatchDocInput):
    """
    批量导入文档到知识库。

    文档会自动切片（每片 500 字）并存入 ChromaDB，ChromaDB 内置 Embedding 模型自动向量化。

    示例请求体：
    ```json
    {
      "documents": [
        {"title": "养胃指南", "content": "规律三餐、少食生冷辛辣，是养胃的基础..."},
        {"title": "睡眠建议", "content": "建议成人每晚保证 7-9 小时睡眠..."}
      ]
    }
    ```
    """
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    count = kb.add_documents([{"title": d.title, "content": d.content} for d in body.documents])
    return {"message": f"成功导入 {count} 个文档片段", "added_chunks": count, "total_chunks": kb.doc_count}


@app.post("/knowledge/upload", tags=["知识库"])
async def upload_knowledge(file: UploadFile = File(...)):
    """
    上传文件导入知识库。

    支持格式：
    - `.txt` / `.md`：整个文件作为一篇文档，文件名作为标题
    - `.json`：JSON 数组格式 `[{"title": "...", "content": "..."}, ...]`
    - `.pdf`：pdfplumber 逐页抽取文本（扫描件/加密件不支持）
    - `.docx`：抽取段落与表格文本
    - `.html` / `.htm`：剥离标签取正文，优先用 <title> 作为标题

    所有格式均自动按 500 字切块入库。文件大小限制：10MB
    """
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "文件大小超过 10MB 限制")

    filename = file.filename or "unknown"
    suffix = filename.lower()
    suffix = suffix[suffix.rfind("."):] if "." in suffix else ""

    if suffix in SUPPORTED_BINARY_SUFFIXES:
        # pdf / docx / html：走文档解析器提取纯文本
        try:
            docs = parse_document(filename, content)
        except DocumentParseError as ex:
            raise HTTPException(400, str(ex))
    elif filename.endswith(".json"):
        import json as _json
        text = content.decode("utf-8", errors="ignore")
        try:
            docs = _json.loads(text)
            if not isinstance(docs, list):
                raise HTTPException(400, "JSON 文件应为数组格式: [{title, content}, ...]")
        except _json.JSONDecodeError as e:
            raise HTTPException(400, f"JSON 解析失败: {e}")
    else:
        # txt / md：整个文件作为一篇文档
        text = content.decode("utf-8", errors="ignore")
        title = filename.rsplit(".", 1)[0] if "." in filename else filename
        docs = [{"title": title, "content": text}]

    count = kb.add_documents(docs)
    return {
        "message": f"文件 {filename} 导入成功",
        "added_chunks": count,
        "total_chunks": kb.doc_count,
    }


@app.get("/knowledge/stats", tags=["知识库"])
async def knowledge_stats():
    """查看知识库统计信息（文档片段总数）。"""
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    return {"total_chunks": kb.doc_count}


@app.get("/knowledge/list", tags=["知识库"])
async def knowledge_list():
    """按文档分组列出知识库全部切片，用于直观查看离线建库结果。"""
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    data = kb._collection.get(include=["documents", "metadatas"])

    docs: Dict[str, Any] = {}
    for cid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        title = (meta or {}).get("title", "未命名")
        entry = docs.setdefault(title, {
            "title": title,
            "total_chunks": (meta or {}).get("total_chunks", 1),
            "chunks": [],
        })
        entry["chunks"].append({
            "id": cid,
            "chunk_index": (meta or {}).get("chunk_index", 0),
            "chars": len(doc or ""),
            "content": doc,
        })

    for entry in docs.values():
        entry["chunks"].sort(key=lambda c: c["chunk_index"])
    result = sorted(docs.values(), key=lambda d: d["title"])
    return {"total_chunks": kb.doc_count, "total_documents": len(result), "documents": result}


@app.delete("/knowledge/doc/{title}", tags=["知识库"])
async def knowledge_delete_doc(title: str):
    """按标题删除一篇文档的全部切片（方便反复测试建库）。"""
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    data = kb._collection.get(where={"title": title}, include=[])
    if not data["ids"]:
        raise HTTPException(404, f"未找到标题为「{title}」的文档")
    kb._collection.delete(ids=data["ids"])
    return {"message": f"已删除「{title}」的 {len(data['ids'])} 个切片", "total_chunks": kb.doc_count}


@app.get("/memory/users", tags=["记忆"])
async def memory_users():
    """枚举所有有记忆痕迹的用户（工作记忆/情景记忆/档案），供记忆可视化页面。"""
    if _memory is None:
        raise HTTPException(503, "记忆系统未初始化")

    def _entry(uid: str) -> Dict[str, Any]:
        return users.setdefault(uid, {"user_id": uid, "has_profile": False,
                                      "episodic_count": 0, "conv_count": 0})

    users: Dict[str, Dict[str, Any]] = {}
    # 工作记忆（Redis）：键格式 wm:{user_id}:{conv_id}
    for key in _memory._redis.scan_iter(match="wm:*", count=200):
        parts = key.split(":", 2)
        if len(parts) == 3:
            _entry(parts[1])["conv_count"] += 1
    # 情景记忆（ChromaDB）
    for meta in _memory._episodic.get(include=["metadatas"])["metadatas"]:
        uid = (meta or {}).get("user_id")
        if uid:
            _entry(uid)["episodic_count"] += 1
    # 用户档案（ChromaDB）
    for meta in _memory._profile.get(include=["metadatas"])["metadatas"]:
        uid = (meta or {}).get("user_id")
        if uid:
            _entry(uid)["has_profile"] = True

    return {"total_users": len(users),
            "users": sorted(users.values(), key=lambda u: u["user_id"])}


@app.get("/memory/{user_id}", tags=["记忆"])
async def memory_detail(user_id: str, limit: int = 50):
    """查看单个用户的三级记忆：工作记忆（按会话分组）、情景记忆、用户健康档案。"""
    if _memory is None:
        raise HTTPException(503, "记忆系统未初始化")

    # 1. 工作记忆（Redis）：遍历该用户所有会话
    convs: List[Dict[str, Any]] = []
    for key in _memory._redis.scan_iter(match=f"wm:{user_id}:*", count=200):
        conv_id = key.split(":", 2)[2]
        msgs = await _memory._get_working_memory(user_id, conv_id)
        convs.append({
            "conv_id": conv_id,
            "summary": _memory._redis.get(_memory._summary_key(user_id, conv_id)) or "",
            "messages": [{"role": m.role.value, "content": m.content,
                          "ts": m.timestamp.isoformat()} for m in msgs],
        })
    convs.sort(key=lambda c: c["messages"][-1]["ts"] if c["messages"] else "", reverse=True)

    # 2. 情景记忆（ChromaDB）：压缩时沉淀的跨会话历史
    data = _memory._episodic.get(where={"user_id": user_id}, include=["documents", "metadatas"])
    episodic = []
    for doc, meta in zip(data["documents"], data["metadatas"]):
        meta = meta or {}
        episodic.append({
            "ts": meta.get("ts", ""), "conv_id": meta.get("conv_id", ""),
            "summary": doc or "", "full_text": meta.get("full_text", ""),
        })
    episodic.sort(key=lambda e: e["ts"], reverse=True)

    # 3. 用户健康档案（取最新一条，与对话链路一致）
    profile = await _memory._get_profile(user_id)

    return {"user_id": user_id, "profile": profile,
            "working": convs, "episodic": episodic[:limit]}


@app.post("/eval/run")
async def run_eval(body: Optional[EvalRunInput] = None):
    """运行内置评测用例，返回评测报告。"""
    if _evaluator is None:
        raise HTTPException(503, "服务未就绪")
    from evaluation.evaluator import DEFAULT_DIALOG_CASES, DEFAULT_INTENT_CASES, IntentTestCase

    if body and body.intent_cases is not None:
        intent_cases = [
            IntentTestCase(
                message=c.message,
                expected_intent=c.expected_intent,
                context=c.context,
            )
            for c in body.intent_cases
        ]
    else:
        intent_cases = DEFAULT_INTENT_CASES

    if body and body.dialog_cases is not None:
        dialog_cases = [
            c.model_dump(exclude_none=True)
            for c in body.dialog_cases
        ]
    else:
        dialog_cases = DEFAULT_DIALOG_CASES

    report = await _evaluator.run(
        intent_cases=intent_cases,
        dialog_cases=dialog_cases,
    )
    return {
        "pass_rate":       report.pass_rate,
        "total":           report.total,
        "passed":          report.passed,
        "avg_scores":      report.avg_scores,
        "regressions":     report.regressions,
        "recommendations": report.recommendations,
        "results": [
            {
                "test_id": r.test_id,
                "passed": r.passed,
                "scores": r.scores,
                "detail": r.detail,
                "metadata": r.metadata,
            }
            for r in report.results
        ],
    }


# ── 交互式 CLI ────────────────────────────────────────────────────────────────
async def _cli():
    print(BANNER)
    print("HealthMind CLI — 输入 quit 退出\n")

    from agents.agent_orchestrator import AgentOrchestrator, Request
    from memory.conversation_memory import MemoryManager, MsgRole
    from core.skill_loader import SkillManager

    cfg = _anthropic_cfg()
    skill_manager = SkillManager(
        root_dir=os.getenv("HEALTHMIND_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills")),
        max_prompt_chars=int(os.getenv("HEALTHMIND_SKILLS_MAX_PROMPT_CHARS", "5000")),
    )
    skill_manager.load()
    orch = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=skill_manager,
    )
    mem  = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "localhost"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/tmp/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    user_id, conv_id = "cli_user", str(uuid.uuid4())

    while True:
        try:
            msg = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见 ʕ•ᴥ•ʔ")
            break
        if not msg or msg.lower() in ("quit", "exit", "退出"):
            print("再见 ʕ•ᴥ•ʔ")
            break

        ctx = await mem.get_context(user_id, conv_id, query=msg)
        history = [
            {"role": m.role.value, "content": m.content}
            for m in ctx.recent_messages[-5:]
        ] if ctx.recent_messages else None
        req = Request(message=msg, user_id=user_id, conv_id=conv_id, context=ctx.to_prompt_text(), history=history)
        result = await orch.run(req)

        await mem.add_message(user_id, conv_id, MsgRole.USER, msg)
        await mem.add_message(user_id, conv_id, MsgRole.ASSISTANT, result.response)

        print(f"\nHealthMind [{result.agent_type.value}]: {result.response}\n")


if __name__ == "__main__":
    if "--cli" in sys.argv:
        asyncio.run(_cli())
    else:
        uvicorn.run(
            "api.main:app",
            host=os.getenv("API_HOST", "0.0.0.0"),
            # 默认 8002：8000 常被 IDE 预览服务抢占（127.0.0.1 精确绑定优先），导致请求黑洞
            port=int(os.getenv("API_PORT", "8002")),
            reload=os.getenv("APP_ENV") == "development",
        )
