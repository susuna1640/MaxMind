"""
亮点：Apple Watch 远程 MCP 接入层（MCP Client 模式）

对接 ice-star-blue/apple-watch-health-mcp 部署的 Cloudflare Worker，
以 Streamable HTTP 传输把 4 个手表工具包装成本地 MCPToolManager 工具，
参与慢通道 function calling 与编排器白名单治理。

网络注意：workers.dev 域名在国内被 DNS 污染 + TCP 封锁，
所有出站请求必须走本地代理（WATCH_PROXY，默认 Clash 的 127.0.0.1:7897）。
"""
import json
import logging
import os
import asyncio
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

# ── 配置 ──────────────────────────────────────────────────────────────────────

WATCH_MCP_BASE = os.getenv(
    "WATCH_MCP_BASE", "https://watch-health-mcp.healthmind-watch.workers.dev"
)
WATCH_MCP_PATH_TOKEN = os.getenv("WATCH_MCP_PATH_TOKEN", "")
# 本地代理：workers.dev 直连不通，经 Clash 等代理出站
WATCH_PROXY = os.getenv("WATCH_PROXY", "http://127.0.0.1:7897")

WATCH_MCP_URL = f"{WATCH_MCP_BASE.rstrip('/')}/mcp/{WATCH_MCP_PATH_TOKEN}"

PROTOCOL_VERSION = "2025-06-18"


def watch_mcp_enabled() -> bool:
    """令牌未配置时不注册手表工具（优雅降级）。"""
    return bool(WATCH_MCP_PATH_TOKEN)


# ── 远程 MCP Client（轻量 JSON-RPC over Streamable HTTP）─────────────────────

class RemoteWatchMCPClient:
    """
    最小化 MCP Client：initialize 握手 + tools/call 转发。

    不做完整 SDK 依赖（项目本地 mcp/ 包与官方 SDK 同名，引入 SDK client 需
    sys.path 技巧；这里只需两个方法，直接用 httpx 更轻、更可控）。
    """

    def __init__(self, url: str, proxy: str, timeout_s: float = 30.0):
        self._url = url
        self._proxy = proxy
        self._timeout = timeout_s
        self._session_id: Optional[str] = None
        self._rpc_id = 0
        self._client: Optional[httpx.AsyncClient] = None

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                proxy=self._proxy,
                timeout=httpx.Timeout(self._timeout),
                trust_env=False,   # 不受 shell 代理环境变量干扰，显式走 WATCH_PROXY
            )
        return self._client

    async def _rpc(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self._rpc_id += 1
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        # 代理对污染域名的首次解析/建连偶发失败，传输层错误重试一次
        last_err: Optional[Exception] = None
        for attempt in range(2):
            client = await self._http()
            try:
                resp = await client.post(
                    self._url,
                    headers=headers,
                    json={"jsonrpc": "2.0", "id": self._rpc_id, "method": method, "params": params},
                )
                resp.raise_for_status()
                sid = resp.headers.get("Mcp-Session-Id")
                if sid:
                    self._session_id = sid
                return resp.json()
            except httpx.TransportError as ex:
                last_err = ex
                if attempt == 0:
                    # 首次建连失败后连接池可能已坏：丢弃重建，稍等再试
                    logger.warning(f"watch MCP 传输错误，重建连接重试: {ex}")
                    await self._reset_client()
                    await asyncio.sleep(0.6)
        raise last_err

    async def _reset_client(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None

    async def ensure_session(self) -> None:
        """initialize 握手（幂等，失败不抛——首次 tools/call 时再试）。"""
        if self._session_id:
            return
        await self._rpc("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "healthmind", "version": "1.0"},
        })

    async def call_tool(self, name: str, params: Dict[str, Any]) -> Any:
        """调用远端工具，返回解析后的结果（JSON 文本自动反序列化）。"""
        await self.ensure_session()
        payload = await self._rpc("tools/call", {"name": name, "arguments": params})
        if "error" in payload:
            raise RuntimeError(f"远端 MCP 错误: {payload['error']}")
        content = (payload.get("result") or {}).get("content") or []
        texts = [c.get("text", "") for c in content if c.get("type") == "text"]
        raw = "\n".join(texts)
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {"text": raw}

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


_client: Optional[RemoteWatchMCPClient] = None


def get_watch_client() -> RemoteWatchMCPClient:
    global _client
    if _client is None:
        _client = RemoteWatchMCPClient(WATCH_MCP_URL, WATCH_PROXY)
    return _client


# ── 工具包装：远端工具 → 本地 Tool ────────────────────────────────────────────

def _make_handler(remote_name: str):
    async def handler(params: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
        return await get_watch_client().call_tool(remote_name, params or {})
    return handler


def _make_fallback(remote_name: str):
    def fallback(params: Dict[str, Any], context: Any, error: Any) -> Any:
        return {
            "tool": remote_name,
            "summary_text": "Apple Watch 数据暂时无法获取（网络或手表未同步），请稍后再试，或先在手表 App 里打开一次同步。",
            "error": str(error),
        }
    return fallback


WATCH_TOOL_DEFS = [
    {
        "name": "watch_health_open_session",
        "description": (
            "Apple Watch 连接自检：查看手表连接状态、数据新鲜度、实时模式是否开启。"
            "用户问「手表连上了吗/数据是最新的吗」时调用。"
        ),
        "schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "watch_get_latest_health",
        "description": (
            "读取用户 Apple Watch 的最新健康指标：心率、静息心率、HRV、血氧、"
            "呼吸频率、步数、距离、活动能量等，含采样时间与数据年龄。"
            "用户问「我现在心率多少/今天走了多少步」时调用。"
        ),
        "schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "watch_get_health_history",
        "description": (
            "读取用户 Apple Watch 健康历史：睡眠保留近 7 天完整阶段（深睡/REM/核心/清醒），"
            "其他指标保留最近 3 次采样。用户问「我昨晚睡得怎么样/最近睡眠/心率趋势」时调用。"
        ),
        "schema": {
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "description": "可选，只查单项指标，如 sleep / heart_rate / hrv / steps",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "watch_measure_now",
        "description": (
            "请求一次实时心率测量：仅当用户已在手表 App 开启实时模式时能拿到新样本，"
            "否则返回提示。用户说「现在测一下心率」时调用。"
        ),
        "schema": {
            "type": "object",
            "properties": {
                "wait_seconds": {
                    "type": "integer",
                    "minimum": 3,
                    "maximum": 25,
                    "default": 15,
                    "description": "等待新心率样本的秒数",
                },
            },
            "additionalProperties": False,
        },
    },
]


def register_watch_tools(tool_manager) -> None:
    """把 4 个手表工具注册进 MCPToolManager（未配置令牌时跳过）。"""
    if not watch_mcp_enabled():
        logger.info("未配置 WATCH_MCP_PATH_TOKEN，跳过 Apple Watch 工具注册")
        return
    from mcp.tool_manager import Tool

    for spec in WATCH_TOOL_DEFS:
        tool_manager.register(Tool(
            name=spec["name"],
            description=spec["description"],
            handler=_make_handler(spec["name"]),
            schema=spec["schema"],
            cache_ttl=0.0,          # 健康数据讲究新鲜度，不缓存
            timeout_s=35.0,         # 远程链路（代理+Worker）留足余量
            fallback=_make_fallback(spec["name"]),
        ))
    logger.info("Apple Watch 远程 MCP 工具注册完成（4 个）")
