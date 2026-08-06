"""冒烟测试：DeepSeek Anthropic 兼容端点是否支持 function calling（tools 参数）。

用法: python scripts/test_deepseek_fc.py
"""
import asyncio
import os
from pathlib import Path

import httpx
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

client = AsyncAnthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
    # trust_env=False：直连，绕开本机时断时续的代理
    http_client=httpx.AsyncClient(trust_env=False),
)

TOOLS = [{
    "name": "bmi_calculator",
    "description": "根据身高体重计算 BMI 指数",
    "input_schema": {
        "type": "object",
        "properties": {
            "height_cm": {"type": "number", "description": "身高（厘米）"},
            "weight_kg": {"type": "number", "description": "体重（公斤）"},
        },
        "required": ["height_cm", "weight_kg"],
    },
}]

MODEL = os.getenv("ANTHROPIC_MODEL", "deepseek-chat")


async def main():
    print("── 用例1: 应触发 tool_use（身高170体重70算BMI）──")
    try:
        resp = await client.messages.create(
            model=MODEL, max_tokens=1024, tools=TOOLS,
            messages=[{"role": "user", "content": "我身高170，体重70公斤，帮我算一下BMI"}],
        )
    except Exception as ex:
        print(f"结果: ❌ 调用异常 {type(ex).__name__}: {ex}")
        return
    print(f"stop_reason = {resp.stop_reason}")
    for block in resp.content:
        print(f"  block.type = {block.type}")
        if block.type == "tool_use":
            print(f"  tool name  = {block.name}")
            print(f"  tool input = {block.input}")
    tool_use = next((b for b in resp.content if b.type == "tool_use"), None)
    case1 = tool_use is not None and tool_use.input.get("height_cm") == 170
    print(f"结果: {'✅ 触发且参数正确' if case1 else '❌ 未触发或参数错误'}")

    print()
    print("── 用例2: 不应触发（闲聊）──")
    resp2 = await client.messages.create(
        model=MODEL, max_tokens=256, tools=TOOLS,
        messages=[{"role": "user", "content": "你好，今天心情不错"}],
    )
    triggered = any(b.type == "tool_use" for b in resp2.content)
    print(f"stop_reason = {resp2.stop_reason}, 触发工具 = {triggered}")
    print(f"结果: {'✅ 未误触发' if not triggered else '⚠️ 闲聊也触发了工具'}")

    if tool_use is None:
        print("\n结论: ❌ DeepSeek Anthropic 兼容端点不支持 function calling")
        return

    print()
    print("── 用例3: tool_result 回传后能否继续生成 ──")
    resp3 = await client.messages.create(
        model=MODEL, max_tokens=512, tools=TOOLS,
        messages=[
            {"role": "user", "content": "我身高170，体重70公斤，帮我算一下BMI"},
            {"role": "assistant", "content": resp.content},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tool_use.id,
                 "content": '{"bmi": 24.2, "category": "正常偏高"}'},
            ]},
        ],
    )
    text = "".join(b.text for b in resp3.content if b.type == "text")
    cited = "24.2" in text
    print(f"stop_reason = {resp3.stop_reason}")
    print(f"回复片段: {text[:120]}")
    print(f"结果: {'✅ 正确引用了工具结果' if cited else '⚠️ 未引用工具结果'}")

    print()
    print("结论:", "✅ DeepSeek Anthropic 兼容端点支持 function calling，可以升级")


asyncio.run(main())
