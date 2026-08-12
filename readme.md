# HealthMind

HealthMind 是一个个性化健康管理 Agent 系统。项目目标不是做单一问答机器人，而是展示一套可解释、可扩展、可评测的健康咨询架构：先保证安全边界，再结合记忆、知识库和确定性工具，最后由多 Agent 编排生成回复。

## 项目目标

- 安全优先：红旗症状先分流到就医预警链路，避免普通养生建议延误就医。
- 个性化：使用 Redis 工作记忆、ChromaDB 情景记忆和用户健康档案，让回复能结合用户长期背景。
- 可控工具：BMI、饮水量、睡眠作息、运动心率、环境健康和 Apple Watch 数据由工具提供，避免模型编造关键数值。
- 专业分工：HealthAgent、NutritionAgent、FitnessAgent、MedicalAlertAgent 按意图路由，复杂问题可并行协作。
- 可运营：保留 Skills 热加载、MCP 工具治理、性能监控、评测与回归检测。

## 主对话流程

`POST /chat` 的端到端流程分为 6 个阶段：

1. 安全分流：规则和 LLM 复核红旗症状，必要时强制进入就医预警。
2. 记忆读取：读取近期对话、情景记忆和用户健康档案。
3. 上下文增强：按需注入知识库检索结果、确定性健康工具结果和环境数据。
4. 编排执行：识别意图，路由到合适 Agent；复合问题可并行调用多个 Agent。
5. 记忆写入：保存本轮用户消息和 Agent 回复。
6. 画像更新：异步更新用户健康档案，不阻塞本轮响应。

职责边界：

- `/chat` 负责安全前置、记忆、RAG、快通道工具和响应落盘。
- `AgentOrchestrator` 负责意图识别、Agent 路由、并行协作、降级和升级标记。
- `BaseAgent` 负责构建 prompt、动态注入 Skills，并在允许时执行 function calling 慢通道。
- `MCPToolManager` 负责工具注册、参数校验、缓存、超时、熔断、降级、查询改写和重排。

## 常用命令

清空并重建 ChromaDB 数据：

```bash
bash scripts/reset_chroma.sh
```

启动调试依赖：

```bash
cd /Users/susuna/Desktop/MyProject/MaxMind
docker compose -f docker-compose.debug.yml up -d
docker compose -f docker-compose.debug.yml ps
```

启动 Python 后端：

```bash
cd /Users/susuna/Desktop/MyProject/MaxMind
LOG_LEVEL=INFO /Users/susuna/miniconda3/envs/echomind/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

启动 Vue 前端：

```bash
cd /Users/susuna/Desktop/MyProject/MaxMind/MaxMindFrontend
npm run dev
```


