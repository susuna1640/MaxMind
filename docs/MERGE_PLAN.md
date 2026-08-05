# HealthMind × rag-health-assistant 融合方案

> 目标：把 MaxMind（企业级智能客服框架）与 rag-health-assistant（个性化健康管理助手）融合为一个简历项目。
> **框架与工程能力全部沿用 MaxMind**，**业务背景换成个性化健康管理助手**。
> 状态：决定已确认，实施中。

## 0. 已确认决定（2026-08-05）

1. 项目名改为 **HealthMind**（全局品牌名替换；工作区目录名暂不改）
2. 健康工具调用方式：**方案 A**（`/chat` 按意图/关键词触发计算，结果注入 context）
3. 图片分析：**不移植**
4. COMPLAINT/FEEDBACK 意图类别：**删除**
5. 中医养生语料（养胃/养肝/祛湿）：**保留**，进知识库与评测，并在安全边界 Skill 与回复中加「传统养生经验，仅供参考」免责声明

## 1. 融合策略

- **保留 MaxMind 的全部架构亮点**（这些是简历核心）：
  - 三路融合意图识别（LLM + Embedding + 关键词）
  - 三层 Agent 路由（意图路由 → 性能路由 → 降级路由）与并行协作
  - Skills 热加载与动态 prompt 注入
  - MCP 风格工具框架（查询改写、重排、熔断、缓存、降级）
  - 三级记忆（Redis 工作记忆 + ChromaDB 情景记忆 + 用户画像）
  - 性能监控（Prometheus/Grafana/告警）与四维评测 + 回归检测
  - FastAPI + Vue 前端 + Docker Compose 部署
- **从 rag-health-assistant 移植领域资产**：
  - 健康计算工具：BMI、饮水量估算、睡眠作息规划、运动心率估算（`backend/app/services/health_tool_service.py`）
  - 安全检测：红旗症状（red flag）危险信号识别（`backend/app/tools/safety_tool.py`）
  - 健康知识库：6 篇健康指南（`examples/*.md`：运动、体质、饮食、季节、睡眠、综合养生）
  - 个性化画像字段：年龄、性别、健康状态 → 融入 MaxMind 的用户画像
  - 评测语料：`eval_data.json` 的养生问答对
- **不引入**：rag-health-assistant 的 Gradio 前端、LangChain 依赖、ReAct executor（MaxMind 已有等价的编排与工具链路）。图片分析（image_tool）作为可选项，见 §6 待确认。

## 2. 业务设定（融合后的产品定位）

MaxMind 变为「个性化健康管理助手」：

| 客服域（现状） | 健康域（目标） |
|---|---|
| GeneralAgent 通用客服 | HealthAgent 通用健康顾问（默认） |
| TechnicalAgent 技术支持 | NutritionAgent 饮食营养顾问 |
| BillingAgent 账单服务 | FitnessAgent 运动睡眠顾问 |
| ESCALATION 转人工 | ESCALATION 就医预警（红旗症状 → 建议线下就医/急诊） |

升级机制语义迁移：「低置信度/紧急 → 转人工客服」变为「红旗症状/急症信号 → 优先就医提示」，这在健康场景比客服场景更自然，是加分项。

## 3. 逐项修改清单

### 3.1 意图识别 — `core/intent_recognizer.py`

- `IntentCategory` 枚举改为健康域：
  - `CONSULT`（健康咨询）、`NUTRITION`（饮食营养）、`FITNESS`（运动/睡眠）、`CALCULATE`（BMI/饮水/心率等计算请求）、`PROFILE`（记录/更新个人健康档案）、`ESCALATION`（急症/就医）、`GREETING`、`OTHER`
  - 保留 `QUERY`/`COMPLAINT`/`FEEDBACK` 与否见 §6（建议精简）
- `_TEMPLATES` few-shot 模板全部换成健康域示例（如「我最近总是失眠怎么办」→ FITNESS）
- `_pattern_recognize` 关键词表换成健康域（失眠/减肥/BMI/吃什么/运动/胸痛…）
- `_URGENCY_KEYWORDS`：CRITICAL 级别由红旗症状关键词承担（胸痛、呼吸困难、昏迷、抽搐、大出血、剧烈头痛、高烧不退等，复用 safety_tool 的词表）
- `_extract_entities` 实体字段改为：`symptom`（症状）、`body_part`、`age`、`height`、`weight`、`goal`（健康目标）

### 3.2 Agent 编排 — `agents/agent_orchestrator.py`

- `AgentType`：`GENERAL→HEALTH`、`TECHNICAL→NUTRITION`、`BILLING→FITNESS`，`ESCALATION` 语义改为就医预警
- 三个 Agent 的 `system_prompt` 全部重写为健康顾问角色，并内置统一安全边界：
  - 不做诊断、不开处方药、不替代医生；涉及疾病判断一律建议就医
  - 回复中涉及计算结果时引用工具输出，不自行编造数值
- `_INTENT_ROUTING` 路由表按新意图映射
- `_collaboration_targets` 复合问题关键词改为健康域（如「减脂」= 饮食 + 运动并行协作）
- `_needs_escalation` 关键词改为：「建议就医」「尽快就医」「急诊」「医院」等
- `Request` 可选增加 `user_profile` 字段，把用户画像传给健康计算工具

### 3.3 安全前置检查（新增亮点）

- 在编排器 `run()` 入口或 `/chat` 链路加一步 **红旗症状检测**：
  - 命中 `RED_FLAG_KEYWORDS` → urgency 强制 CRITICAL → 路由到 ESCALATION，回复以就医提示优先
  - 对应 rag-health-assistant 的 `safety_tool.py`，在 MaxMind 里实现为编排器前置步骤或一个注册到 MCP 框架的工具

### 3.4 MCP 工具 — `mcp/tool_manager.py` + `api/main.py`

- `tool_manager.py` 框架本身不动，只在 `api/main.py` 注册新工具：
  - `health_calculators`：移植 `health_tool_service.py` 的四个确定性计算（BMI、饮水、睡眠作息、运动心率），注册为 Tool，带 schema 与缓存
  - `detect_red_flags`：移植红旗症状检测（也可选择不注册为工具，只做编排前置检查，见 §6）
  - `knowledge_search` 保留，知识库内容替换（见 3.5）
- Agent 侧需要能调用工具：当前 MaxMind 的 RAG 是在 `/chat` 里统一注入 context 的，健康计算工具可沿用同一模式（在 `/chat` 里根据意图/关键词触发计算，把结果注入 context），改动最小；如想让 Agent 自主选工具再另议（见 §6）

### 3.5 知识库与数据 — `mcp/knowledge_base.py`、`data/`

- `data/demo_docs/`：
  - 删除/替换 `sample_knowledge.json`（客服产品介绍、订阅计划等）与 `troubleshooting.md`
  - 导入 rag-health-assistant 的 `examples/*.md` 六篇指南（转为 `[{title, content}]` JSON 或直接 md 上传）
- `KnowledgeBase._load_default_docs()` 的默认文档同步替换
- 清空重建 `data/chroma`（collection 内容已变；collection 名可保持 `knowledge_base`）
- `_should_use_knowledge()` 的业务关键词改为健康域（失眠、养胃、湿气、BMI、减脂…），寒暄跳过逻辑保留

### 3.6 记忆与个性化 — `memory/conversation_memory.py`

- 三级记忆架构不动（这是最大亮点），调整**用户画像的语义**：
  - 画像字段引导词从客服偏好改为健康档案：年龄、性别、身高体重、体质特点、慢性病史、健康目标、作息习惯
  - `update_profile` 的 LLM prompt 相应改写；画像注入 prompt 的格式改为「[用户健康档案]」
- 情景记忆（ChromaDB）无需改动，天然记录健康对话历史，体现「跨会话个性化」

### 3.7 Skills — `skills/`

- 删除三个客服 Skill，新建健康域 Skills（保持 SKILL.md + front matter 规范）：
  - `skills/nutrition/SKILL.md` — 饮食营养处理规范（agents: nutrition）
  - `skills/fitness_sleep/SKILL.md` — 运动与睡眠指导规范（agents: fitness）
  - `skills/general_health/SKILL.md` — 通用健康咨询规范（agents: health）
  - `skills/safety_boundary/SKILL.md` — 安全边界全局 Skill（无 keywords = 全局注入）：不诊断、不开药、红旗症状必须建议就医、隐私（病史/体检数据）保护
- `skills/README.md` 同步改写示例与 agents 取值说明（`health`、`nutrition`、`fitness`）
- 内容素材可从 rag-health-assistant 的 `examples/*.md`、health_tool_service 的免责声明改写而来

### 3.8 评测 — `evaluation/evaluator.py`、`data/eval/`

- `DEFAULT_INTENT_CASES`：换成健康域意图用例
- `DEFAULT_DIALOG_CASES`：移植 `eval_data.json` 的养生问答（养胃、养肝、湿气、睡眠、秋季养生等），expected_keywords 可直接复用
- 建议新增「安全评测」用例：红旗症状输入必须触发就医提示（可作为 dialog case 的断言）
- `data/eval/baseline.json` 是历史客服基线，融合后需跑一次 `/eval/run` 重新生成
- evaluator.py 框架（LLM-as-Judge、回归检测）代码本身不动

### 3.9 API 层 — `api/main.py`

- `BANNER`、FastAPI title 改为健康管理助手（名字见 §6）
- `ChatResponse` 可加 `red_flag: bool` 字段，展示安全检测结果
- CLI（`_cli`）提示语改写
- 端点结构、Skills/Knowledge/Eval/Monitor 路由全部保留

### 3.10 前端 — `MaxMindFrontend/`

- `src/App.vue`：标题、欢迎语、示例提问改为健康域（如「身高170体重70，帮我算BMI」「最近失眠怎么调理」）
- `src/lib/backends.js`：仅品牌文案（如有）
- 功能不动

### 3.11 配置与文档

- `.env.example` / `.env.example.env`：无结构性变化，可补健康工具开关
- `docker-compose.yml`、`Dockerfile`：不动（服务编排与领域无关）
- `config/grafana/dashboards`、`config/alerts`：如有客服措辞则改写
- 根目录 README / `docs`（如新建）：改写项目介绍为「个性化健康管理助手」叙事，突出融合后的完整能力栈

## 4. 保持不动的部分

- 三路意图融合、投票、LRU 缓存、在线学习 `learn()` 的机制代码
- 编排器性能路由 `routing_score()`、并行协作、降级链路机制
- MCP 框架的查询改写/重排/熔断/缓存/降级
- 三级记忆、上下文窗口管理、异步画像更新
- PerformanceMonitor、Prometheus/Grafana 接入
- Docker Compose 部署链路

## 5. 建议实施顺序

1. 意图识别域迁移（3.1）+ Agent 编排域迁移（3.2）+ 安全前置检查（3.3）——核心链路
2. 健康工具注册（3.4）+ 知识库替换（3.5）——能力补齐
3. 记忆画像语义调整（3.6）+ Skills 重写（3.7）——个性化与规则层
4. 评测用例迁移 + 重跑基线（3.8）
5. API/前端文案、文档（3.9–3.11）
6. 端到端验证：启动服务 → 普通健康问答 / BMI 计算 / 红旗症状 / 多轮记忆 / `/eval/run` 回归

## 6. 待确认的开放问题

1. **项目名**：融合后仍叫 MaxMind，还是改为 MaxMind Health / 其他健康向名称？
2. **健康工具调用方式**：
   - 方案 A（推荐，改动小）：沿用现有模式，`/chat` 按意图/关键词触发健康计算并把结果注入 context；
   - 方案 B（更有含金量）：让 Agent 通过 MCP 框架自主决定调工具（需要给 Agent 加 tool-use 循环，工作量更大）。
3. **图片分析**：是否移植 rag-health-assistant 的 image_tool（饮食/体检单图片理解）？默认不移植。
4. **意图类别精简度**：是否保留 COMPLAINT/FEEDBACK 等客服残留类别？建议删除，保持健康域纯净。
5. **中医养生内容**（养胃/养肝/祛湿等）是否保留进知识库与评测？rag-health-assistant 的语料偏中医养生，保留可体现知识库真实度，但需注意免责声明。

> 以上开放问题均已在 §0 确认并实施完毕。

## 7. 实施状态（2026-08-05 全部完成）

- ✅ 意图识别 / Agent 编排域迁移（删除 COMPLAINT/FEEDBACK）
- ✅ 红旗症状安全前置检查（core/safety_checker.py，命中强制路由就医预警，ESCALATION 不参与降级）
- ✅ 健康计算工具（tools/health_calculators.py，方案 A：/chat 关键词触发 + 结果注入 context）
- ✅ 知识库语料替换为健康指南（含中医养生 + 免责声明），data/chroma 已清空待重建
- ✅ 用户画像改为健康档案语义；4 个健康域 Skills（safety_boundary 全局注入）
- ✅ 评测用例迁移（含红旗症状安全用例），旧基线已删除待 /eval/run 重生成
- ✅ 全局品牌名 MaxMind → HealthMind（环境变量/Docker/监控/前端；工作区目录名暂不改）
- ✅ 语法检查 + 安全检测/四个健康工具冒烟测试全部通过

