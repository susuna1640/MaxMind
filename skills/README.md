# HealthMind Skills 文档

HealthMind 启动时会从 `HEALTHMIND_SKILLS_DIR` 读取 Skills，并在匹配用户请求时注入到对应 Agent 的 system prompt。Skills 适合维护健康咨询规范、饮食/运动指导原则、安全边界、就医升级规则和禁止事项。

当前内置四类 Skills：

```text
skills/general_health/SKILL.md    # 通用健康咨询：澄清、个性化建议、就医提示
skills/nutrition/SKILL.md         # 饮食营养：养胃食疗、营养搭配、减脂饮食
skills/fitness_sleep/SKILL.md     # 运动睡眠：运动计划、心率、作息调理、失眠改善
skills/safety_boundary/SKILL.md   # 安全边界（全局）：不诊断、不开药、红旗症状就医、隐私保护
```

## Skill 文件格式

推荐每个 Skill 使用独立目录，并将主文件命名为 `SKILL.md`：

```text
skills/<skill_name>/SKILL.md
```

文件顶部使用简单 front matter：

```markdown
---
name: 饮食营养处理规范
description: 适用于 NutritionAgent 的饮食、营养与食疗指导规范
keywords: 饮食,吃什么,养胃,食疗,减脂,营养,忌口
agents: nutrition
enabled: true
---
```

字段说明：

- `name`：Skill 展示名称，会出现在注入给模型的 prompt 中。
- `description`：简短说明，方便 `/skills` 接口排查。
- `keywords`：触发关键词，用户消息命中后才注入；多个关键词用英文逗号或中文逗号分隔均可。**不填 keywords 则作为全局 Skill 每次对话都注入**（如 safety_boundary）。
- `agents`：适用 Agent，可填 `health`、`nutrition`、`fitness`、`escalation`，多个值用逗号分隔。
- `enabled`：是否启用，支持 `true/false`。

## 编写要求

- 重要规则放在文档前半部分，因为过长内容会按 prompt 预算截断。
- 一类 Skill 只描述一类职责，不要把饮食、运动、通用咨询规则混在一个文件里。
- 必须包含“角色定位”“处理流程”“升级条件”“禁止事项”等稳定章节。
- 健康域升级条件统一为「建议就医」：涉及疾病诊断、用药、急症、特殊人群（孕产妇/婴幼儿/老人）时必须写明。
- 传统养生内容必须注明「传统经验，仅供参考」。
- 对用户隐私（病史、体检数据、用药记录、身份证号等）必须写明禁止收集或禁止公开。
- 对无法保证的事项使用保守措辞，例如“通常”“一般建议”“需要医生评估后确认”。

## 热加载

修改 Skill 文件后，不需要重启服务，调用：

```bash
curl -X POST http://localhost:8000/skills/reload
```

查看加载结果和解析错误：

```bash
curl http://localhost:8000/skills
```
