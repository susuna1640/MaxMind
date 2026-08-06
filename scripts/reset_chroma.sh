#!/bin/bash
# ─────────────────────────────────────────────────────────────
# HealthMind · ChromaDB 一键重置
# 清掉 Docker 容器卷 + 本地降级目录中的全部知识库数据，
# 重启后端后会自动重新导入 6 篇默认健康文档。
# 用法：bash scripts/reset_chroma.sh
# ─────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."

echo "▶ 1/3 停止运行中的后端进程（防止降级到本地目录或占用文件）"
PIDS=$(pgrep -f "api.main" || true)
if [ -n "$PIDS" ]; then
  echo "  发现后端进程: $PIDS"
  kill $PIDS 2>/dev/null || true
  sleep 2
else
  echo "  后端未在运行"
fi

echo "▶ 2/3 重建 Docker 容器与数据卷（chromadb-data / redis-data 会被清空）"
if docker info >/dev/null 2>&1; then
  docker compose -f docker-compose.debug.yml down -v
  docker compose -f docker-compose.debug.yml up -d
else
  echo "  ⚠ Docker 未运行，跳过容器重建（请先打开 Docker Desktop 再跑一次）"
fi

echo "▶ 3/3 清理本地降级目录 data/chroma（移入备份目录，不直接删除）"
if [ -d data/chroma ] && [ "$(ls -A data/chroma 2>/dev/null)" ]; then
  BACKUP="data/chroma_backup_$(date +%Y%m%d_%H%M%S)"
  mv data/chroma "$BACKUP"
  mkdir data/chroma
  echo "  已备份到 $BACKUP"
else
  echo "  本地目录为空，跳过"
fi

echo ""
echo "✅ 重置完成。请重启后端："
echo "   LOG_LEVEL=INFO /Users/susuna/miniconda3/envs/echomind/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"
echo "   启动日志出现「知识库导入 6 个文档片段」即代表建库成功，前端 /knowledge 页可直观查看。"
