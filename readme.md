<!-- 清空chroma的命令 -->
bash scripts/reset_chroma.sh

# docker 部分
<!-- 先确保 Docker Desktop 已打开（菜单栏有鲸鱼图标），然后： -->
cd /Users/susuna/Desktop/MyProject/MaxMind
docker compose -f docker-compose.debug.yml up -d

<!-- 可选：确认容器健康 -->
docker compose -f docker-compose.debug.yml ps

# python 后端部分
cd /Users/susuna/Desktop/MyProject/MaxMind
LOG_LEVEL=INFO /Users/susuna/miniconda3/envs/echomind/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# vue 前端部分
cd /Users/susuna/Desktop/MyProject/MaxMind/MaxMindFrontend

npm run dev



