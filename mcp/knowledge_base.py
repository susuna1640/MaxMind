"""
RAG 知识库 —— 基于 ChromaDB 的真实检索实现。

功能：
  1. 文档导入：将文本切片后存入 ChromaDB（自动生成 Embedding）
  2. 语义检索：根据 query 从知识库中检索最相关的文档片段
  3. 与 MCP 工具框架集成：作为 knowledge_search 工具的真实 handler

ChromaDB 在这里的角色：
  - memory/ 中用于存储对话记忆（情景记忆 + 用户画像）
  - 这里用于存储知识库文档（RAG 检索）
  两者是不同的 collection，互不干扰。
"""
import hashlib
import logging
from typing import Any, Dict, List, Optional

import chromadb

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    基于 ChromaDB 的 RAG 知识库。

    ChromaDB 内置了 Embedding 模型（all-MiniLM-L6-v2），
    调用 add() 时自动生成向量，query() 时自动做语义匹配。
    不需要额外调用 Anthropic Embeddings API。
    """

    COLLECTION_NAME = "knowledge_base"

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
    ):
        # 优先连接独立 ChromaDB 服务（服务端内置 embedding 模型，客户端无需下载）
        self._use_server = False
        try:
            # HttpClient 默认也会初始化 ChromaDB telemetry；显式关闭避免 posthog 兼容性错误日志。
            self._client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            self._client.heartbeat()
            self._use_server = True
            logger.info(f"知识库 ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception:
            logger.info(f"知识库 ChromaDB 服务不可用，使用本地模式: {chroma_path}")
            self._client = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        # 使用服务端时不传 embedding_function，让服务端处理
        # 本地模式时也不传，使用 ChromaDB 默认的（会触发模型下载）
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "HealthMind 健康知识库"},
        )

        # 如果知识库为空，导入默认文档
        if self._collection.count() == 0:
            self._load_default_docs()

    # ── 文档管理 ──────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict[str, str]]) -> int:
        """
        批量导入文档到知识库。

        documents 格式: [{"title": "...", "content": "..."}, ...]
        长文档会自动切片（每片 500 字）。
        """
        ids, docs, metas = [], [], []

        for doc in documents:
            title   = doc.get("title", "")
            content = doc.get("content", "")
            chunks  = self._chunk_text(content, chunk_size=500)

            for i, chunk in enumerate(chunks):
                doc_id = hashlib.md5(f"{title}_{i}_{chunk[:50]}".encode()).hexdigest()
                ids.append(doc_id)
                docs.append(chunk)
                metas.append({"title": title, "chunk_index": i, "total_chunks": len(chunks)})

        if ids:
            # ChromaDB 会自动生成 Embedding
            self._collection.add(ids=ids, documents=docs, metadatas=metas)
            logger.info(f"知识库导入 {len(ids)} 个文档片段")

        return len(ids)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        语义检索：根据 query 返回最相关的文档片段。

        ChromaDB 内部自动将 query 转为向量，与存储的文档向量做余弦相似度匹配。
        """
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
        )

        items = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                items.append({
                    "title":    meta.get("title", ""),
                    "content":  doc,
                    "score":    round(1.0 - dist, 4),  # ChromaDB 返回距离，转为相似度
                    "chunk":    meta.get("chunk_index", 0),
                })

        return items

    @property
    def doc_count(self) -> int:
        return self._collection.count()

    # ── MCP 工具 handler ─────────────────────────────────────────────────────

    async def search_handler(self, params: Dict[str, Any], context: Any) -> List[Dict]:
        """
        作为 MCP 工具的 handler 注册。

        MCPToolManager.register(Tool(
            name="knowledge_search",
            handler=kb.search_handler,
            ...
        ))
        """
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        return self.search(query, top_k=top_k)

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """将长文本按 chunk_size 切片，保留语义完整性（按句号/换行切分）。"""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        current = ""
        # 按句子切分
        sentences = text.replace("\n", "。").split("。")
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(current) + len(sent) + 1 > chunk_size:
                if current:
                    chunks.append(current)
                current = sent
            else:
                current = f"{current}。{sent}" if current else sent

        if current:
            chunks.append(current)

        return chunks

    def _load_default_docs(self) -> None:
        """导入默认知识库文档（健康养生场景常见问题）。"""
        default_docs = [
            {
                "title": "养胃要点",
                "content": (
                    "养胃要做到饮食规律，少食多餐，避免生冷油腻。"
                    "多吃小米、山药、南瓜等温和易消化的食物。"
                    "饭后散步 10 分钟有助消化，忌冰饮和凉菜。"
                    "长期胃部不适、反酸、黑便等情况应及时就医。"
                    "以上内容为传统养生经验参考，不能替代医生诊断。"
                ),
            },
            {
                "title": "养肝要点",
                "content": (
                    "养肝最重要的是 23 点前入睡，少生气，多吃绿色蔬菜如菠菜、西兰花，可适量饮用枸杞水。"
                    "伤肝行为包括熬夜、喝酒、暴饮暴食。"
                    "肝功能异常、持续乏力、黄疸等症状应及时就医检查。"
                    "以上内容为传统养生经验参考，不能替代医生诊断。"
                ),
            },
            {
                "title": "湿气调理",
                "content": (
                    "湿气重表现为身体沉重、大便粘马桶、皮肤油腻。"
                    "调理方法有喝红豆薏米水、生姜泡脚、少吃甜食、多运动。"
                    "祛湿是一个渐进过程，需要配合规律作息。"
                    "以上内容为传统养生经验参考，不能替代医生诊断。"
                ),
            },
            {
                "title": "睡眠养生要点",
                "content": (
                    "最佳睡眠时间为 23:00-7:00，午睡 15-30 分钟为宜。"
                    "睡前不碰手机、不剧烈运动、不喝浓茶咖啡。"
                    "失眠可按揉涌泉穴或喝温牛奶辅助放松。"
                    "连续多周严重失眠或伴随焦虑、胸闷、心悸时，应及时寻求专业医生帮助。"
                ),
            },
            {
                "title": "运动安全与心率",
                "content": (
                    "温和有氧运动目标心率可参考（220 - 年龄）的 50%-70%。"
                    "运动中应以能说话但略喘为宜，出现胸痛、头晕应立即停止。"
                    "新手建议从快走、骑行等低冲击运动开始，每周 3-5 次，每次 30 分钟左右。"
                    "有心血管疾病或关节伤痛者，运动前请先咨询医生。"
                ),
            },
            {
                "title": "安全边界与就医提醒",
                "content": (
                    "HealthMind 提供健康养生科普和日常管理建议，不做疾病诊断、不推荐处方药、不替代医生。"
                    "出现胸痛、胸闷、呼吸困难、昏迷、抽搐、大出血、吐血、咯血、黑便、剧烈头痛、偏瘫、高烧不退、意识模糊等红旗症状时，"
                    "请立即就医或拨打急救电话。慢性病患者调整饮食或运动计划前，请先咨询医生。"
                ),
            },
        ]
        self.add_documents(default_docs)
        logger.info(f"已导入默认知识库: {len(default_docs)} 篇文档")
