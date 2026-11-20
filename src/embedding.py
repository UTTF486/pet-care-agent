"""文本向量化（Embedding）。

为什么不直接用 Chroma 自带的默认模型：
它会在首次运行时从 HuggingFace 下载 all-MiniLM-L6-v2（约 80MB），
国内网络经常失败，新手第一步就卡住。所以这里做了两种实现：

1. local（默认）：字符 n-gram + 哈希签名，纯标准库实现，零下载、离线可用。
   对中文短文本检索效果足够（本质是特征哈希版 TF 向量 + 余弦相似度）。
2. dashscope：调用阿里云 text-embedding-v3，语义泛化更好，需要 API Key。

切换 embedding 后必须重建知识库（维度不同），build_kb 脚本会自动检测并提示。
"""

import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from typing import List

from config import settings


class BaseEmbedder:
    """向量化接口。实现类只需提供 embed_documents / embed_query。"""

    provider: str = "base"
    dim: int = 0

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class LocalHashEmbedder(BaseEmbedder):
    """字符 n-gram 哈希向量化。

    原理：
    1. 把文本切成 n-gram（中文 2-gram / 3-gram，英文按词）
    2. 每个 gram 做 md5，取前 4 字节映射到 [0, dim) 作为下标
    3. 用第 5 个字节的奇偶决定 +1/-1（signed hashing，抵消一部分哈希碰撞）
    4. L2 归一化，使余弦相似度等价于点积

    优点：零依赖、确定性、可离线；缺点：没有语义泛化能力
    （"拉稀" 和 "腹泻" 不算近义），所以召回仍靠 n-gram 重合度。
    """

    provider = "local"

    def __init__(self, dim: int = 768):
        self.dim = dim

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = (text or "").lower()
        # 去掉空白，避免 "猫 粮" 和 "猫粮" 被切碎
        compact = re.sub(r"\s+", "", text)
        tokens: List[str] = []

        # 英文 / 数字：按词切，权重最高（出现一次记一次）
        tokens.extend(re.findall(r"[a-z0-9]+", text))

        # 中文：2-gram 与 3-gram 都切，短词靠 2-gram，长词靠 3-gram
        for n in (2, 3):
            if len(compact) < n:
                continue
            for i in range(len(compact) - n + 1):
                tokens.append(compact[i:i + n])
        return tokens

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for token in self._tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign

        # L2 归一化
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class DashScopeEmbedder(BaseEmbedder):
    """阿里云 text-embedding-v3（OpenAI 兼容协议，用标准库发请求，不额外装 SDK）。"""

    provider = "dashscope"
    _URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"

    def __init__(self, api_key: str, model: str = "text-embedding-v3", dim: int = 1024):
        self.api_key = api_key
        self.model = model
        self.dim = dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 接口单次最多 10 条，这里分批
        results: List[List[float]] = []
        for i in range(0, len(texts), 10):
            results.extend(self._call(texts[i:i + 10]))
        return results

    def _call(self, batch: List[str]) -> List[List[float]]:
        payload = json.dumps({"model": self.model, "input": batch}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self._URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
        # 按 index 排序，避免返回乱序导致与文本错位
        items = sorted(body["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]


def get_embedder() -> BaseEmbedder:
    """按配置返回向量化实现；云端失败自动降级到本地。"""
    if settings.embedding_provider == "dashscope" and settings.dashscope_api_key:
        try:
            embedder = DashScopeEmbedder(settings.dashscope_api_key, dim=1024)
            # 探活：真发一条短文本，失败会抛异常
            embedder.embed_documents(["连接测试"])
            return embedder
        except (urllib.error.URLError, KeyError, TimeoutError, OSError) as exc:
            print(f"[embedding] 云端向量化不可用（{exc}），已降级为本地实现")
    return LocalHashEmbedder(dim=settings.embedding_dim)
