"""知识库构建与检索。

Markdown 文档先按二级标题分节，再按语义边界切片并写入 Chroma。
查询阶段先做向量召回，再结合词项覆盖率进行混合重排。
"""

import hashlib
import json
import math
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHROMA_DIR, COLLECTION_NAME, KNOWLEDGE_DIR, settings
from src.embedding import get_embedder

META_FILE = CHROMA_DIR / "_meta.json"
_client = None
_collection = None
_knowledge_lock = threading.Lock()

# 先按中文语义边界切，再退化为按字符切
SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    separators=["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""],
)


# ------------------------- 底层连接 -------------------------

def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        with _knowledge_lock:
            if _client is None:
                CHROMA_DIR.mkdir(parents=True, exist_ok=True)
                _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def get_collection(rebuild: bool = False):
    global _collection
    if _collection is not None and not rebuild:
        return _collection

    client = get_client()
    with _knowledge_lock:
        if rebuild:
            try:
                client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass  # 不存在时忽略
            _collection = None
        if _collection is None:
            # cosine 距离：1 - 余弦相似度，数值越小越相似
            _collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
    return _collection


def corpus_fingerprint() -> str:
    """基于文档名和内容生成稳定指纹，用于识别同片段数下的内容变更。"""
    digest = hashlib.sha256()
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _stable_id(*parts: str) -> str:
    payload = "\0".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def _read_meta() -> Dict:
    if META_FILE.exists():
        try:
            return json.loads(META_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _write_meta(meta: Dict) -> None:
    META_FILE.parent.mkdir(parents=True, exist_ok=True)
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------- 文档解析与切分 -------------------------

def _split_sections(text: str, fallback_title: str) -> List[Dict[str, str]]:
    """按二级标题（## ）切章节，返回 [{title, content}]。"""
    parts = re.split(r"\n(?=##\s)", text)
    sections: List[Dict[str, str]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lines = part.splitlines()
        if lines[0].startswith("##"):
            title = lines[0].lstrip("#").strip()
            body = "\n".join(lines[1:]).strip()
        else:
            # 一级标题或前言部分
            title = fallback_title
            body = "\n".join(line for line in lines if not line.startswith("#")).strip()
        if body:
            sections.append({"title": title, "content": body})
    return sections


def load_chunks() -> List[Dict]:
    """把 data/knowledge 下的 markdown 全部切分成 chunk。"""
    chunks: List[Dict] = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        # 文件名形如「01-猫咪喂养.md」，去掉编号作为兜底标题
        stem = re.sub(r"^\d+[-_]?", "", path.stem)
        for section in _split_sections(raw, fallback_title=stem):
            pieces = SPLITTER.split_text(section["content"])
            for idx, piece in enumerate(pieces):
                piece = piece.strip()
                if len(piece) < 20:  # 太短的碎片没有检索价值
                    continue
                chunks.append(
                    {
                        "id": _stable_id(path.stem, section["title"], piece, str(idx)),
                        "text": piece,
                        # 把标题拼进正文，提升召回（用户问「疫苗」时标题也能命中）
                        "embed_text": f"{section['title']}\n{piece}",
                        "source": path.name,
                        "title": section["title"],
                        "category": stem,
                    }
                )
    return chunks


# ------------------------- 构建 -------------------------

def build_knowledge_base(rebuild: bool = True, verbose: bool = True) -> Dict:
    """全量重建知识库，返回统计信息。"""
    embedder = get_embedder()
    chunks = load_chunks()

    collection = get_collection(rebuild=rebuild)
    if chunks:
        embeddings = embedder.embed_documents([c["embed_text"] for c in chunks])
        # 分批写入，避免单次请求过大
        batch = 64
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            collection.add(
                ids=[c["id"] for c in part],
                documents=[c["text"] for c in part],
                metadatas=[
                    {"source": c["source"], "title": c["title"], "category": c["category"]}
                    for c in part
                ],
                embeddings=embeddings[i:i + batch],
            )

    _write_meta(
        {
            "provider": embedder.provider,
            "dim": embedder.dim,
            "chunks": len(chunks),
            "fingerprint": corpus_fingerprint(),
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        }
    )
    _TERM_INDEX.update(count=-1, idf={})
    stats = {
        "chunks": len(chunks),
        "sources": sorted({c["source"] for c in chunks}),
        "provider": embedder.provider,
        "dim": embedder.dim,
    }
    if verbose:
        print(f"[knowledge] 构建完成：{stats['chunks']} 个片段，{len(stats['sources'])} 篇文档，"
              f"向量维度 {stats['dim']}（{stats['provider']}）")
    return stats


def ensure_knowledge_base(verbose: bool = False) -> Dict:
    """应用启动时调用：库为空或向量化方案变化时自动重建。"""
    embedder = get_embedder()
    meta = _read_meta()
    collection = get_collection()
    count = collection.count()

    need_rebuild = (
        count == 0
        or meta.get("provider") != embedder.provider
        or meta.get("dim") != embedder.dim
        or meta.get("fingerprint") != corpus_fingerprint()
        or meta.get("chunk_size") != settings.chunk_size
        or meta.get("chunk_overlap") != settings.chunk_overlap
    )
    if need_rebuild:
        if verbose:
            print(f"[knowledge] 需要重建（现有 {count} 条，provider={embedder.provider}, dim={embedder.dim}）")
        return build_knowledge_base(rebuild=True, verbose=verbose)

    return {"chunks": count, "sources": list_sources(), "provider": meta.get("provider"), "dim": meta.get("dim")}


# ------------------------- 检索 -------------------------

def _terms(text: str) -> set:
    """抽取用于匹配的词项。

    中文没有空格，使用「去标点 + 去停用字 + 字符 2-gram」：
    - 去标点：问号、逗号不应该参与匹配
    - 去停用字：的/了/是/吗 这类字组合出来的 bigram 没有区分度
    - 2-gram：在中文上等价于粗分词，不需要额外依赖 jieba

    该组合可避免标点和无效二元词干扰，同时不引入额外分词依赖。
    """
    compact = re.sub(r"[^\w\u4e00-\u9fa5]+", "", text or "")
    compact = "".join(ch for ch in compact if ch not in STOP_CHARS)
    grams = {compact[i:i + 2] for i in range(len(compact) - 1)} if len(compact) > 1 else set()
    # 英文/数字单独按词处理，避免被 2-gram 切碎
    grams |= set(re.findall(r"[a-z0-9]{2,}", (text or "").lower()))
    return grams


# 停用字：参与组合会产生大量无效 bigram
STOP_CHARS = set("的了是在和就都也还很太吗呢吧啊呀哦嘛么有要会能可以把被让给对从到与及或但而所以如果这样那样一个我你他她它们这那只不没")

# 品种词：只用来做过滤（猫的服务不推给狗），不参与主题相关性打分。
# 否则「狗狗拉稀」会被《剪指甲与足部护理》这种纯犬类文档顶到第一。
PET_TERMS = {"狗", "猫", "狗狗", "猫咪", "犬", "喵", "毛孩子", "宠物"}

# 词项索引缓存：key 为当前片段数，库发生变化时自动失效
_TERM_INDEX = {"count": -1, "idf": {}}


def _get_idf(collection) -> Dict[str, float]:
    """计算语料级 IDF：越少见的词项权重越高。

    没有 IDF 时，"猫"、"喂" 这种高频字会把所有文档的分数拉到同一水平，
    排序完全失效。IDF 让「布偶」「拉稀」这类有信息量的词占主导。
    """
    count = collection.count()
    if _TERM_INDEX["count"] == count:
        return _TERM_INDEX["idf"]

    data = collection.get(include=["documents"])
    docs = data.get("documents") or []
    df: Dict[str, int] = {}
    for text in docs:
        for term in _terms(text):
            df[term] = df.get(term, 0) + 1

    total = len(docs) or 1
    idf = {term: math.log((total + 1) / (freq + 1)) + 1.0 for term, freq in df.items()}
    _TERM_INDEX.update(count=count, idf=idf)
    return idf


def search(query: str, top_k: Optional[int] = None) -> List[Dict]:
    """检索：先向量召回，再混合重排。

    余弦相似度容易受片段长度影响，因此先扩大向量召回范围，
    再用 IDF 加权词项覆盖率进行重排：

        score = 0.65 * 覆盖率 + 0.35 * 余弦

    覆盖率 = 命中的查询词项 IDF 之和 / 查询词项总 IDF。
    它的含义是「这个问题有多少信息量被这个片段覆盖了」，
    取值 0-1、绝对值可比，阈值才有意义。

    该流程对应轻量级 recall-then-rerank；更换 embedding 或 rerank 模型时，
    可相应调整混合权重。
    """
    top_k = max(1, min(int(top_k or settings.top_k), 20))
    collection = get_collection()
    if collection.count() == 0:
        return []

    embedder = get_embedder()
    query_vec = embedder.embed_query(query)
    idf = _get_idf(collection)

    # 第一步：向量召回，取 4 倍候选（至少 12 条），保证重排有得选
    recall_n = min(max(top_k * 4, 12), collection.count())
    result = collection.query(
        query_embeddings=[query_vec],
        n_results=recall_n,
        include=["documents", "metadatas", "distances"],
    )

    q_terms = _terms(query) - PET_TERMS   # 品种词不参与主题打分
    q_total = sum(idf.get(t, 1.0) for t in q_terms) or 1.0

    hits: List[Dict] = []
    for text, meta, dist in zip(
        (result.get("documents") or [[]])[0],
        (result.get("metadatas") or [[]])[0],
        (result.get("distances") or [[]])[0],
    ):
        cosine = max(0.0, min(1.0, 1.0 - float(dist)))
        matched = sum(idf.get(t, 1.0) for t in (q_terms & (_terms(text) - PET_TERMS)))
        coverage = matched / q_total
        score = 0.65 * coverage + 0.35 * cosine
        hits.append(
            {
                "content": text,
                "score": round(score, 4),
                "cosine": round(cosine, 4),
                "coverage": round(coverage, 4),
                "source": meta.get("source", ""),
                "title": meta.get("title", ""),
                "category": meta.get("category", ""),
            }
        )

    # 第二步：按混合分重排，取前 top_k
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:top_k]


# ------------------------- 上传扩展 -------------------------

def add_uploaded_text(filename: str, text: str) -> int:
    """把用户上传的 txt/md 内容加入知识库，返回新增片段数（Streamlit 用）。"""
    embedder = get_embedder()
    collection = get_collection()
    safe_filename = Path(filename).name
    source = f"upload:{safe_filename}"
    sections = _split_sections(text, fallback_title=Path(safe_filename).stem)

    rows: List[Dict] = []
    for section in sections:
        for idx, piece in enumerate(SPLITTER.split_text(section["content"])):
            piece = piece.strip()
            if len(piece) < 20:
                continue
            rows.append(
                {
                    "id": _stable_id(source, section["title"], piece, str(idx)),
                    "text": piece,
                    "embed_text": f"{section['title']}\n{piece}",
                    "title": section["title"],
                }
            )

    embeddings = embedder.embed_documents([r["embed_text"] for r in rows]) if rows else []

    # 同名文件再次上传时替换旧内容，避免新旧片段同时被检索。
    collection.delete(where={"source": source})
    if rows:
        collection.upsert(
            ids=[r["id"] for r in rows],
            documents=[r["text"] for r in rows],
            metadatas=[{"source": source, "title": r["title"], "category": "用户上传"} for r in rows],
            embeddings=embeddings,
        )

    meta = _read_meta()
    meta["chunks"] = collection.count()
    _write_meta(meta)
    _TERM_INDEX.update(count=-1, idf={})
    return len(rows)


def list_sources() -> List[str]:
    """知识库覆盖的文档列表（去重）。"""
    collection = get_collection()
    if collection.count() == 0:
        return []
    data = collection.get(include=["metadatas"])
    return sorted({(m or {}).get("source", "") for m in data.get("metadatas", []) if m})


def stats() -> Dict:
    collection = get_collection()
    return {
        "chunks": collection.count(),
        "sources": list_sources(),
        "provider": _read_meta().get("provider", "unknown"),
        "dim": _read_meta().get("dim", 0),
    }
