"""服务推荐。

推荐分数融合语义相似度与关键词规则，并结合宠物类型和上下架状态过滤候选。
结果中的 service id 与上门服务平台 t_service.id 对应。
"""

import json
from typing import Dict, List, Optional

from config import SERVICES_FILE, settings
from src.embedding import get_embedder


class ServiceRecommender:
    def __init__(self) -> None:
        data = json.loads(SERVICES_FILE.read_text(encoding="utf-8"))
        self.platform_name: str = data.get("platform", "宠物服务上门平台")
        self.base_path: str = data.get("base_path", "/user/service")
        self.services: List[Dict] = data["services"]
        self.base_url = settings.platform_base_url.rstrip("/")

        self._embedder = get_embedder()
        # 启动时把服务文案向量化一次，后续推荐直接复用
        self._vectors = [self._embedder.embed_query(self._doc_text(s)) for s in self.services]

    @staticmethod
    def _doc_text(service: Dict) -> str:
        """拼一段能代表该服务的检索文本。"""
        return " ".join(
            [
                service.get("name", ""),
                service.get("category", ""),
                service.get("description", ""),
                " ".join(service.get("tags", [])),
                " ".join(service.get("suitable_issues", [])),
            ]
        )

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        # 向量已做 L2 归一化，点积即余弦相似度
        return sum(x * y for x, y in zip(a, b))

    def recommend(
        self,
        question: str,
        profile: Optional[Dict] = None,
        top_n: Optional[int] = None,
        require_match: bool = False,
    ) -> List[Dict]:
        profile = profile or {}
        top_n = top_n or settings.recommend_top_n
        pet_type = profile.get("pet_type")
        query_vec = self._embedder.embed_query(question)

        candidates: List[Dict] = []
        for service, vec in zip(self.services, self._vectors):
            # 已下架的服务不推荐（与平台规则一致）
            if service.get("status", 1) != 1:
                continue
            # 品种不匹配直接跳过（猫的服务不会推给狗）
            types = service.get("pet_types") or []
            if pet_type and types and pet_type not in types:
                continue

            # ---- 规则分：关键词命中 ----
            hits = [kw for kw in service.get("keywords", []) if kw and kw in question]
            issue_hits = [i for i in service.get("suitable_issues", []) if i and i in question]
            rule_score = min(1.0, (len(hits) * 0.5 + len(issue_hits) * 0.3))
            if require_match and not (hits or issue_hits):
                continue

            # ---- 语义分 ----
            semantic_raw = max(0.0, self._cosine(query_vec, vec))

            candidates.append(
                {
                    "service": service,
                    "rule_score": rule_score,
                    "semantic_raw": semantic_raw,
                    "hits": hits + issue_hits,
                }
            )

        if not candidates:
            return []

        # 语义分做「组内归一化」：避免不同问题下阈值飘忽
        max_sem = max(c["semantic_raw"] for c in candidates) or 1.0
        for c in candidates:
            sem_norm = c["semantic_raw"] / max_sem
            c["score"] = round(0.5 * sem_norm + 0.5 * c["rule_score"], 4)

        # 过滤掉既不命中关键词、语义也平庸的候选
        candidates = [c for c in candidates if c["score"] >= 0.35 or c["hits"]]
        candidates.sort(key=lambda c: c["score"], reverse=True)

        results: List[Dict] = []
        for item in candidates[:top_n]:
            service = item["service"]
            results.append(
                {
                    "id": service["id"],
                    "name": service["name"],
                    "category": service.get("category", ""),
                    "price": service.get("price"),
                    "duration": service.get("duration"),
                    "description": service.get("description", ""),
                    "score": item["score"],
                    "reason": self._build_reason(service, item["hits"], pet_type),
                    "url": f"{self.base_url}{self.base_path}/{service['id']}",
                }
            )
        return results

    @staticmethod
    def _build_reason(service: Dict, hits: List[str], pet_type: Optional[str]) -> str:
        """生成可展示的推荐理由。"""
        reasons: List[str] = []
        if hits:
            reasons.append("匹配到「" + "、".join(dict.fromkeys(hits)) + "」")
        if pet_type and pet_type in (service.get("pet_types") or []):
            reasons.append(f"适用于{pet_type}")
        if not reasons:
            reasons.append("与你的描述语义最接近")
        reasons.append(f"约 {service.get('duration')} 分钟 / ¥{service.get('price')}")
        return "，".join(reasons)

    def list_all(self) -> List[Dict]:
        """给前端做服务清单展示。"""
        return [
            {
                "id": s["id"],
                "name": s["name"],
                "category": s.get("category", ""),
                "price": s.get("price"),
                "status": s.get("status", 1),
            }
            for s in self.services
        ]
