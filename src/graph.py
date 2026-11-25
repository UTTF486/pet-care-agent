"""LangGraph 多轮对话工作流。

工作流：

                       ┌──────────┐
        START ────────▶│ analyze  │  意图识别 + 槽位抽取 + 追问判断
                       └────┬─────┘
          ┌─────────────────┼──────────────────┬─────────────┐
          │ 信息不足         │ 知识/症状类        │ 服务/预约类   │ 打招呼
          ▼                 ▼                  ▼             ▼
     ┌──────────┐     ┌──────────┐       ┌────────────┐  ┌──────────┐
     │ ask_more │     │ retrieve │       │ recommend  │  │ chitchat │
     └────┬─────┘     └────┬─────┘       └─────┬──────┘  └────┬─────┘
          │                ▼                    │              │
          │          ┌──────────┐               │              │
          │          │ generate │  RAG 生成      │              │
          │          └────┬─────┘               │              │
          │               ▼                     │              │
          │         ┌────────────┐              │              │
          └────────▶│ recommend  │◀─────────────┘              │
                    └─────┬──────┘                             │
                          ▼                                    │
                     ┌─────────┐                               │
                     │ finish  │◀──────────────────────────────┘
                     └────┬────┘
                          ▼
                         END

要点：
- 条件边（conditional edges）让「答什么」由意图决定，而不是一股脑调模型
- 知识类问题走完 generate 后仍会进 recommend，实现「咨询 → 转化」
- 所有路径最终都经过 finish，统一做上下文记忆写入，避免遗漏
"""

import logging
import re
from typing import Dict, List, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from config import settings
from src.knowledge import ensure_knowledge_base, search
from src.llm import get_llm
from src.prompts import (
    CHITCHAT_REPLY,
    FOLLOWUP_TEMPLATES,
    SYSTEM_PROMPT,
    build_answer_prompt,
    compose_extractive_answer,
)
from src.recommender import ServiceRecommender

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    """在节点之间流转的状态。total=False 表示字段可缺省。"""

    question: str                  # 本轮用户问题
    history: List[Dict[str, str]]  # 多轮对话历史（跨轮持久化）
    profile: Dict[str, str]        # 用户画像：pet_type / age / 症状关键词
    intent: str                    # knowledge / service / chitchat / ask_more
    pending_intent: str            # 追问前保留的业务意图，下一轮补齐槽位后恢复
    need_slots: List[str]          # 缺失的必要信息
    docs: List[Dict]               # 检索到的知识片段
    answer: str                    # 最终回答
    recommendations: List[Dict]    # 推荐的服务
    provider: str                  # 本轮使用的模型
    trace: List[str]               # 执行轨迹，界面可展示


# ---------------- 意图识别规则 ----------------

CHITCHAT_WORDS = ("你好", "您好", "谢谢", "再见", "你是谁", "介绍一下你", "在吗", "hi", "hello", "嗨")

SERVICE_WORDS = (
    "预约", "上门", "下单", "价格", "多少钱", "收费", "怎么买", "找人", "安排人",
    "服务费", "推荐个服务", "有没有人", "帮我约", "出差", "没人管", "没人照顾",
    "不在家", "托管", "寄养", "帮忙喂", "帮我遛",
)

KNOWLEDGE_WORDS = (
    "为什么", "怎么回事", "怎么办", "正常吗", "能不能", "可以吃", "能吃", "喂多少",
    "症状", "治疗", "预防", "疫苗", "驱虫", "洗澡", "洗护", "喂养", "多久", "掉毛",
    "呕吐", "拉稀", "腹泻", "不吃", "没精神", "发烧", "咳嗽", "打喷嚏", "眼屎",
    "便便", "尿", "体重", "怀孕", "绝育", "应激", "驱虫药", "猫粮", "狗粮",
)

# 症状词，用于画像与推荐
ISSUE_WORDS = (
    "呕吐", "拉稀", "腹泻", "软便", "便秘", "不吃", "食欲差", "没精神", "发烧",
    "咳嗽", "打喷嚏", "眼屎", "掉毛", "皮肤", "瘙痒", "应激", "术后", "老年",
    "拆家", "精力旺盛", "肥胖", "挑食",
)

BREED_CAT = ("布偶", "英短", "美短", "折耳", "缅因", "暹罗", "橘猫", "狸花", "蓝猫", "加菲", "金渐层")
BREED_DOG = ("柯基", "柴犬", "金毛", "拉布拉多", "哈士奇", "萨摩耶", "边牧", "泰迪", "比熊",
             "德牧", "比格", "博美", "阿拉斯加", "吉娃娃", "雪纳瑞")


def _contains_any(text: str, words: tuple) -> Optional[str]:
    """返回第一个命中的关键词，便于在轨迹里展示「为什么判成这个意图」。"""
    for word in words:
        if word in text:
            return word
    return None


def _extract_profile(question: str, base: Optional[Dict[str, str]]) -> Dict[str, str]:
    """从这句话里抽取宠物画像，并与历史画像合并（后者不覆盖前者）。"""
    profile = dict(base or {})

    if "猫" in question:
        profile["pet_type"] = "猫"
    elif "狗" in question or "犬" in question:
        profile["pet_type"] = "狗"

    # 用户只说品种不说"猫/狗"时，靠品种反推物种（"我家是柯基" -> 狗）
    cat_breed = _contains_any(question, BREED_CAT)
    dog_breed = _contains_any(question, BREED_DOG)
    if cat_breed:
        profile["pet_type"] = "猫"
        profile["breed"] = cat_breed
    elif dog_breed:
        profile["pet_type"] = "狗"
        profile["breed"] = dog_breed

    # 年龄：3个月 / 2岁
    month = re.search(r"(\d{1,2})\s*(个月|月龄|月大)", question)
    year = re.search(r"(\d{1,2})\s*(岁|年)", question)
    if month:
        profile["age"] = f"{month.group(1)}个月"
    elif year:
        profile["age"] = f"{year.group(1)}岁"

    breed = _contains_any(question, BREED_CAT + BREED_DOG)
    if breed:
        profile["breed"] = breed

    issues = [w for w in ISSUE_WORDS if w in question]
    if issues:
        # 最多保留最近提到的 3 个症状，避免画像无限膨胀
        previous = [i for i in profile.get("issues", "").split(",") if i]
        merged = list(dict.fromkeys(issues + previous))
        profile["issues"] = ",".join([i for i in merged if i][:3])

    return profile


def _contextual_query(
    question: str,
    profile: Dict[str, str],
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """把当前问题和已确认画像拼成检索语句，修复“只回答一个猫字”时的召回失败。"""
    parts = [question.strip()]
    has_new_intent = any(
        word in question
        for word in KNOWLEDGE_WORDS + SERVICE_WORDS + CHITCHAT_WORDS
    )
    if history and len(re.sub(r"\s", "", question)) <= 8 and not has_new_intent:
        previous = next(
            (
                (item.get("content") or "").strip()
                for item in reversed(history)
                if item.get("role") == "user" and (item.get("content") or "").strip() != question
            ),
            "",
        )
        if previous:
            parts.append(previous)
    for key in ("pet_type", "breed", "age", "issues"):
        value = (profile.get(key) or "").strip()
        if value and value not in question:
            parts.append(value.replace(",", " "))
    return " ".join(part for part in parts if part)


class PetConsultAgent:
    """对外的唯一入口：封装知识库、模型、推荐器和编译后的图。"""

    def __init__(self, build_kb: bool = True) -> None:
        if build_kb:
            info = ensure_knowledge_base(verbose=True)
            logger.info("知识库就绪: %s", info)
        self.llm = get_llm()
        self.recommender = ServiceRecommender()
        self.graph = self._build_graph()

    # ---------------- 构图 ----------------

    def _build_graph(self):
        builder = StateGraph(AgentState)

        builder.add_node("analyze", self._analyze)
        builder.add_node("ask_more", self._ask_more)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("generate", self._generate)
        builder.add_node("recommend", self._recommend)
        builder.add_node("chitchat", self._chitchat)
        builder.add_node("finish", self._finish)

        builder.add_edge(START, "analyze")
        builder.add_conditional_edges(
            "analyze",
            self._route,
            {
                "ask_more": "ask_more",
                "retrieve": "retrieve",
                "recommend": "recommend",
                "chitchat": "chitchat",
            },
        )
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", "recommend")
        builder.add_edge("recommend", "finish")
        builder.add_edge("chitchat", "finish")
        builder.add_edge("ask_more", "finish")
        builder.add_edge("finish", END)

        # MemorySaver：按 thread_id 保存状态，实现跨轮记忆（进程内存储，重启即清空）
        return builder.compile(checkpointer=MemorySaver())

    # ---------------- 各节点 ----------------

    def _analyze(self, state: AgentState) -> Dict:
        question = (state.get("question") or "").strip()
        history = state.get("history") or []
        profile = _extract_profile(question, state.get("profile"))
        pending_intent = state.get("pending_intent") or ""

        trace: List[str] = []

        # 1. 意图判定：规则优先（快、可解释、不烧 token）
        if _contains_any(question, CHITCHAT_WORDS) and len(question) <= 20:
            intent, hit = "chitchat", _contains_any(question, CHITCHAT_WORDS)
        elif _contains_any(question, SERVICE_WORDS):
            intent, hit = "service", _contains_any(question, SERVICE_WORDS)
        elif _contains_any(question, KNOWLEDGE_WORDS):
            intent, hit = "knowledge", _contains_any(question, KNOWLEDGE_WORDS)
        else:
            # 兜底当作知识问答（比乱推服务更安全）
            intent, hit = "knowledge", "默认"

        # 上一轮因缺槽位而追问时，短回答（如“猫”）应延续原业务意图。
        if hit == "默认" and pending_intent in ("knowledge", "service"):
            intent = pending_intent
            trace.append(f"延续待补全意图：{pending_intent}")

        trace.append(f"意图识别：{intent}（命中关键词「{hit}」）")

        # 2. 槽位检查：信息不足就先追问，不要硬答
        #    猫和狗的喂养/症状判断差异很大，不知道物种就回答等于瞎猜。
        #    注意只对知识问答和服务咨询做检查，闲聊不需要槽位。
        need_slots: List[str] = []
        if intent in ("knowledge", "service"):
            if not profile.get("pet_type"):
                need_slots.append("pet_type")
            elif (intent == "knowledge"
                  and not profile.get("issues")
                  and len(re.sub(r"\s", "", question)) <= 6):
                need_slots.append("symptom")

        if need_slots:
            trace.append(f"信息不足，需要补充：{need_slots}")

        trace.append(f"用户画像：{profile or '暂无'}")

        return {
            "question": question,
            "history": history,
            "profile": profile,
            "intent": "ask_more" if need_slots else intent,
            "pending_intent": intent if need_slots else "",
            "need_slots": need_slots,
            # 每轮重置上一轮的中间结果，避免旧答案串到新问题
            "docs": [],
            "answer": "",
            "recommendations": [],
            "provider": self.llm.provider,
            "trace": trace,
        }

    def _route(self, state: AgentState) -> str:
        intent = state.get("intent")
        if intent == "ask_more":
            return "ask_more"
        if intent == "service":
            return "recommend"
        if intent == "chitchat":
            return "chitchat"
        return "retrieve"

    def _ask_more(self, state: AgentState) -> Dict:
        slots = state.get("need_slots") or []
        # 一次只追问一个，避免连环拷问
        slot = slots[0] if slots else "symptom"
        followup = FOLLOWUP_TEMPLATES.get(slot, FOLLOWUP_TEMPLATES["symptom"])
        return {
            "answer": followup,
            "trace": (state.get("trace") or []) + [f"追问：{slot}"],
        }

    def _retrieve(self, state: AgentState) -> Dict:
        question = state.get("question", "")
        profile = state.get("profile") or {}
        query = _contextual_query(question, profile, state.get("history") or [])
        docs = search(query, top_k=settings.top_k)
        strong = [d for d in docs if d["score"] >= settings.min_similarity]
        best = docs[0]["score"] if docs else 0.0
        detail = ""
        if docs:
            top = docs[0]
            detail = (f"（top1 覆盖率 {top.get('coverage', 0):.2f} / "
                      f"余弦 {top.get('cosine', 0):.2f}）")
        trace = (state.get("trace") or []) + [
            f"检索问题：{query}",
            f"混合检索：召回 {len(docs)} 条，{len(strong)} 条达阈值 {settings.min_similarity}"
            f"，最高匹配度 {best:.2f}{detail}"
        ]
        return {"docs": strong or docs[:1], "trace": trace}

    def _generate(self, state: AgentState) -> Dict:
        question = state.get("question", "")
        docs = state.get("docs") or []
        profile = state.get("profile") or {}
        history = state.get("history") or []

        best = max((d["score"] for d in docs), default=0.0)
        # 检索太弱就别让模型硬编，直接走兜底话术
        if not docs or best < settings.min_similarity:
            answer = (
                "这个问题我在现有知识库里没有找到足够可靠的依据，先不瞎猜。\n"
                "你可以补充一下宠物的品种、年龄和具体表现，我再帮你分析；\n"
                "如果是精神差、不吃不喝、持续呕吐腹泻，建议直接去宠物医院。"
            )
            trace = (state.get("trace") or []) + ["知识库置信度不足，已触发兜底话术"]
            return {"answer": answer, "trace": trace}

        if self.llm.available:
            answer = self.llm.generate(
                SYSTEM_PROMPT,
                build_answer_prompt(question, docs, profile, history),
            )
            if answer:
                trace = (state.get("trace") or []) + [f"大模型生成：{self.llm.model_name}"]
                return {"answer": answer, "trace": trace}
            note = "大模型调用失败，已降级为抽取式回答"
        else:
            note = "离线模式，使用抽取式回答"

        answer = compose_extractive_answer(question, docs, profile)
        return {"answer": answer, "trace": (state.get("trace") or []) + [note]}

    def _recommend(self, state: AgentState) -> Dict:
        question = state.get("question", "")
        profile = state.get("profile") or {}
        query = _contextual_query(question, profile, state.get("history") or [])
        answer = state.get("answer", "")
        # 健康问答的附加推荐必须命中明确规则，避免纯语义分把“遛狗”推给腹泻宠物。
        recs = self.recommender.recommend(
            query,
            profile,
            require_match=state.get("intent") != "service",
        )

        # 服务意图不经过 generate，在推荐节点补齐引导文案。
        if not answer:
            if recs:
                names = "、".join(f"{r['name']}（¥{r['price']}，约 {r['duration']} 分钟）" for r in recs)
                answer = (
                    f"看你的情况，人手上门会更省心。我帮你匹配到：\n\n{names}\n\n"
                    "点下方卡片里的「去预约」就能选时间下单，服务师上门会全程拍照反馈。"
                )
            else:
                answer = (
                    "我可以帮你安排上门服务。方便说一下是什么宠物、具体需要什么帮助吗？\n"
                    "常见的有：上门喂养、上门洗护、遛狗、术后或老年陪护。"
                )

        trace = (state.get("trace") or []) + [f"服务推荐：{len(recs)} 个候选"]
        return {"recommendations": recs, "answer": answer, "trace": trace}

    def _chitchat(self, state: AgentState) -> Dict:
        return {
            "answer": CHITCHAT_REPLY,
            "trace": (state.get("trace") or []) + ["闲聊分支，直接返回能力介绍"],
        }

    def _finish(self, state: AgentState) -> Dict:
        """统一收口：写入多轮记忆并按轮数裁剪。"""
        history = list(state.get("history") or [])
        question = state.get("question", "")
        answer = state.get("answer", "")
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})

        # 只保留最近 N 轮，控制上下文长度（也控制 token 成本）
        limit = settings.max_history_turns * 2
        if len(history) > limit:
            history = history[-limit:]

        return {"history": history}

    # ---------------- 对外接口 ----------------

    def chat(self, question: str, thread_id: str = "default") -> Dict:
        """执行一轮对话，返回结构化结果供界面渲染。"""
        question = (question or "").strip()
        if not question:
            raise ValueError("问题不能为空")
        if len(question) > 2000:
            raise ValueError("问题过长，请精简到 2000 字以内")
        state = self.graph.invoke(
            {"question": question},
            config={"configurable": {"thread_id": thread_id}},
        )
        return {
            "answer": state.get("answer", ""),
            "intent": state.get("intent", ""),
            "profile": state.get("profile", {}),
            "docs": state.get("docs", []),
            "recommendations": state.get("recommendations", []),
            "provider": state.get("provider", ""),
            "trace": state.get("trace", []),
            "history": state.get("history", []),
        }

    def get_history(self, thread_id: str = "default") -> List[Dict[str, str]]:
        snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        return (snapshot.values or {}).get("history", []) if snapshot else []


# 供脚本 / Streamlit 复用
_agent: Optional[PetConsultAgent] = None


def get_agent() -> PetConsultAgent:
    global _agent
    if _agent is None:
        _agent = PetConsultAgent()
    return _agent


def reset_agent() -> None:
    """清空 Agent 单例，配合 Streamlit 的参数重建与测试使用。"""
    global _agent
    _agent = None
