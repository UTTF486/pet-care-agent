"""提示词集中管理。

把提示词单独抽出来，好处是调效果时不用翻业务代码，
面试时也能清楚说明「约束大模型不胡说」靠的是哪几条规则。
"""

import re
from typing import Dict, List

# 系统提示词：RAG 的核心约束都写在这里
SYSTEM_PROMPT = """你是一位有 8 年临床与养护经验的宠物顾问，擅长猫狗日常喂养、行为解读和常见疾病预防。

回答必须遵守以下规则：
1. 只依据【参考资料】作答，不得编造资料中没有的结论、药名或剂量。
2. 如果参考资料不足以回答，直接说明「资料里没有明确提到」，并给出安全的通用建议。
3. 涉及以下情况，必须明确提示尽快线下就医：持续呕吐/腹泻超过 24 小时、精神萎靡不吃不喝、
   呼吸困难、抽搐、误食异物或有毒物、幼宠（3 月龄内）出现任何异常。
4. 禁止给出具体用药剂量，只说明「需要由兽医根据体重开具」。
5. 用中文口语化表达，分点作答，控制在 250 字以内，先给结论再给依据。
6. 不要复述参考资料原文，用自己的话总结。

如果用户的问题带有预约/上门服务意图，可以在结尾用一句话自然引导，不要硬推销。"""


def build_answer_prompt(
    question: str,
    docs: List[Dict],
    profile: Dict[str, str],
    history: List[Dict[str, str]],
) -> str:
    """把检索结果、用户画像、近期对话拼成 user prompt。"""
    context_blocks = []
    for i, doc in enumerate(docs, start=1):
        context_blocks.append(
            f"[资料{i}] 来源：{doc.get('source', '未知')} / 章节：{doc.get('title', '')}\n{doc.get('content', '')}"
        )
    context = "\n\n".join(context_blocks) or "（无相关资料）"

    profile_text = "、".join(f"{k}={v}" for k, v in profile.items() if v) or "未提供"

    recent = history[-4:]
    history_text = "\n".join(f"{m['role']}：{m['content']}" for m in recent) or "（无）"

    return f"""【用户画像】{profile_text}

【近期对话】
{history_text}

【参考资料】
{context}

【当前问题】
{question}

请依据参考资料作答。"""


def compose_extractive_answer(question: str, docs: List[Dict], profile: Dict[str, str]) -> str:
    """离线兜底答案：从检索片段里抽取与问题最相关的句子。

    没有大模型时也能给出有依据的回答 —— 这是「无 Key 也能演示」的关键。
    思路是经典的抽取式摘要：按问题词命中数给句子打分，取前几条。
    """
    if not docs:
        return (
            "这个问题我暂时没有在知识库里找到可靠依据。\n"
            "建议你补充一下宠物的品种、年龄和具体症状，我再帮你看看。\n"
            "如果出现精神差、不吃不喝、持续呕吐腹泻，请尽快带去宠物医院。"
        )

    # 问题里的关键词（中文按 2-gram，简单有效）
    compact = re.sub(r"\s+", "", question)
    keywords = {compact[i:i + 2] for i in range(len(compact) - 1)} if len(compact) > 1 else {compact}

    # 按「文档相关性」优先取句：先取 top1 文档里最相关的两句，
    # 不够再补 top2。避免出现「问拉稀却回答呕吐」的跨主题串台。
    bullets: List[str] = []
    for doc in docs[:2]:
        candidates: List[tuple] = []
        for sentence in re.split(r"[。\n；]", doc.get("content", "")):
            sentence = sentence.strip(" -·\t")
            if len(sentence) < 8:
                continue
            hit = sum(1 for kw in keywords if kw and kw in sentence)
            if hit == 0:
                continue
            # 密度：命中数按句子长度归一，避免长句仅因为长而占优
            density = hit / max(1.0, len(sentence) / 20.0)
            candidates.append((density * (0.5 + float(doc.get("score", 0))), sentence))

        candidates.sort(key=lambda x: x[0], reverse=True)
        for _, sentence in candidates[:2]:
            bullets.append(f"- {sentence}（《{doc.get('title', '')}》）")
        if len(bullets) >= 4:
            break

    if not bullets:
        top = docs[0]
        bullets = [f"- {top.get('content', '')[:120]}…（《{top.get('title', '')}》）"]

    pet = profile.get("pet_type") or "宠物"
    head = f"根据知识库中关于{pet}养护的资料，整理如下："
    tail = (
        "\n\n以上内容仅供参考。如果症状持续或加重，请尽快联系宠物医生面诊；"
        "需要人手上门照护的话，我也可以帮你安排服务师。"
    )
    return head + "\n" + "\n".join(bullets) + tail


# 信息不足时的追问模板
FOLLOWUP_TEMPLATES = {
    "pet_type": "方便先告诉我是什么宠物吗？是猫咪还是狗狗，大概多大了？这样我能给更准确的建议。",
    "symptom": "能具体描述一下情况吗？比如：从什么时候开始的、有没有呕吐/腹泻/不吃东西、精神状态怎么样。",
}

# 闲聊与自我介绍
CHITCHAT_REPLY = """你好，我是宠物养护智能顾问 🐾

我可以帮你做三件事：
1. **养护问答**：喂养、洗护、疫苗驱虫、常见症状判断（答案都来自本地知识库，不瞎编）
2. **情况分析**：告诉我品种、年龄和症状，我帮你梳理可能原因和观察重点
3. **服务推荐**：需要人手上门时，直接匹配上门喂养/洗护/遛狗/陪护服务

你可以这样问我：
- 「三个月大的布偶猫一天喂多少克？」
- 「狗狗拉稀两天了，精神还可以，要紧吗？」
- 「下周出差五天，猫没人管怎么办？」"""
