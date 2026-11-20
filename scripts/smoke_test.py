"""一键自检：不启动网页，验证整条链路是否正常。

用法：
    python scripts/smoke_test.py

覆盖 4 类场景：知识问答、信息不足追问、服务推荐、闲聊。
适合改完代码或换模型后快速回归。
"""

import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from src.graph import get_agent  # noqa: E402

CASES = [
    ("知识问答", "三个月大的布偶猫一天喂多少克猫粮？"),
    ("症状咨询", "狗狗拉稀两天了，精神还可以，要紧吗？"),
    ("信息不足", "它一直吐，还不吃饭"),
    ("服务推荐", "下周出差五天，猫没人管怎么办？"),
    ("闲聊", "你好"),
]


def main() -> None:
    print("=" * 70)
    print(f"自检开始 ｜ 模型：{settings.llm_display_name} ｜ 向量：{settings.embedding_provider}")
    print("=" * 70)

    agent = get_agent()
    failed = 0

    # 每个用例独立会话，避免上一轮的宠物画像影响下一轮判断
    for name, question in CASES:
        result = agent.chat(question, thread_id=uuid4().hex)
        answer = (result.get("answer") or "").strip()
        recs = result.get("recommendations") or []
        docs = result.get("docs") or []

        print(f"\n【{name}】{question}")
        print(f"  意图: {result.get('intent')} ｜ 命中片段: {len(docs)} ｜ 推荐: {len(recs)}")
        if docs:
            print(f"  最高匹配度: {docs[0]['score']:.3f} 《{docs[0]['title']}》")
        print(f"  回答: {answer[:120]}{'…' if len(answer) > 120 else ''}")
        for rec in recs:
            print(f"  → 推荐 {rec['name']} ¥{rec['price']} ｜ {rec['reason']}")

        if not answer:
            failed += 1
            print("  ✗ 回答为空")

    # 多轮记忆验证：同一会话里先说品种，再问症状，看画像是否被沿用
    print("\n【多轮记忆】同一会话连续提问")
    thread = uuid4().hex
    agent.chat("我家是两岁的柯基", thread_id=thread)
    follow = agent.chat("它最近老是掉毛，怎么办", thread_id=thread)
    print(f"  第二轮识别出的画像: {follow.get('profile')}")
    if follow.get("profile", {}).get("pet_type") == "狗":
        print("  ✓ 成功沿用上一轮的宠物品种")
    else:
        failed += 1
        print("  ✗ 未沿用上下文")

    print("\n" + "=" * 70)
    print("自检结果：" + ("全部通过 ✓" if failed == 0 else f"{failed} 个用例异常 ✗"))
    print("=" * 70)


if __name__ == "__main__":
    main()
