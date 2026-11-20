"""终端调试入口（不启动网页也能验证整条链路）。

用法：
    python scripts/chat_cli.py

适合排查「到底是检索没召回，还是模型没按资料答」这类问题。
"""

import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from src.graph import get_agent  # noqa: E402


def main() -> None:
    print("=" * 60)
    print("宠物养护智能咨询 Agent（终端调试模式）")
    print(f"模型: {settings.llm_display_name}")
    print(f"向量: {settings.embedding_provider} / top_k={settings.top_k}")
    print("输入 exit 退出，输入 new 新建会话")
    print("=" * 60)

    agent = get_agent()
    thread_id = uuid4().hex

    while True:
        try:
            question = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("再见")
            break
        if question.lower() == "new":
            thread_id = uuid4().hex
            print("[已新建会话，上下文已清空]")
            continue

        result = agent.chat(question, thread_id=thread_id)

        print(f"\n助手: {result['answer']}")
        print(f"\n[意图] {result['intent']}    [画像] {result['profile']}")

        if result["recommendations"]:
            print("[推荐服务]")
            for rec in result["recommendations"]:
                print(f"  · {rec['name']} ¥{rec['price']} — {rec['reason']}")
                print(f"    {rec['url']}")

        if settings.verbose:
            print("[检索片段]")
            for doc in result["docs"]:
                print(f"  · {doc['score']:.3f} 《{doc['title']}》")
            print("[执行轨迹]")
            for step in result["trace"]:
                print(f"  → {step}")


if __name__ == "__main__":
    main()
