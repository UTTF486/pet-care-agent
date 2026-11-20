"""命令行重建知识库。

用法：
    python scripts/build_kb.py            # 全量重建
    python scripts/build_kb.py --check    # 只看当前状态，不重建
"""

import argparse
import sys
from pathlib import Path

# 让脚本能 import 到项目根目录下的包（直接在 scripts 目录下运行时需要）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.knowledge import build_knowledge_base, stats  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="构建宠物养护知识库")
    parser.add_argument("--check", action="store_true", help="只输出当前知识库状态")
    args = parser.parse_args()

    if args.check:
        info = stats()
        print(f"片段数: {info['chunks']}")
        print(f"向量维度: {info['dim']}（{info['provider']}）")
        print("文档:")
        for name in info["sources"]:
            print(f"  - {name}")
        return

    result = build_knowledge_base(rebuild=True)
    print("\n重建完成：")
    print(f"  片段数: {result['chunks']}")
    print(f"  文档数: {len(result['sources'])}")
    print(f"  向量维度: {result['dim']}（{result['provider']}）")


if __name__ == "__main__":
    main()
