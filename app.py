"""Streamlit 前端。

界面分三块：
- 左侧边栏：运行状态、参数调试、知识库管理
- 主区域：多轮对话
- 每条回答下方：推荐服务 / 参考资料 / 工作流轨迹（可折叠，演示时很有说服力）

启动：streamlit run app.py
"""

from uuid import uuid4

import streamlit as st

import src.graph as graph_module
import src.llm as llm_module
from config import settings
from src.graph import get_agent
from src.knowledge import add_uploaded_text, build_knowledge_base, stats

st.set_page_config(page_title="宠物养护智能咨询 Agent", page_icon="🐾", layout="wide")


@st.cache_resource(show_spinner="正在加载知识库与工作流…")
def load_agent():
    """缓存 Agent 实例：Streamlit 每次交互都会重跑脚本，不缓存会反复建库。"""
    return get_agent()


def init_session() -> None:
    if "thread_id" not in st.session_state:
        # 用 uuid 做 thread_id，LangGraph 按它隔离上下文
        st.session_state.thread_id = uuid4().hex
    if "messages" not in st.session_state:
        st.session_state.messages = []


def reset_conversation() -> None:
    st.session_state.thread_id = uuid4().hex
    st.session_state.messages = []


# ------------------------- 侧边栏 -------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 🐾 运行状态")
        st.caption(f"模型：{settings.llm_display_name}")
        if not settings.llm_available:
            st.info(
                "未配置 API Key，当前为**离线兜底模式**：\n"
                "检索 + 抽取式回答，流程完整可演示。\n"
                "在 `.env` 里填 `DASHSCOPE_API_KEY` 即切换为大模型生成。",
                icon="ℹ️",
            )

        info = stats()
        col1, col2 = st.columns(2)
        col1.metric("知识片段", info["chunks"])
        col2.metric("文档数", len(info["sources"]))
        st.caption(f"向量维度 {info['dim']} · {info['provider']}")

        st.divider()
        st.markdown("### ⚙️ 参数调试")
        # 直接改单例 settings，下一次调用立即生效（无需重启）
        settings.top_k = st.slider("检索条数 top_k", 1, 10, settings.top_k,
                                   help="越大召回越全，但噪声也越多")
        settings.min_similarity = st.slider("相似度阈值", 0.0, 0.6, settings.min_similarity, 0.01,
                                            help="低于该值视为知识库未覆盖，触发兜底话术")
        settings.recommend_top_n = st.slider("推荐服务数量", 1, 4, settings.recommend_top_n)
        new_temp = st.slider("模型温度", 0.0, 1.0, settings.temperature, 0.05,
                             help="修改后需点下面的按钮重建模型客户端")
        if st.button("应用温度并重建客户端", use_container_width=True):
            settings.temperature = new_temp
            llm_module._llm_client = None   # 清空单例
            graph_module._agent = None
            load_agent.clear()
            st.success("已重建，下一轮对话生效")

        st.divider()
        st.markdown("### 📚 知识库管理")
        uploaded = st.file_uploader("上传 .md / .txt 扩充知识库", type=["md", "txt"])
        if uploaded is not None and st.button("加入知识库", use_container_width=True):
            text = uploaded.read().decode("utf-8", errors="ignore")
            added = add_uploaded_text(uploaded.name, text)
            st.success(f"《{uploaded.name}》已切分为 {added} 个片段并入库")

        if st.button("重建知识库（全量）", use_container_width=True):
            build_knowledge_base(rebuild=True)
            load_agent.clear()
            st.success("已重建")

        with st.expander("当前知识库文件"):
            for name in stats()["sources"] or ["（暂无）"]:
                st.markdown(f"- {name}")

        st.divider()
        if st.button("🆕 新建对话", use_container_width=True):
            reset_conversation()
            st.rerun()


# ------------------------- 消息渲染 -------------------------
def render_assistant(msg: dict) -> None:
    st.markdown(msg.get("answer", ""))

    recs = msg.get("recommendations") or []
    if recs:
        st.markdown("**🎯 推荐服务**")
        cols = st.columns(len(recs))
        for col, rec in zip(cols, recs):
            with col.container(border=True):
                st.markdown(f"**{rec['name']}**")
                st.caption(rec["reason"])
                st.markdown(f"¥{rec['price']} · 约 {rec['duration']} 分钟")
                st.link_button("去预约", rec["url"], use_container_width=True)

    docs = msg.get("docs") or []
    if docs:
        with st.expander(f"📚 参考资料（{len(docs)} 条，均为本地知识库命中）"):
            for doc in docs:
                st.markdown(f"**{doc['title']}** · 相似度 `{doc['score']:.3f}` · `{doc['source']}`")
                content = doc.get("content", "")
                st.caption(content[:300] + ("…" if len(content) > 300 else ""))
                st.divider()

    trace = msg.get("trace") or []
    if trace:
        with st.expander("🧭 工作流执行轨迹（LangGraph 节点）"):
            for step in trace:
                st.markdown(f"- {step}")


def render_history() -> None:
    if not st.session_state.messages:
        st.info(
            "试试这些问题：\n\n"
            "- 三个月大的布偶猫一天喂多少克猫粮？\n"
            "- 狗狗拉稀两天了但精神还行，需要立刻去医院吗？\n"
            "- 下周出差五天，猫没人照顾怎么办？（会推荐上门喂养服务）",
            icon="💡",
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🐾" if msg["role"] == "assistant" else "🧑"):
            if msg["role"] == "assistant":
                render_assistant(msg)
            else:
                st.markdown(msg["content"])


# ------------------------- 主流程 -------------------------
def main() -> None:
    init_session()
    render_sidebar()

    st.title("🐾 宠物养护智能咨询 Agent")
    st.caption(
        "RAG 知识库问答 · LangGraph 多轮工作流 · 咨询到上门服务的转化 ｜ "
        f"当前模型：{settings.llm_display_name}"
    )

    agent = load_agent()
    render_history()

    question = st.chat_input("描述你的问题，例如：我家柯基 2 岁，最近总是掉毛…")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🐾"):
        with st.spinner("正在检索知识库并分析…"):
            try:
                result = agent.chat(question, thread_id=st.session_state.thread_id)
            except Exception as exc:  # 兜底，避免一次异常把整个页面刷白
                st.error(f"处理失败：{exc}")
                return
        msg = {"role": "assistant", **result}
        render_assistant(msg)

    st.session_state.messages.append(msg)


if __name__ == "__main__":
    main()
