"""集中管理配置。

设计原则：所有可调参数都能用环境变量覆盖，且**默认值保证项目开箱即跑**。
即使没有大模型 API Key，项目也能以「离线兜底模式」完整演示流程。
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（本文件所在目录）
BASE_DIR = Path(__file__).resolve().parent

# 加载 .env（不存在也不报错，方便新手直接跑）
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
SERVICES_FILE = DATA_DIR / "services.json"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "pet_knowledge"


def _env_bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    """运行期配置。字段含义见 .env.example。"""

    # ---------------- 大模型 ----------------
    # auto: 配了谁用谁，都没配则离线兜底
    llm_provider: str = os.getenv("LLM_PROVIDER", "auto").lower()

    # DeepSeek（推荐，便宜、国内直连、中文好）
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    dashscope_model: str = os.getenv("DASHSCOPE_MODEL", "qwen-plus")
    ark_api_key: str = os.getenv("ARK_API_KEY", "")
    # 火山方舟的模型名一般填「推理接入点 ID」(ep-xxxx) 或模型名
    ark_model: str = os.getenv("ARK_MODEL", "doubao-pro-32k")

    # 通用 OpenAI 兼容端点：魔搭、硅基流动、各类中转站都能接
    # 只要平台提供 /v1/chat/completions，填这三个值即可，无需改代码
    compat_api_key: str = os.getenv("LLM_API_KEY", "")
    compat_base_url: str = os.getenv("LLM_BASE_URL", "")
    compat_model: str = os.getenv("LLM_MODEL", "")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "800"))

    # ---------------- 向量化 ----------------
    # local: 内置字符 n-gram 哈希向量（零依赖、离线可用）
    # dashscope: 调用 text-embedding-v3（需要 Key，效果更好）
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local").lower()
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "768"))

    # ---------------- 检索 ----------------
    top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "4"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "380"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "60"))
    # 低于该值视为「知识库没覆盖」，触发兜底话术，减少瞎编。
    # 注意这是「0.65*词项覆盖率 + 0.35*余弦」的混合分，不是纯余弦，
    # 所以标尺和常见向量库的 0.7/0.8 完全不同：相关片段通常在 0.15-0.45，
    # 无关片段在 0.03-0.12。调高更保守（更容易说不知道），调低更激进。
    min_similarity: float = float(os.getenv("MIN_SIMILARITY", "0.12"))

    # ---------------- 业务联动 ----------------
    # 宠物服务上门平台的前端地址，推荐结果里给出可点击的预约链接
    platform_base_url: str = os.getenv("PLATFORM_BASE_URL", "http://localhost:5173")
    recommend_top_n: int = int(os.getenv("RECOMMEND_TOP_N", "2"))

    # ---------------- 其他 ----------------
    verbose: bool = _env_bool("VERBOSE", False)
    max_history_turns: int = int(os.getenv("MAX_HISTORY_TURNS", "6"))

    # 派生字段：实际生效的提供方
    effective_provider: str = field(init=False, default="")

    def __post_init__(self) -> None:
        self.effective_provider = self._resolve_provider()

    def _resolve_provider(self) -> str:
        """把 auto 解析成具体实现。"""
        if self.llm_provider != "auto":
            return self.llm_provider
        if self.deepseek_api_key:
            return "deepseek"
        if self.dashscope_api_key:
            return "dashscope"
        if self.ark_api_key:
            return "ark"
        if self.compat_api_key and self.compat_base_url:
            return "compat"
        return "mock"

    @property
    def llm_available(self) -> bool:
        """是否有真实大模型可用。"""
        return self.effective_provider in ("deepseek", "dashscope", "ark", "compat")

    @property
    def llm_display_name(self) -> str:
        """给界面展示用的模型名。"""
        if self.effective_provider == "deepseek":
            return f"DeepSeek / {self.deepseek_model}"
        if self.effective_provider == "dashscope":
            return f"通义千问 / {self.dashscope_model}"
        if self.effective_provider == "ark":
            return f"豆包 / {self.ark_model}"
        if self.effective_provider == "compat":
            host = self.compat_base_url.replace("https://", "").replace("http://", "").split("/")[0]
            return f"{self.compat_model} @ {host}"
        return "离线兜底模式（未配置 API Key）"


settings = Settings()
