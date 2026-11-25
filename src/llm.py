"""大模型客户端。

DeepSeek、通义千问和豆包均通过 OpenAI 兼容接口接入。未配置密钥时，
客户端标记为不可用，由上层切换到抽取式回答。
"""

import logging
from typing import Optional

from langchain_openai import ChatOpenAI

from config import settings

logger = logging.getLogger(__name__)

# 国内几家的 OpenAI 兼容端点
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


class LLMClient:
    """薄封装：只暴露 generate(system, user) 一个方法，便于替换实现。"""

    def __init__(self) -> None:
        self.provider = settings.effective_provider
        self.model_name = settings.llm_display_name
        self._client: Optional[ChatOpenAI] = None
        self._init_error: Optional[str] = None

        if self.provider == "deepseek":
            self._client = self._build(settings.deepseek_api_key, settings.deepseek_model, DEEPSEEK_BASE_URL)
        elif self.provider == "dashscope":
            self._client = self._build(settings.dashscope_api_key, settings.dashscope_model, DASHSCOPE_BASE_URL)
        elif self.provider == "ark":
            self._client = self._build(settings.ark_api_key, settings.ark_model, ARK_BASE_URL)
        elif self.provider == "compat":
            # 任意 OpenAI 兼容端点（魔搭 / 硅基流动 / 中转站…）
            self._client = self._build(settings.compat_api_key, settings.compat_model, settings.compat_base_url)

    def _build(self, api_key: str, model: str, base_url: str) -> Optional[ChatOpenAI]:
        if not api_key:
            self._init_error = "未配置 API Key"
            return None
        try:
            return ChatOpenAI(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                timeout=45,
                max_retries=1,
            )
        except Exception as exc:  # 配置写错时不要让整个应用起不来
            self._init_error = str(exc)
            logger.warning("初始化大模型失败: %s", exc)
            return None

    @property
    def available(self) -> bool:
        return self._client is not None

    def generate(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """返回模型文本；失败返回 None，由上层降级为模板答案。"""
        if not self._client:
            return None
        try:
            response = self._client.invoke(
                [
                    ("system", system_prompt),
                    ("human", user_prompt),
                ]
            )
            content = response.content
            if isinstance(content, list):  # 部分模型返回分块结构
                content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
            return (content or "").strip() or None
        except Exception as exc:
            logger.warning("调用大模型失败，降级为模板答案: %s", exc)
            return None


# 全局单例：避免每次对话都重建客户端
_llm_client: Optional[LLMClient] = None


def get_llm() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def reset_llm() -> None:
    """清空单例，让温度等运行期配置在下一次调用时生效。"""
    global _llm_client
    _llm_client = None
