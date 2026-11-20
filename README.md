# 宠物养护智能咨询 Agent

面向宠物养护垂直领域的 **RAG + LangGraph** 智能体，与「智慧宠物服务上门平台」业务打通：
用户咨询养护 / 症状问题时给出有据可依的回答，并顺势推荐可下单的上门服务，把「咨询」变成「转化」。

不是孤立的 AI Demo：服务推荐里的 `service_id` 直接对应平台 `t_service` 表，点击即可跳转预约页。

---

## 一、整体架构

```
                      ┌──────────────────────────────────────┐
                      │        Streamlit 前端（app.py）        │
                      │  对话展示 / 参数调试 / 知识库上传        │
                      └───────────────┬──────────────────────┘
                                      │ chat(question, thread_id)
┌─────────────────────────────────────▼─────────────────────────────────────┐
│                        LangGraph 对话工作流（src/graph.py）                  │
│                                                                           │
│   analyze（意图识别 + 槽位抽取 + 画像更新）                                  │
│      │                                                                    │
│      ├── 信息不足 ──▶ ask_more（追问，一次只问一个）                          │
│      ├── 知识/症状 ─▶ retrieve ──▶ generate（RAG 生成）──┐                   │
│      ├── 服务/预约 ─────────────────────────────────────┤                   │
│      └── 打招呼 ───▶ chitchat                            ▼                   │
│                                                    recommend（服务推荐）     │
│                                                          │                  │
│                                                    finish（写多轮记忆）      │
└───────────┬───────────────────────────┬──────────────────┬────────────────┘
            │                           │                  │
   ┌────────▼────────┐        ┌─────────▼────────┐  ┌──────▼───────┐
   │ Chroma 向量库    │        │ 大模型（可选）      │  │ services.json │
   │ 本地持久化       │        │ 通义 / 豆包        │  │ 与平台服务对齐 │
   │ 6 篇养护文档     │        │ OpenAI 兼容协议    │  │ 8 个服务项    │
   └─────────────────┘        └──────────────────┘  └──────────────┘
```

**一次问答的完整链路**

1. `analyze`：正则 + 关键词判定意图（knowledge / service / chitchat），抽取宠物品种、年龄、症状，与历史画像合并
2. 信息不足（如没说是猫还是狗）→ `ask_more` 追问，不硬答
3. `retrieve`：问题向量化 → Chroma 召回 4 倍候选 → **IDF 加权词项覆盖率 + 余弦混合重排** → 低于阈值判为「知识库未覆盖」
4. `generate`：把「用户画像 + 近期对话 + 检索片段」拼成 prompt；有 API Key 走大模型，没有则用抽取式摘要兜底
5. `recommend`：语义相似度 + 关键词规则融合打分，过滤已下架服务与不匹配品种
6. `finish`：把本轮问答写入 `history`，按轮数裁剪，配合 `MemorySaver` 实现跨轮记忆

---

## 二、技术选型说明（为什么是这几个）

| 选择 | 理由 | 备选与放弃原因 |
| --- | --- | --- |
| **LangChain** | 只用到文本切分、ChatModel 封装，不滥用 Chain | 手写 HTTP 调用也可以，但切分器与提示词模板省事 |
| **LangGraph** | 对话是**有状态的分支流程**，不是单轮问答。条件边让「问什么答什么」可控，比一个巨型 prompt 稳定 | 纯 LangChain Agent（ReAct）意图容易跑偏，调试困难 |
| **Chroma** | 嵌入式向量库，`pip install` 即用，持久化到本地目录 | Milvus 要起服务、FAISS 不带元数据，都更重 |
| **OpenAI 兼容协议** | DeepSeek、通义、豆包都提供 `/v1/chat/completions` 兼容端点，一个客户端切三家，不为每家装 SDK | 分别用 dashscope / volcengine SDK，代码要写两套 |
| **本地哈希向量（默认）** | 零下载、离线可用，避开 HuggingFace 下载失败 | 默认 all-MiniLM 需下载 80MB，国内新手常卡在第一步 |
| **Streamlit** | 纯 Python 写交互，不用写前端工程 | Gradio 也可，但自定义布局更弱 |

**设计取舍（面试可以讲）**

- **不用「一个大 prompt 解决所有问题」**：意图识别走规则优先，稳定、零 token、可解释；大模型只负责「基于资料组织语言」
- **两段式兜底**：检索弱 → 直接说「没找到依据」；模型挂了 → 抽取式摘要。保证任何情况下都不胡说、页面不白屏
- **Embedding 可切换**：`local` 与 `dashscope` 二选一，切换后自动重建知识库（检测维度不一致）

---

## 三、环境依赖

- Python **3.10+**（本项目在 3.13 上验证）
- 依赖见 `requirements.txt`，**安装体积约 300MB**（chromadb 会带上 onnxruntime、pyarrow）
- 大模型 API **可选**：不配也能跑（离线兜底模式）

> 弱网环境首次 `pip install` 可能要 20 分钟以上，建议直接用国内镜像（见步骤 2）。

| 包 | 用途 |
| --- | --- |
| langchain / langchain-openai | 文本切分、ChatModel 封装 |
| langgraph | 对话状态机 |
| chromadb | 向量库（持久化） |
| streamlit | 前端 |
| python-dotenv | 读取 .env |

---

## 四、目录结构

```
宠物养护智能咨询 Agent/
├── app.py                      # Streamlit 入口
├── config.py                   # 配置（环境变量，带安全默认值）
├── requirements.txt
├── .env.example                # 配置模板（复制为 .env）
├── src/
│   ├── graph.py                # LangGraph 工作流（核心）
│   ├── knowledge.py            # 知识库构建与检索
│   ├── embedding.py            # 向量化（本地哈希 / 云端）
│   ├── llm.py                  # 大模型客户端（DeepSeek/通义/豆包/离线）
│   ├── recommender.py          # 服务推荐
│   └── prompts.py              # 提示词与离线抽取式摘要
├── data/
│   ├── knowledge/*.md          # 6 篇垂直知识文档
│   └── services.json           # 服务清单（与平台对齐）
├── scripts/
│   ├── build_kb.py             # 命令行重建知识库
│   ├── chat_cli.py             # 终端调试（不启网页也能聊）
│   └── smoke_test.py           # 一键自检（5 类场景 + 多轮记忆）
├── .streamlit/config.toml      # 主题（与平台同色系）
├── docs/
│   ├── 01-架构设计.md
│   └── 02-简历亮点与面试.md
└── chroma_db/                  # 运行后自动生成
```

---

## 五、本地运行（Windows，从零开始）

### 步骤 1：准备 Python 环境

```bat
:: 换成你自己的项目路径（路径含空格时务必加引号）
cd /d "D:\projects\宠物养护智能咨询 Agent"
python -m venv .venv
.venv\Scripts\activate
```

> 装好依赖前不要关这个窗口。若提示 `python 不是内部或外部命令`，说明 Python 没加进 PATH。

### 步骤 2：安装依赖

```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

国内建议加清华镜像加速：

```bat
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 步骤 3：配置大模型（可选）

不配置也能跑（离线兜底模式）。想用真实模型：

```bat
copy .env.example .env
notepad .env
```

填其中一个即可（四选一）：

```ini
# ① DeepSeek（推荐：便宜、国内直连、中文好）
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DEEPSEEK_MODEL=deepseek-chat

# ② 通义千问（阿里云百炼，有免费额度）
# DASHSCOPE_API_KEY=sk-xxxxxxxx
# DASHSCOPE_MODEL=qwen-plus

# ③ 豆包（火山方舟，Key 是 UUID 格式，模型名可填接入点 ID）
# ARK_API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
# ARK_MODEL=doubao-pro-32k

# ④ 任意 OpenAI 兼容端点（魔搭 / 硅基流动 / 中转站）
# LLM_API_KEY=xxx
# LLM_BASE_URL=https://api-inference.modelscope.cn/v1
# LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
```

申请入口：通义 https://bailian.console.aliyun.com/ ｜ 豆包 https://console.volcengine.com/ark

### 步骤 4：启动

```bat
streamlit run app.py
```

浏览器自动打开 http://localhost:8501 。

首次启动会自动构建知识库（几秒钟），日志里会看到：

```
[knowledge] 构建完成：45 个片段，6 篇文档，向量维度 768（local）
```

### 步骤 5（可选）一键自检

改完代码或换模型后，先跑自检比手点页面快：

```bat
python scripts\smoke_test.py
```

它跑 5 类场景（知识问答 / 症状咨询 / 信息不足追问 / 服务推荐 / 闲聊）加一个多轮记忆验证，
全部通过会打印 `全部通过 ✓`。

### 步骤 6（可选）终端调试

不启网页，直接在终端对话：

```bat
python scripts\chat_cli.py
```

这个入口更适合排查「是检索没召回，还是模型没按资料答」。

### 步骤 7（可选）重建知识库

改了 `data/knowledge/` 下的文档后：

```bat
python scripts\build_kb.py
python scripts\build_kb.py --check   # 只看状态
```

界面上也能点「重建知识库（全量）」，或者直接上传 `.md` / `.txt` 追加。

---

## 六、演示脚本（建议按这个顺序讲）

| 顺序 | 输入 | 预期表现 |
| --- | --- | --- |
| 1 | `你好` | 走 chitchat 分支，介绍能力清单 |
| 2 | `三个月大的布偶猫一天喂多少克猫粮？` | 命中《猫咪喂养》章节，给出克数区间与体况评分建议，参考资料可展开 |
| 3 | `狗狗拉稀两天了，精神还可以，要紧吗？` | 命中《常见疾病预警》腹泻章节，给出观察要点与就医红线 |
| 4 | `它一直吐，还不吃饭` | 信息不足 → 触发追问（品种/年龄），体现槽位检查 |
| 5 | `下周出差五天，猫没人管怎么办？` | 意图判为 service → 推荐「猫咪上门喂养·单次」，带预约链接 |
| 6 | 侧边栏把 `top_k` 调到 1、阈值调到 0.5 | 再问同样的养护问题 → 明显看到「知识库未覆盖」兜底话术，说明阈值真实生效 |

第 6 步是加分项：**主动演示参数如何影响效果**，比只演示"能答"更专业。

---

## 七、与「宠物服务上门平台」的联动

| 联动点 | 实现 |
| --- | --- |
| 服务数据同源 | `data/services.json` 的 `id` 与平台 `t_service.id` 一一对应 |
| 下架过滤 | 平台下架的服务（`status=0`）不会出现在推荐里 |
| 一键转化 | 推荐卡片按钮跳转 `{PLATFORM_BASE_URL}/user/service/{id}`，直接进预约页 |
| 场景互补 | 平台解决「谁上门」，Agent 解决「要不要上门、上门做什么」 |

想让链接可点，先按平台项目的 README 把前端跑起来（默认 5173）。不启动也不影响对话，只是链接打不开。

---

## 八、常见问题

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| `pip install` 卡住 / 超时 | 网络 | 加 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| 首次启动很慢 | 正在构建知识库 | 正常，约 3-10 秒；之后走 `chroma_db/` 缓存 |
| 回答总是「知识库未覆盖」 | 阈值太高或向量方案变了 | 侧边栏调低相似度阈值；或重建知识库 |
| 切换了 EMBEDDING_PROVIDER 后报维度错误 | 旧库维度不匹配 | `python scripts\build_kb.py` 重建（会自动检测） |
| 页面报 `ModuleNotFoundError` | 没激活虚拟环境 | `.venv\Scripts\activate` 后重跑 |
| 中文乱码 | 终端编码 | `chcp 65001`，或用 Windows Terminal |
| 模型调用失败但流程还在跑 | 已降级为抽取式回答 | 看侧边栏是否显示「离线兜底模式」，检查 Key 与额度 |

---

## 九、已知边界

- `MemorySaver` 是进程内存储，**重启后上下文丢失**；生产应换 Redis / Postgres checkpointer
- 检索是单轮改写，没做 query 改写与多轮指代消解（「它」指谁要靠画像兜）
- 没有做答案的事实校验（faithfulness 评估），只靠 prompt 约束 + 引用展示
- 知识库是人工整理的 6 篇文档，不是海量语料，覆盖面有限

这些边界写在文档里，比假装完美更经得起追问。
