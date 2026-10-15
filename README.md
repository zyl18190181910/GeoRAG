# 地学知识库问答与自动化评测系统

这是一个面向实习作品集的小型 RAG 工程项目。它强调可解释、可测试和可复现，而不是简单调用聊天接口。系统采用“关键词检索 + 本地语义检索 + DeepSeek 受约束生成”的混合架构。

系统支持：

- 导入 TXT、Markdown 和 PDF 地学资料
- 中文 PDF 字间空格清洗、文本分块、TF-IDF 与 BGE-M3 多语种语义混合检索
- 使用 `BAAI/bge-reranker-base` 交叉编码器对候选片段进行精排
- 使用 DeepSeek 自动识别问题意图并生成多组检索表达
- 多路检索、来源/章节多样性排序、相邻片段补全和动态上下文数量
- 事实查找、归纳总结、分析建议三种回答方式
- 支持连续追问，并使用最近对话理解省略与指代
- 使用 DeepSeek 对多个资料片段进行改写、归纳、制表和有依据的分析
- 回答附带来源、原文证据和相关度分数；PDF 保留页码，Markdown 保留章节标题
- 资料不足时拒绝回答
- DeepSeek 或网络不可用时自动降级为本地抽取式回答
- DeepSeek 每次调用使用独立连接，连接失败时自动重建连接并进行指数退避重试
- 支持资料启用、停用、删除和一键重建索引，状态可持久化
- 聊天式作品集界面，可展开查看查询改写、检索过程与引用证据
- 内置生成质量与引用可靠性看板；检索消融通过独立脚本和 API 保留，避免干扰主展示页面
- FastAPI 接口与自动生成的 OpenAPI 文档
- JSONL 评测集、命中率、拒答准确率和延迟统计
- pytest 单元测试与接口测试

## 当前实测结果

在 Windows、Python 3.13 环境下运行：

- 79 项自动化测试全部通过
- 核心代码测试覆盖率 93%
- 4 道跨片段展示评测全部通过：生成成功率、来源覆盖率、要点召回率和引用有效率均为 1.000
- 展示评测平均端到端耗时约 7.1 秒（包含问题规划、重排序和 DeepSeek 生成）
- 以 4 篇中文公开论文、317 个候选片段和 60 道中英双语问题进行严格证据片段评测
- 在相同论文、问题、分块和精排器下，BGE-M3 相比 `bge-small-zh-v1.5` 将 Hit@3 提升 11.7 个百分点、MRR 提升 7.2 个百分点
- 完整原始指标、失败用例与延迟均保存在 `reports/ablation_latest.json`，便于复核而不占用主展示页面

评测数字来自仓库内保留的可复现评测报告；演示数据规模较小，不能等同于生产环境效果。

## 项目结构

```text
geo_rag/                 核心代码
  api.py                 FastAPI 接口
  chunking.py            文本规范化与分块
  evaluation.py          自动化评测
  ablation.py            三种检索链路的消融对比
  loaders.py             TXT Markdown PDF 解析
  retrieval.py           稀疏 TF-IDF 检索
  semantic.py            本地语义向量与混合检索
  reranking.py           BGE 交叉编码重排序
  generation.py          DeepSeek 受约束回答生成
  showcase.py            跨片段总结质量评测
  service.py             RAG 业务流程与拒答策略
  storage.py             JSON 持久化
data/sample_docs/        生成质量评测用的原创演示资料
data/public_papers/      公开论文清单；PDF 由脚本下载且不提交仓库
data/eval/               基础评测集与 60 题公开论文检索集
scripts/                 初始化与评测脚本
tests/                   单元测试和 API 测试
```

## Windows 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python scripts\download_public_papers.py
python scripts\rebuild_public_knowledge.py
python scripts\prepare_embedding_model.py
python scripts\build_public_vector_cache.py
python -m uvicorn geo_rag.api:app --reload
```

浏览器打开 `http://127.0.0.1:8000` 使用中文上传与问答页面；打开 `http://127.0.0.1:8000/docs` 可以查看开发者接口文档。

## 启用 DeepSeek 智能回答

在 Windows 用户环境变量中配置 `DEEPSEEK_API`（也兼容 `DEEPSEEK_API_KEY`），然后重新打开 PowerShell并启动服务。项目默认使用 `deepseek-flash`、`BAAI/bge-m3` 本地多语种向量模型和 `BAAI/bge-reranker-base` 本地重排序模型。结构化回答阶段关闭隐藏思考，以避免长上下文耗尽输出额度后无法生成 JSON。

首次启动会下载向量模型并缓存到 `data/models/`；论文片段向量另存于 `data/models/vector_cache/`，资料和模型不变时，后续重启会直接复用。可通过 `GEO_RAG_EMBEDDING_DEVICE=cpu|cuda|auto` 和 `GEO_RAG_EMBEDDING_BATCH_SIZE` 控制运行资源；CUDA 显存不足时会自动回退到 CPU。原始 PDF 不会发送给 DeepSeek；问题规划阶段只发送问题与最近对话，回答阶段发送检索命中的少量文本片段。不要导入保密资料，使用个人资料前请确认你愿意把这些内容发送给 DeepSeek。

## 运行测试

```powershell
python -m pytest
python -m pytest --cov=geo_rag --cov-report=term-missing
```

## 运行评测

```powershell
python scripts\evaluate.py
python scripts\evaluate_showcase.py
python scripts\evaluate_ablation.py
```

首次运行公开论文检索评测前，执行 `python scripts\download_public_papers.py`。下载脚本依据 `data/public_papers/manifest.json` 获取 4 篇公开论文，并校验文件是否为 PDF；论文文件已被 `.gitignore` 排除，不会随代码仓库重复分发。

评测输出包括：

- `source_hit_at_k`：期望资料是否出现在前 K 个检索结果中
- `refusal_accuracy`：该回答时回答、该拒答时拒答的比例
- `mean_latency_ms`：平均问答延迟

基础评测结果默认保存到本地 `reports/evaluation_latest.json`；该文件可能包含用户当前知识库的资料名称，因此默认不提交公开仓库。

`evaluate_showcase.py` 只使用 `data/sample_docs/` 中的三份原创样例资料，运行 4 个跨片段总结与分析用例，报告生成成功率、来源覆盖率、要点召回率、引用有效率和端到端延迟；结果保存到 `reports/showcase_latest.json`，也会显示在首页效果看板中。

`evaluate_ablation.py` 使用 4 篇公开论文和 60 道中英双语问题，在严格证据片段粒度比较 TF-IDF、TF-IDF + BGE 向量和加入 BGE Cross-Encoder 精排的三条检索链路。只有返回片段同时属于预期论文并包含标注证据词才算命中。报告包含 Hit@1、Hit@3、MRR、失败用例和单题平均延迟，保存到 `reports/ablation_latest.json`。评测不调用 DeepSeek，也不读取用户上传资料。

公开评测资料包括深度自编码网络地壳厚度反演、Transformer 大地电磁反演成像、余弦退火与空洞卷积遥感语义分割、地学知识图谱引导遥感语义分割 4 篇中文论文。标题、期刊、官方下载地址和使用说明记录在 `data/public_papers/manifest.json`；PDF 只下载到本机并被 Git 忽略，避免重复分发期刊文件，也避免把个人简历或课程材料混入公开结果。

## 主要接口

- `GET /health`：服务、资料和分块数量
- `GET /sources`：已导入资料列表
- `GET /sources/details`：资料状态、分块数量和文件信息
- `PATCH /sources/{source_id}`：启用或停用资料
- `DELETE /sources/{source_id}`：删除资料及其索引
- `POST /reindex`：根据当前启用资料重建索引
- `POST /ingest/text`：导入一段文本
- `POST /ingest/files`：上传 TXT、Markdown 或 PDF
- `POST /ask`：支持回答方式、最近对话、多查询检索和带出处回答
- `GET /evaluation/showcase`：读取最近一次展示评测
- `POST /evaluation/showcase/run`：仅用原创样例重新运行展示评测
- `GET /evaluation/ablation`：读取最近一次检索消融报告
- `POST /evaluation/ablation/run`：运行三种检索方案的证据片段级对比

## 如何替换成生成式模型

回答器通过 `AnswerGenerator` 接口与检索逻辑解耦。目前接入 DeepSeek 的 OpenAI 兼容接口；没有 API Key 或远端调用失败时，自动使用确定性的抽取式回答，检索、引用、拒答和评测模块无需重写。

## 简历使用原则

只有实际完成并能演示的功能才能写入简历。测试数量、问题数量、缺陷数量和性能数据应以真实运行结果为准，不要沿用计划数字。

## License

MIT

