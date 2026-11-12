# HyperRAG：在线段落逻辑图 + 双曲双路召回

保留原始 question decomposition 提示词、示例和顺序回答结构。修复原代码缺少 model 参数和 End 解析；API 失败会抛出异常，不把错误消息当答案学习。

## 运行

在本目录建立独立环境，不修改 chemical 环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY='你的密钥'
python main.py --corpus corpus.example.jsonl --state state --question "Who wrote Don Juan, and how was he related to Ada Lovelace?"
```

默认沿用 gpt-4o-mini；用 --model 或 OPENAI_MODEL 修改。OPENAI_BASE_URL 可用于兼容端点。初次运行 SentenceTransformer 会下载模型；--encoder 可指定本地模型目录。输入为 JSONL，每行一个 id 唯一且 text 非空的 paragraph。状态缓存包含编码器和语料指纹，语料或模型变化时请指定新 --state。

## 算法

1. 一次分解原问题，简单问题 End 直接作为单步回答。
2. 每个子问题使用「上一步答案 + 本子问题」构造查询。Sentence-BERT cosine top-k 始终保留。
3. 冷启动只有语义召回。已有连接后，以本次语义 top-k 的双曲坐标为锚点，按最近 Poincaré 距离再取 top-k；两路去重合并，最多 2k 段，不让融合重排挤掉语义候选。该路线是 query-conditioned 锚点检索，不是训练 query encoder，也不是只查询边表。
4. 用当前候选和此前答案回答子问题，最终调用模型生成原问题答案。
5. 额外一次 LLM 调用审核所有步骤中的段落对，只有共同支撑本题的具体逻辑连接才写入；允许跨子问题连接，不因为同时召回就自动连边。校验 ID、去自环和本题重复边；保存原问题、理由、问题序号。
6. 回答完成后，局部更新本题连接端点及历史一跳邻居，其他节点保持不变。每 100 个成功处理的问题做全局校正（--global-every 配置，0 关闭）。本题新边从下一题开始影响检索。

## 双曲优化

使用曲率 -1 的 Poincaré ball，SBERT 归一化向量乘 0.2 初始化，等价于 exp_0(atanh(0.2) * unit_embedding)。语义向量冻结，只更新段落双曲坐标。

使用 Geoopt 的 `PoincareBall(c=1.0)`、`ManifoldParameter` 和 `RiemannianAdam`，由 Geoopt 计算双曲距离和处理流形更新。

局部训练和全局校正均固定为 **1 epoch**：随机打乱当前选中的所有边，每条边恰好遍历一次，每批最多 256 条。一次 epoch 可包含多个 optimizer step；每 100 题的全局校正在局部训练后额外执行一次 epoch。历史重复确认可重复保留，形成隐式边权。

损失 = 正边双曲距离平方 + 非已知邻居负样本的 margin 排斥 + `10 × 到原始初始化坐标的双曲距离平方`。默认学习率 **1e-4**，不训练 Sentence-BERT encoder，原始语义向量始终固定。

为防止多题小更新累积后偏离原始语义，每个段落相对 `0.2 × normalized SBERT` 初始化坐标的累计欧氏位移上限为 **0.01**（初始化范数的 5%）。更新后及旧状态加载后均执行该约束。旧 checkpoint 可直接读取，但超出上限的历史坐标会收回该范围。训练超参数写入 index.pt 的 training 字段，恢复时采用当前命令行配置。

可用 `--learning-rate 1e-5` 进一步调低学习率；`--anchor-weight` 和 `--max-drift` 分别调整初始化约束与累计位移上限。

这是简洁研究原型：精确检索 O(NKd)，全局优化含 O(Nd) 坐标/约束计算；未实现百万段落 ANN 或并发写状态。用单进程顺序处理问题，语料规模大时再加 ANN/批量优化。

## 文件与验证

- main.py：分解、链式回答、LLM 连边审核、命令行。
- retrieve.py：语义索引、双曲检索、局部/全局更新、状态恢复。
- prompts.py：原始提示词。
- state/index.pt：坐标、语义向量、边及计数，原子替换保存。
- state/edges.json：可读边记录；question_*.json：每题完整轨迹。
- test_hyperrag.py：无 API、无模型下载的测试，使用模拟编码器和 LLM。

```bash
python -m unittest -v test_hyperrag.py
```

保留了原始回答提示词允许模型使用自身知识的设置；因此最终答案不一定全部来自 corpus，LLM 判断的连接也不是人工核实事实。评估时应区分语义召回、双路召回和在线学习，并固定问题顺序、排除测试答案监督。

API JSON 模式参照官方文档：https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat 。未调用真实付费 API 验证回答质量。

## 答案评估（与 judge_links 分开）

`evaluate.py` 恢复原来的两种评估，结果不参与连边或双曲训练：

- `str-match`：小写、去标点、去英文冠词、空格归一化后，只要预测包含 gold 即记 ContainsMatch=1；不是严格 exact match。额外修正空 gold 自动命中所有答案的问题。
- `llm`：沿用原始 yes/no 判分提示词，最多 5 路并行，与标准答案比较，返回 LLMAccuracy。模型由传入的 llm 或 --model 指定，修复原始 get_gpt_response 缺少 model 参数的问题。yes 忽略首尾空白和大小写；其他输出不算 yes。API 失败抛异常，不悄悄算错。
- `both`：同时返回两项指标。

输入 JSONL 每行包含 `question`、`answer`、`correct_answer`：

```bash
python evaluate.py --input predictions.jsonl --mode str-match --output evaluation.json
python evaluate.py --input predictions.jsonl --mode both --model gpt-4o-mini --output evaluation.json
```

也可调用 `evaluate_answers(questions, generated_answers, correct_answers, mode='both', llm=llm)`。`judge_links` 仍只用于判断段落间逻辑联系，不使用 gold answer。

## HotpotQA + DeepSeek

`main.LLM` 默认使用 DeepSeek，环境变量 `DEEPSEEK_API_KEY` 提供密钥，`DEEPSEEK_BASE_URL` 可覆盖端点（默认 https://api.deepseek.com）。默认模型 deepseek-flash，问答及 LLM judge 使用同一模型。

```bash
python run_hotpotqa.py --dataset dataset/hotpotqa.json --corpus dataset/hotpotqa_corpus.json --run-dir runs/hotpotqa_deepseek --model deepseek-flash
```

按输入顺序在线学习，完整 corpus 检索，每题语义 top-5 + 双曲 top-5 去重；语料 title 与 text 共同编码。只读 question 和 corpus 生成答案和图连接，answer 仅用于随后评估，完全不读取 context/supporting_facts。每题保存回答、检索证据、图连接和两项判分；summary.json 是截至当前已完成题目的统计，不代表全量结果。固定数据文件 hash、模型与参数，重跑相同命令可跳过已有记录，API 判分失败时下次只补判分。状态更新与轨迹写入之间意外中断会报错，避免默默重复训练。
