# DeepSeek Zebra Puzzle: Inverse Scaling in Test-Time Compute

实验代码仓库，对应毕业论文《推理模型中的策略异化：长推理中的逆缩放机制分析与干预》。

## 项目结构

```
deepseek-zebra-puzzle/
├── run.py                     # 实验入口
├── src/                       # 核心实验框架
│   ├── evaluator.py           # 评估编排
│   ├── model_interface.py     # 两阶段 API 调用
│   ├── results_manager.py     # 结果持久化
│   ├── task_loader.py         # 数据加载
│   └── utils/
│       ├── model_helpers.py   # Prompt 模板与答案提取
│       ├── analysis.py        # 逆缩放统计分析
│       └── plotting.py        # 可视化工具
├── scripts/                   # 分析脚本
│   ├── extract_deduction_nodes.py   # LLM deduction 节点提取
│   ├── extract_deductions.py        # 正则 deduction 节点提取
│   ├── extract_deductions_llm.py    # 分块 LLM 节点提取
│   ├── extract_markers.py           # 显式 <deduction> 标记提取
│   ├── analyze_results.py           # 实验结果分析与可视化
│   └── test_two_stage.py            # 两阶段框架单元测试
├── config/                    # 配置文件
│   ├── model/deepseek_r1.yaml       # DashScope API 模型配置
│   └── task/zebra_puzzle.yaml       # 斑马谜题任务配置
└── requirements.txt
```

## 数据与结果

数据集与实验结果托管于 HuggingFace：`[username]/deepseek-zebra-puzzle`

## 环境配置

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

配置 API 密钥：复制 `.env.example` 为 `.env`，填入 `DASHSCOPE_API_KEY` 和 `DEEPSEEK_API_KEY`。

## 运行实验

```bash
# 自然推理（baseline）
python run.py --model-config config/model/deepseek_r1.yaml \
  --reasoning-budgets="-1" --concurrency 40

# 两阶段截断推理
python run.py --model-config config/model/deepseek_r1.yaml \
  --reasoning-budgets="0,1,1024,2048,4096,8192" --concurrency 40
```

## 引用

```bibtex
@thesis{wang2026strategy,
  title  = {推理模型中的策略异化：长推理中的逆缩放机制分析与干预},
  author = {王浩宇},
  school = {浙江大学},
  year   = {2026},
}
```
