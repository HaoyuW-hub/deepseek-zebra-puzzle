# DeepSeek Zebra Puzzle — Two-Stage Reasoning Experiment

## 实验目的

研究 LLM 在约束推理任务上，推理链长度与准确率的关系，验证是否存在逆缩放（inverse scaling）现象。

**两个实验模式：**

| 模式 | budget 符号 | Stage 1 prompt | 含义 |
|------|------------|----------------|------|
| Interrupted（通知截断） | >0 | 告知模型"推理将在 N tokens 后被中断" | 模型有预期地安排推理节奏 |
| Natural（自然推理） | <0 | 正常推理，不告知截断 | 模型自然思考，被 API 暴力截断 |

**两阶段框架：**
- Stage 1：模型推理，`max_tokens` 硬截断，捕获部分推理链
- Stage 2：注入 "Time's Up!" 提示 + assistant prefill `<answer>`，强制提取答案

## 环境配置

### 前置要求

- Python 3.12+
- GPU：H800 80GB × 1 或 × 2（推荐）
- CUDA 12.8+ 驱动

### 安装

```bash
# 1. 克隆项目
git clone <repo-url>
cd deepseek-zebra-puzzle

# 2. 创建虚拟环境
python3.12 -m venv .venv
.venv/bin/pip install -e ./safety-tooling
.venv/bin/pip install -e .

# 3. 配置 API key（仅 DashScope 模式需要）
cp .env.example .env
# 编辑 .env，填写 DASHSCOPE_API_KEY
```

### vLLM 本地部署（推荐）

```bash
# 安装 vLLM
/opt/vllm-env/bin/pip install vllm

# 下载模型（首次）
pip install huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --local-dir /opt/models/DeepSeek-R1-Distill-Qwen-14B

# 启动 vLLM（H800 单卡）
/opt/vllm-env/bin/vllm serve /opt/models/DeepSeek-R1-Distill-Qwen-14B \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.85
```

## 运行实验

### 完整实验（本地 vLLM）

```bash
.venv/bin/python run.py \
  --model-config config/model/deepseek_r1_local.yaml \
  --validation --validation-samples 103 --validation-runs 5 \
  --reasoning-budgets="-5,-4,-3,-2,-1,0,1024,2048,4096,8192" \
  --concurrency 8
```

### 双卡并行（减半时间）

先切分数据：

```bash
python3 -c "
import json, random
with open('data/bbeh_zebra_puzzles_scored_g567.jsonl') as f:
    puzzles = [json.loads(line) for line in f if line.strip()]
random.seed(42)
random.shuffle(puzzles)
for i, chunk in enumerate([puzzles[:52], puzzles[52:]]):
    with open(f'data/g567_part{i+1}.jsonl', 'w') as f:
        for p in chunk: f.write(json.dumps(p, ensure_ascii=False)+'\n')
    print(f'Part {i+1}: {len(chunk)}')
"
```

然后两终端分别跑：

```bash
# 终端1（vLLM 端口 8000）
CUDA_VISIBLE_DEVICES=0 /opt/vllm-env/bin/vllm serve \
  /opt/models/DeepSeek-R1-Distill-Qwen-14B \
  --host 0.0.0.0 --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.85

.venv/bin/python run.py --model-config config/model/deepseek_r1_local.yaml \
  --task-config config/task/zebra_puzzle_p1.yaml \
  --validation --validation-samples 52 --validation-runs 5 \
  --reasoning-budgets="-5,-4,-3,-2,-1,0,1024,2048,4096,8192" \
  --concurrency 8 --output-dir results/run1

# 终端2（vLLM 端口 8001，另一张 H800）
CUDA_VISIBLE_DEVICES=1 /opt/vllm-env/bin/vllm serve \
  /opt/models/DeepSeek-R1-Distill-Qwen-14B \
  --host 0.0.0.0 --port 8001 --max-model-len 65536 --gpu-memory-utilization 0.85

.venv/bin/python run.py --model-config config/model/deepseek_r1_local_8001.yaml \
  --task-config config/task/zebra_puzzle_p2.yaml \
  --validation --validation-samples 52 --validation-runs 5 \
  --reasoning-budgets="-5,-4,-3,-2,-1,0,1024,2048,4096,8192" \
  --concurrency 8 --output-dir results/run2
```

### DashScope API 模式（备选）

```bash
.venv/bin/python run.py --model-config config/model/deepseek_r1.yaml \
  --validation --validation-samples 103 --validation-runs 3 \
  --reasoning-budgets="-5,-4,-3,-2,-1,0,1024,2048,4096,8192" \
  --concurrency 40
```

## budget 参数说明

| budget | max_tokens | 模式 |
|--------|-----------|------|
| `0` | 16384 | Interrupted (DO NOT THINK) |
| `1024` | 16384 | Interrupted (告知 ~1024 tokens 后中断) |
| `2048` | 16384 | Interrupted |
| `4096` | 16384 | Interrupted |
| `8192` | 16384 | Interrupted |
| `-1` | 16384 | Natural |
| `-2` | 20480 | Natural |
| `-3` | 24576 | Natural |
| `-4` | 28672 | Natural |
| `-5` | 32768 | Natural |

## 结果分析

### 框架验证

在跑完整实验前，先验证两阶段框架正常工作：

```bash
.venv/bin/python scripts/test_two_stage.py
```

包括 4 个纯逻辑测试（prompt 模板、Stage 2 结构、max_tokens 映射、答案提取）和 2 个 API 测试（截断验证、natural vs interrupted 分叉）。无 API 时至少 4 个逻辑测试应全部通过。

### 综合可视化

实验结束后一键生成所有图表：

```bash
.venv/bin/python scripts/analyze_results.py results/<result_dir>
```

**产出：**

| 文件 | 内容 |
|------|------|
| `accuracy_vs_reasoning_tokens.png` | 双面板：natural / interrupted，各含 overall + grid=5/6/7 子线，x 轴为实际推理 token 数 |
| `truncation_rate.png` | 分组柱状图：每个 budget 下两种模式的 Stage 1 截断率 |
| `reasoning_token_stats.png` | 三面板：箱线图、均值±std 折线、per grid_size 折线 |
| `analysis_summary.json` | 数值统计（每 budget 的准确率、推理 token 均值/中位数、截断率、Stage2 跳过率） |

### 提取推理链 deduction 节点

```bash
# 正则提取（零成本）
.venv/bin/python scripts/extract_deductions.py results/<dir>/raw/<model>/<task>.jsonl

# LLM 提取（更高精度）
.venv/bin/python scripts/extract_deductions_llm.py results/<dir>/raw/<model>/<task>.jsonl
```

结果目录结构：

```
results/<model>__<task>/<timestamp>/
├── raw/<model>/<task>.jsonl              # 原始评测结果
├── results_df.csv                        # 汇总 DataFrame
└── analysis/
    ├── accuracy_vs_reasoning_tokens.png
    ├── truncation_rate.png
    ├── reasoning_token_stats.png
    ├── analysis_summary.json
    └── *_deductions.json / *_markers.json
```

## 脚本速查

| 脚本 | 用途 |
|------|------|
| `scripts/test_two_stage.py` | 验证两阶段框架（逻辑 + API 测试） |
| `scripts/analyze_results.py` | 生成 accuracy vs tokens 图、截断率图、token 统计 |
| `scripts/extract_deductions.py` | 正则提取推理链中的位置-属性对 |
| `scripts/extract_deductions_llm.py` | LLM 提取推理链中的位置-属性对 |
| `scripts/extract_markers.py` | 提取显式 `<deduction>` 标记 |

## 关键配置

| 文件 | 说明 |
|------|------|
| `config/model/deepseek_r1.yaml` | DashScope API 配置 |
| `config/model/deepseek_r1_local.yaml` | 本地 vLLM 配置 |
| `config/task/zebra_puzzle.yaml` | 数据集配置（103 题，grid 5-7） |
| `src/model_interface.py` | 两阶段调用核心逻辑 |
| `src/utils/model_helpers.py` | Stage 1/2 prompt 模板 |
