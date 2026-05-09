"""
Plotting utilities for inverse scaling evaluation.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import ast

import matplotlib
matplotlib.use('Agg')
try:
    matplotlib.rcParams['figure.constrained_layout.use'] = False
except (KeyError, ValueError):
    pass

matplotlib.rcParams['figure.max_open_warning'] = 0
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from matplotlib.cm import ScalarMappable
import math
from safetytooling.apis.inference.openai.utils import count_tokens
import textwrap

logger = logging.getLogger(__name__)


NON_REASONING_MODELS = ["claude-3-5-sonnet-20240620", "claude-3-5-sonnet-20241022", "gpt-4.1-mini-2025-04-14"]
OPENAI_REASONING_MODELS = ["o3-mini-2025-01-31", "o3-2025-04-16", "o4-mini-2025-04-16"]

MODEL_TO_PRETTY_NAME = {
    "deepseek-reasoner": "DeepSeek R1",
    "claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet",
    "claude-3-7-sonnet-20250219_natural_overthinking": "Claude 3.7 Sonnet (Natural)",
    "claude-3-7-sonnet-20250219_not_use_all_budget": "Claude 3.7 Sonnet (Cautioned)",
    "claude-sonnet-4-20250514": "Claude 4 Sonnet",
    "claude-opus-4-20250514": "Claude 4 Opus",
    "Qwen3-8B": "Qwen3 8B",
    "Qwen3-14B": "Qwen3 14B",
    "Qwen3-32B": "Qwen3 32B",
    "O4-mini": "o4-mini",
    "o4-mini-2025-04-16": "o4-mini",
    "o3-mini-2025-01-31": "o3-mini",
    "o3-2025-04-16": "o3",
}

TASK_TO_PRETTY_NAME = {
    "synthetic_misleading_math": "Synthetic Misleading Math",
    "synthetic_misleading_python": "Synthetic Misleading Python",
    "student_lifestyle_regression_Grades": "Student Lifestyle Regression",
    "synthetic_misleading_math_famous_paradoxes": "Famous Paradoxes",
    "bbeh_zebra_puzzles": "BBEH Zebra Puzzles",
}

COLOR_MAP = {
    "DeepSeek R1": "#4C72B0",
    "Claude 3.7 Sonnet": "#DD8452",
    "Claude 4 Sonnet": "#55A868",
    "Claude 4 Opus": "#C44E52",
    "o4-mini": "#8172B2",
    "o3-mini": "#937860",
    "o3": "#DA8BC3",
}

BUDGET_MARKER_MAP = {
    0: 'o',
    1024: 's',
    2048: 'D',
    4096: '^',
    8192: 'v',
    16384: 'P',
}

BUDGET_COLOR_MAP = {
    0: '#1f77b4',
    1024: '#ff7f0e',
    2048: '#2ca02c',
    4096: '#d62728',
    8192: '#9467bd',
    16384: '#8c564b',
}


def _get_pretty_name(model_id: str) -> str:
    return MODEL_TO_PRETTY_NAME.get(model_id, model_id)

def _get_task_pretty_name(task_id: str) -> str:
    return TASK_TO_PRETTY_NAME.get(task_id, task_id)

def _get_color(model_id: str) -> str:
    pretty = _get_pretty_name(model_id)
    return COLOR_MAP.get(pretty, "#333333")

def _parse_classes(classes_raw: Any) -> List[str]:
    if isinstance(classes_raw, list):
        return [str(c).strip() for c in classes_raw]
    if isinstance(classes_raw, str):
        try:
            evaluated = ast.literal_eval(classes_raw)
            if isinstance(evaluated, list):
                return [str(c).strip() for c in evaluated]
        except (ValueError, SyntaxError, TypeError):
            pass
    return []


def plot_token_correlations(
    df: pd.DataFrame,
    model: str,
    task: str,
    plot_dir: Path,
    metric: str = "accuracy",
) -> None:
    task_df = df[(df["model"] == model) & (df["task_id"] == task)].copy()

    if len(task_df) == 0:
        return

    task_df["output_tokens"] = pd.to_numeric(task_df["output_tokens"], errors="coerce")
    task_df["input_tokens"] = pd.to_numeric(task_df["input_tokens"], errors="coerce")

    task_df_clean = task_df.dropna(subset=["output_tokens", "input_tokens", "correct"])

    if len(task_df_clean) < 3:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax1 = axes[0]
    budgets = sorted(task_df_clean["reasoning_budget"].unique())
    for budget in budgets:
        budget_df = task_df_clean[task_df_clean["reasoning_budget"] == budget]
        if len(budget_df) > 0:
            ax1.scatter(
                budget_df["output_tokens"],
                budget_df["correct"].astype(float),
                alpha=0.6,
                label=f"Budget {budget}",
            )
    ax1.set_xlabel("Output Tokens")
    ax1.set_ylabel(metric.capitalize())
    ax1.set_title(f"{_get_pretty_name(model)} - {_get_task_pretty_name(task)}\nOutput Tokens vs {metric}")
    ax1.legend(fontsize="small")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    task_df_for_hist = task_df_clean.copy()
    task_df_for_hist["budget_label"] = task_df_for_hist["reasoning_budget"].apply(lambda b: f"Budget {b}")
    for budget in budgets:
        budget_df = task_df_for_hist[task_df_for_hist["reasoning_budget"] == budget]
        if len(budget_df) > 0:
            ax2.hist(budget_df["output_tokens"], alpha=0.5, label=f"Budget {budget}", bins=20)
    ax2.set_xlabel("Output Tokens")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Output Token Distribution by Budget")
    ax2.legend(fontsize="small")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    if len(task_df_clean) >= 5:
        budget_groups = task_df_clean.groupby("reasoning_budget")
        budget_means = budget_groups["output_tokens"].mean()
        budget_acc = budget_groups["correct"].mean()
        corr, p_val = stats.pearsonr(budget_means, budget_acc)
        ax3.scatter(budget_means, budget_acc, s=100, alpha=0.8)
        for budget in budget_means.index:
            ax3.annotate(
                str(budget),
                (budget_means[budget], budget_acc[budget]),
                fontsize=8,
            )
        ax3.set_xlabel("Mean Output Tokens")
        ax3.set_ylabel(f"Mean {metric}")
        ax3.set_title(f"Token-Accuracy Correlation\nr={corr:.3f}, p={p_val:.3f}")
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_model = model.replace("/", "_").replace(":", "-")
    safe_task = task.replace("/", "_").replace(":", "-")
    output_path = plot_dir / f"token_correlations_{safe_model}_{safe_task}_{metric}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_budget_length_boxplot(
    df: pd.DataFrame,
    plot_dir: Path,
    task_loader=None,
) -> None:
    if "reasoning_budget" not in df.columns or "correct" not in df.columns:
        return

    plot_df = df.copy()
    if "output_tokens" in plot_df.columns:
        plot_df["length"] = pd.to_numeric(plot_df["output_tokens"], errors="coerce")
    elif "response" in plot_df.columns:
        plot_df["length"] = plot_df["response"].apply(
            lambda x: len(str(x)) if pd.notna(x) else 0
        )
    else:
        return

    plot_df = plot_df.dropna(subset=["length", "correct", "reasoning_budget"])

    if len(plot_df) == 0:
        return

    budgets = sorted(plot_df["reasoning_budget"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    budget_data = [plot_df[plot_df["reasoning_budget"] == b]["length"].dropna().values for b in budgets]
    bp = ax1.boxplot(budget_data, labels=[str(b) for b in budgets], patch_artist=True)
    for patch, budget in zip(bp["boxes"], budgets):
        color = BUDGET_COLOR_MAP.get(budget, "#333333")
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax1.set_xlabel("Reasoning Budget")
    ax1.set_ylabel("Response Length")
    ax1.set_title("Response Length Distribution by Reasoning Budget")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    if task_loader is not None:
        try:
            task_ids = plot_df["task_id"].unique()
            task_budget_data = []
            task_labels = []
            for task_id in task_ids:
                for budget in budgets:
                    sub = plot_df[(plot_df["task_id"] == task_id) & (plot_df["reasoning_budget"] == budget)]
                    if len(sub) > 0:
                        task_budget_data.append(sub["length"].dropna().values)
                        task_labels.append(f"{_get_task_pretty_name(task_id)[:15]}\nB={budget}")
            if task_budget_data:
                ax2.boxplot(task_budget_data, labels=task_labels, patch_artist=True)
                ax2.set_xlabel("Task / Budget")
                ax2.set_ylabel("Response Length")
                ax2.set_title("Response Length by Task and Budget")
                ax2.grid(True, alpha=0.3)
                plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        except Exception:
            pass

    plt.tight_layout()
    output_path = plot_dir / "budget_length_boxplot.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_token_scaling_curves_improved(
    df: pd.DataFrame,
    task_loader,
    models: List[str],
    tasks: List[str],
    plot_dir: Path,
    plot_type: str = "accuracy",
    min_samples_per_point: int = 3,
) -> None:
    if plot_type == "accuracy":
        metric_col = "correct"
        ylabel = "Accuracy"
    elif plot_type == "mse":
        metric_col = "squared_error"
        ylabel = "Mean Squared Error"
    else:
        metric_col = plot_type
        ylabel = plot_type.replace("_", " ").title()

    plot_df = df[df["model"].isin(models) & df["task_id"].isin(tasks)].copy()

    if metric_col not in plot_df.columns or plot_df[metric_col].isna().all():
        return

    plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric_col])

    n_tasks = len(tasks)
    n_cols = min(3, n_tasks)
    n_rows = math.ceil(n_tasks / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    model_styles = {}
    for model in models:
        pretty = _get_pretty_name(model)
        color = _get_color(model)
        marker = 'o'
        model_styles[model] = {"label": pretty, "color": color, "marker": marker}

    for idx, task in enumerate(tasks):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]

        task_df = plot_df[plot_df["task_id"] == task]
        if len(task_df) == 0:
            ax.set_title(f"{_get_task_pretty_name(task)}\n(No Data)")
            continue

        for model in models:
            model_task_df = task_df[task_df["model"] == model]
            if len(model_task_df) == 0:
                continue

            budget_groups = model_task_df.groupby("reasoning_budget")[metric_col]
            budget_means = budget_groups.mean()
            budget_counts = budget_groups.count()

            valid_budgets = budget_counts[budget_counts >= min_samples_per_point].index
            valid_means = budget_means[valid_budgets]

            if len(valid_budgets) < 2:
                continue

            style = model_styles[model]
            ax.plot(
                valid_budgets,
                valid_means,
                marker=style["marker"],
                color=style["color"],
                label=style["label"],
                linewidth=2,
                markersize=8,
            )

        ax.set_xlabel("Reasoning Budget")
        ax.set_ylabel(ylabel)
        ax.set_title(_get_task_pretty_name(task))
        ax.grid(True, alpha=0.3)
        if col == n_cols - 1:
            ax.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1.05, 1))

    for idx in range(n_tasks, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row][col].set_visible(False)

    plt.suptitle(f"{ylabel} vs Reasoning Budget", fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_path = plot_dir / f"token_scaling_curves_{plot_type}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_token_slopes(
    df: pd.DataFrame,
    models: List[str],
    tasks: List[str],
    plot_dir: Path,
    plot_type: str = "accuracy",
) -> None:
    if plot_type == "accuracy":
        metric_col = "correct"
        ylabel = "Accuracy Slope"
    else:
        metric_col = plot_type
        ylabel = f"{plot_type} Slope"

    plot_df = df[df["model"].isin(models) & df["task_id"].isin(tasks)].copy()

    if metric_col not in plot_df.columns or plot_df[metric_col].isna().all():
        return

    plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric_col])

    fig, ax = plt.subplots(figsize=(10, 6))

    slope_data = []
    for model in models:
        for task in tasks:
            model_task_df = plot_df[(plot_df["model"] == model) & (plot_df["task_id"] == task)]
            if len(model_task_df) < 2:
                continue

            budget_groups = model_task_df.groupby("reasoning_budget")[metric_col].mean()
            budgets = budget_groups.index.values
            values = budget_groups.values

            if len(budgets) >= 2:
                slope, _, _, _, _ = stats.linregress(budgets, values)
                slope_data.append({
                    "model": _get_pretty_name(model),
                    "task": _get_task_pretty_name(task),
                    "slope": slope,
                    "color": _get_color(model),
                })

    if not slope_data:
        plt.close(fig)
        return

    slope_df = pd.DataFrame(slope_data)
    slope_df = slope_df.sort_values("slope")

    colors = slope_df["color"].values
    bars = ax.barh(
        range(len(slope_df)),
        slope_df["slope"].values,
        color=colors,
        alpha=0.7,
    )

    ax.set_yticks(range(len(slope_df)))
    ax.set_yticklabels(
        [f"{r['model']} - {r['task']}" for _, r in slope_df.iterrows()],
        fontsize=8,
    )
    ax.set_xlabel(ylabel)
    ax.set_title(f"Performance Slope (Change per Reasoning Budget Increase)")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = plot_dir / f"token_slopes_{plot_type}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
