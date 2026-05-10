#!/usr/bin/env python3
"""Post-experiment analysis & visualization."""

import json, sys, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

MODE_COLORS = {"interrupted": "#4C72B0", "natural": "#DD8452"}
GS_MARKERS = {5: "o", 6: "s", 7: "^"}
GS_COLORS = {5: "#55A868", 6: "#C44E52", 7: "#8C564B"}


def get_reasoning_tokens(r: dict) -> int | None:
    usage = r.get("usage")
    if isinstance(usage, dict):
        ctd = usage.get("completion_tokens_details", {})
        return ctd.get("reasoning_tokens")
    return None


def get_grid_size(r: dict) -> int | None:
    prompt = r.get("prompt", "")
    if isinstance(prompt, str):
        for gs in [5, 6, 7, 8]:
            if f"grid_size\": {gs}" in prompt or f"grid_size\":{gs}" in prompt:
                return gs
        # Try from prompt text
        import re
        m = re.search(r"are (\d+) people", prompt)
        if m:
            return int(m.group(1))
    return None


def main():
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        base = Path("results")
        dirs = sorted(base.glob("*__*"), reverse=True)
        if not dirs:
            print("No results found")
            return
        results_dir = dirs[0]
        # Find the raw jsonl
        jsonl_files = list(results_dir.rglob("*.jsonl"))
        if not jsonl_files:
            print(f"No jsonl found in {results_dir}")
            return
        results_path = jsonl_files[0]
    if results_dir.is_file():
        results_path = results_dir
        results_dir = results_path.parent.parent
    else:
        jsonl_files = list(results_dir.rglob("*.jsonl"))
        results_path = jsonl_files[0] if jsonl_files else None
        if not results_path:
            print(f"No jsonl found")
            return

    print(f"Loading: {results_path}")
    results = []
    with open(results_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                if "Batch processing failed" not in str(r.get("error", "")):
                    results.append(r)
            except json.JSONDecodeError:
                pass

    print(f"Loaded {len(results)} valid results")

    # Enrich data
    for r in results:
        r["rtokens"] = get_reasoning_tokens(r)
        r["gs"] = get_grid_size(r)
        r["mode"] = r.get("mode", "interrupted")
        r["is_truncated"] = r.get("stop_reason") in ("max_tokens", "length")

    df = pd.DataFrame(results)
    analysis_dir = results_dir.parent / "analysis"
    analysis_dir.mkdir(exist_ok=True, parents=True)

    # ========== Plot 1: Accuracy vs Reasoning Tokens (Natural vs Controlled) ==========
    print("\n=== Plot 1: Accuracy vs Reasoning Tokens ===")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, mode in enumerate(["interrupted", "natural"]):
        ax = axes[ax_idx]
        mode_df = df[df["mode"] == mode]

        if len(mode_df) == 0:
            ax.set_title(f"{mode.capitalize()} (no data)")
            continue

        # Overall line
        by_budget = mode_df.groupby("reasoning_budget")
        budgets = []
        accs = []
        tokens = []
        for b, grp in sorted(by_budget, key=lambda x: x[0]):
            mean_rt = grp["rtokens"].dropna().mean()
            mean_acc = grp["correct"].dropna().mean()
            if not np.isnan(mean_rt) and not np.isnan(mean_acc):
                budgets.append(b)
                tokens.append(mean_rt)
                accs.append(mean_acc)

        ax.plot(tokens, accs, marker="D", color=MODE_COLORS[mode],
                linewidth=2.5, markersize=8, label=f"{mode.capitalize()} (overall)")

        # Per grid_size
        for gs in [5, 6, 7]:
            gs_df = mode_df[mode_df["gs"] == gs]
            if len(gs_df) < 5:
                continue
            gs_budgets = []
            gs_tokens = []
            gs_accs = []
            for b, grp in sorted(gs_df.groupby("reasoning_budget"), key=lambda x: x[0]):
                mean_rt = grp["rtokens"].dropna().mean()
                mean_acc = grp["correct"].dropna().mean()
                if not np.isnan(mean_rt) and not np.isnan(mean_acc):
                    gs_budgets.append(b)
                    gs_tokens.append(mean_rt)
                    gs_accs.append(mean_acc)
            if len(gs_tokens) >= 2:
                ax.plot(gs_tokens, gs_accs, marker=GS_MARKERS[gs],
                        color=GS_COLORS[gs], linewidth=1.5, markersize=6,
                        alpha=0.7, linestyle="--",
                        label=f"grid={gs}")

        ax.set_xlabel("Mean Reasoning Tokens", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"{mode.capitalize()} Mode", fontsize=13)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

    plt.suptitle("Accuracy vs Reasoning Length", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = analysis_dir / "accuracy_vs_reasoning_tokens.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # ========== Plot 2: Truncation Rate per Budget ==========
    print("\n=== Plot 2: Truncation Rate ===")
    fig, ax = plt.subplots(figsize=(12, 5))

    modes = df["mode"].unique()
    budgets_all = sorted(df["reasoning_budget"].unique())
    x = np.arange(len(budgets_all))
    width = 0.35

    for i, mode in enumerate(sorted(modes)):
        mode_df = df[df["mode"] == mode]
        rates = []
        for b in budgets_all:
            sub = mode_df[mode_df["reasoning_budget"] == b]
            if len(sub) > 0:
                rates.append(sub["is_truncated"].mean() * 100)
            else:
                rates.append(0)
        bars = ax.bar(x + i * width, rates, width, label=mode.capitalize(),
                      color=MODE_COLORS.get(mode, "#333"), alpha=0.7)
        for bar, rate in zip(bars, rates):
            if rate > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{rate:.0f}%", ha="center", fontsize=8)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([str(b) for b in budgets_all])
    ax.set_xlabel("Budget", fontsize=11)
    ax.set_ylabel("Truncation Rate (%)", fontsize=11)
    ax.set_title("Stage 1 Truncation Rate by Budget and Mode", fontsize=13)
    ax.legend()
    path = analysis_dir / "truncation_rate.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # ========== Plot 3: Reasoning Token Statistics per Budget ==========
    print("\n=== Plot 3: Reasoning Token Statistics ===")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 3a: Boxplot
    ax = axes[0]
    box_data = []
    box_labels = []
    for b in budgets_all:
        sub = df[df["reasoning_budget"] == b]["rtokens"].dropna()
        if len(sub) > 0:
            box_data.append(sub.values)
            box_labels.append(str(b))
    ax.boxplot(box_data, labels=box_labels, patch_artist=True)
    ax.set_xlabel("Budget", fontsize=11)
    ax.set_ylabel("Reasoning Tokens", fontsize=11)
    ax.set_title("Reasoning Token Distribution by Budget", fontsize=12)

    # 3b: Mean +/- std
    ax = axes[1]
    for mode in sorted(modes):
        mode_df = df[df["mode"] == mode]
        means, stds, bs = [], [], []
        for b in budgets_all:
            sub = mode_df[mode_df["reasoning_budget"] == b]["rtokens"].dropna()
            if len(sub) > 0:
                bs.append(int(b))
                means.append(sub.mean())
                stds.append(sub.std())
        if bs:
            bs = np.array(bs)
            means = np.array(means)
            stds = np.array(stds)
            ax.plot(bs, means, marker="o", color=MODE_COLORS.get(mode, "#333"),
                    label=mode.capitalize(), linewidth=2)
            ax.fill_between(bs, means - stds, means + stds,
                            color=MODE_COLORS.get(mode, "#333"), alpha=0.15)
    ax.set_xlabel("Budget", fontsize=11)
    ax.set_ylabel("Mean Reasoning Tokens", fontsize=11)
    ax.set_title("Mean Reasoning Tokens ± Std", fontsize=12)
    ax.legend()

    # 3c: Per grid_size breakdown
    ax = axes[2]
    for gs in [5, 6, 7]:
        gs_df = df[df["gs"] == gs]
        if len(gs_df) < 5:
            continue
        means, bs = [], []
        for b in budgets_all:
            sub = gs_df[gs_df["reasoning_budget"] == b]["rtokens"].dropna()
            if len(sub) > 1:
                bs.append(int(b))
                means.append(sub.mean())
        if bs:
            ax.plot(bs, means, marker=GS_MARKERS[gs], color=GS_COLORS[gs],
                    label=f"grid={gs}", linewidth=1.5, markersize=6)
    ax.set_xlabel("Budget", fontsize=11)
    ax.set_ylabel("Mean Reasoning Tokens", fontsize=11)
    ax.set_title("Reasoning Tokens by Grid Size", fontsize=12)
    ax.legend()

    plt.suptitle("Reasoning Token Length Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = analysis_dir / "reasoning_token_stats.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # ========== Text Summary ==========
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    for b in budgets_all:
        sub = df[df["reasoning_budget"] == b]
        tokens = sub["rtokens"].dropna()
        truncated = sub["is_truncated"].mean() * 100
        acc = sub["correct"].dropna().mean() * 100
        mode = sub["mode"].iloc[0] if len(sub) > 0 else "?"
        print(f"  budget={b:>5} mode={mode:>12} | n={len(sub):>4} acc={acc:.1f}% "
              f"rtokens: median={tokens.median():.0f} mean={tokens.mean():.0f} "
              f"truncated={truncated:.1f}%")

    summary_path = analysis_dir / "analysis_summary.json"
    summary = {
        "total_results": len(df),
        "by_budget": {},
    }
    for b in budgets_all:
        sub = df[df["reasoning_budget"] == b]
        tokens = sub["rtokens"].dropna()
        summary["by_budget"][str(b)] = {
            "mode": sub["mode"].iloc[0] if len(sub) > 0 else "?",
            "n": len(sub),
            "accuracy": float(sub["correct"].dropna().mean()),
            "rtokens_mean": float(tokens.mean()) if not tokens.empty else None,
            "rtokens_median": float(tokens.median()) if not tokens.empty else None,
            "rtokens_std": float(tokens.std()) if not tokens.empty else None,
            "truncation_rate": float(sub["is_truncated"].mean()),
            "stage2_skip_rate": float(sub.get("stage2_skipped", pd.Series([False])).mean()),
        }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
