#!/usr/bin/env python
"""
Run DeepSeek API evaluation on Zebra Puzzle task.
Simplified entry point without Hydra/WandB/VLLM dependencies.
"""

import argparse
import asyncio
import logging
import os
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from src.model_interface import ModelInterface
from src.evaluator import Evaluator
from src.results_manager import ResultsManager
from src.task_loader import TaskLoader
from src.utils.analysis import analyze_inverse_scaling
from src.utils.plotting import (
    plot_token_scaling_curves_improved,
    plot_token_slopes,
    plot_budget_length_boxplot,
    plot_token_correlations,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


def load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DeepSeek API evaluation on Zebra Puzzle task"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="config/model/deepseek_r1.yaml",
        help="Path to model config YAML",
    )
    parser.add_argument(
        "--task-config",
        type=str,
        default="config/task/zebra_puzzle.yaml",
        help="Path to task config YAML",
    )
    parser.add_argument(
        "--reasoning-budgets",
        type=str,
        default="0,1024,2048,4096,8192",
        help="Comma-separated reasoning budgets (0=no thinking, >0=interrupted, <0=natural)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: results/<timestamp>)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable response caching",
    )
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip analysis and plotting after evaluation",
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="Run in validation mode (sample instances)",
    )
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=20,
        help="Number of instances to sample in validation mode",
    )
    parser.add_argument(
        "--validation-runs",
        type=int,
        default=3,
        help="Number of runs per sampled instance in validation mode",
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=42,
        help="Random seed for validation sampling",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds for each budget run (e.g. '42,123,456,42,123,456')",
    )
    parser.add_argument(
        "--icl-shot-count",
        type=int,
        default=None,
        help="Enable In-Context Learning with k examples per prompt",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Max concurrent API requests",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default="bbeh_zebra_puzzles",
        help="Task ID to evaluate",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from a previous results directory",
    )
    return parser.parse_args()


async def run_async(
    models_config: dict,
    tasks_config: dict,
    models_to_run: List[str],
    tasks_to_run: List[str],
    reasoning_budgets: List[int],
    output_dir: Path,
    use_cache: bool,
    icl_config: Optional[dict],
    seeds: Optional[List[int]],
    validation_mode: bool,
    validation_samples: int,
    validation_runs: int,
    validation_seed: int,
    resume_from: Optional[str],
):
    task_loader = TaskLoader(tasks_config)

    model_cfg = next(iter(models_config.values()), {})
    two_stage_config = model_cfg.get("two_stage", {})
    evaluation_config = {
        "thinking": True,
        "prompt_use_all_budget": True,
        "stage2_max_tokens": two_stage_config.get("stage2_max_tokens", 4096),
        "time_up_phrase": two_stage_config.get("time_up_phrase", "Time's Up! Your reasoning was interrupted after {budget} tokens."),
    }

    model_interface = ModelInterface(
        models_config,
        use_cache=use_cache,
        evaluation_config=evaluation_config,
    )

    previous_results_dir = Path(resume_from) if resume_from else None
    results_manager = ResultsManager(output_dir, previous_results_dir)
    evaluator = Evaluator(task_loader, model_interface, results_manager)

    validation_cfg = {
        "enabled": validation_mode,
        "samples": validation_samples,
        "runs": validation_runs,
        "seed": validation_seed,
    } if validation_mode else None

    logger.info(f"Starting evaluation: {len(models_to_run)} models, {len(tasks_to_run)} tasks, {len(reasoning_budgets)} budgets")

    summary, loaded_results_list = await evaluator.run_evaluations(
        models=models_to_run,
        tasks=tasks_to_run,
        reasoning_budgets=reasoning_budgets,
        validation_mode=validation_mode,
        validation_samples=validation_samples,
        validation_runs=validation_runs,
        validation_seed=validation_seed,
        icl_config=icl_config,
        seeds=seeds,
    )

    summary_path = results_manager.save_summary(summary)
    logger.info(f"Summary saved to {summary_path}")

    logger.info("Creating DataFrame...")
    df = results_manager.create_dataframe(results_list=loaded_results_list)
    df_path = output_dir / "results_df.csv"
    df.to_csv(df_path, index=False)
    logger.info(f"DataFrame saved to {df_path}")

    return results_manager, df, models_to_run, tasks_to_run, reasoning_budgets, task_loader


def main():
    args = parse_args()

    # Load model config
    raw_model_config = load_yaml_config(args.model_config)
    model_id = raw_model_config["id"]
    # Set concurrency from CLI
    raw_model_config["concurrency_limit"] = args.concurrency
    models_config = {model_id: raw_model_config}

    # Load task config
    raw_task_config = load_yaml_config(args.task_config)
    tasks_config = raw_task_config.get("task_definitions", {})

    # Parse reasoning budgets
    reasoning_budgets = [
        int(b.strip()) for b in args.reasoning_budgets.split(",")
    ]

    models_to_run = [model_id]
    tasks_to_run = [args.task_id]

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / f"{model_id}__{args.task_id}" / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Setup ICL config
    icl_config = None
    if args.icl_shot_count is not None:
        icl_config = {
            "enabled": True,
            "num_examples": args.icl_shot_count,
        }
        logger.info(f"Using {args.icl_shot_count}-shot In-Context Learning")

    # Parse seeds
    seeds = None
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # Validate API key
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("DEEPSEEK_API_KEY not set in environment or .env file!")
        sys.exit(1)

    logger.info(f"Model: {model_id} ({raw_model_config.get('name', 'N/A')})")
    logger.info(f"Task: {args.task_id}")
    logger.info(f"Reasoning budgets: {reasoning_budgets}")
    logger.info(f"Concurrency: {args.concurrency}")
    logger.info(f"Cache: {'enabled' if not args.no_cache else 'disabled'}")
    logger.info("=" * 60)

    # Run evaluation
    results_manager, df, models_to_run, tasks_to_run, reasoning_budgets, task_loader = asyncio.run(
        run_async(
            models_config=models_config,
            tasks_config=tasks_config,
            models_to_run=models_to_run,
            tasks_to_run=tasks_to_run,
            reasoning_budgets=reasoning_budgets,
            output_dir=output_dir,
            use_cache=not args.no_cache,
            icl_config=icl_config,
            seeds=seeds,
            validation_mode=args.validation,
            validation_samples=args.validation_samples,
            validation_runs=args.validation_runs,
            validation_seed=args.validation_seed,
            resume_from=args.resume_from,
        )
    )

    # Run analysis
    if not args.no_analysis and len(df) > 0:
        logger.info("=" * 60)
        logger.info("Running analysis...")
        analysis_dir = output_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        plot_dir = analysis_dir / "plots"
        plot_dir.mkdir(exist_ok=True)

        try:
            available_metrics = []
            if 'correct' in df.columns and df['correct'].notna().any():
                available_metrics.append('accuracy')
            if 'squared_error' in df.columns and df['squared_error'].notna().any():
                available_metrics.append('mse')
            if 'relative_error' in df.columns and df['relative_error'].notna().any():
                available_metrics.append('relative_error')
            logger.info(f"Metrics found in results: {available_metrics}")

            for metric_type in available_metrics:
                logger.info(f"Generating {metric_type.upper()} plots...")

                logger.info(f"... {metric_type} vs. token scaling curves...")
                plot_token_scaling_curves_improved(
                    df,
                    task_loader,
                    models_to_run,
                    tasks_to_run,
                    plot_dir,
                    plot_type=metric_type,
                    min_samples_per_point=1,
                )

                logger.info(f"... {metric_type} token slope...")
                plot_token_slopes(
                    df,
                    models_to_run,
                    tasks_to_run,
                    plot_dir,
                    plot_type=metric_type,
                )

            logger.info("Generating budget vs length boxplot...")
            plot_budget_length_boxplot(df, plot_dir, task_loader=task_loader)

            # Generate token correlation plots per model-task pair
            for model in models_to_run:
                for task in tasks_to_run:
                    try:
                        plot_token_correlations(df, model, task, plot_dir, metric="accuracy")
                    except Exception as e:
                        logger.warning(f"Could not generate token correlation plot for {model}/{task}: {e}")

            # Run statistical analysis
            analyze_inverse_scaling(
                df, models_to_run, tasks_to_run, reasoning_budgets, output_dir=analysis_dir
            )

            logger.info(f"Analysis complete. Plots saved to {plot_dir}")

        except Exception as e:
            logger.error(f"Error during analysis: {e}", exc_info=True)

    logger.info("=" * 60)
    logger.info(f"Experiment finished. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
