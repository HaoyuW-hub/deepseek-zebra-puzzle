"""
Results manager for inverse scaling evaluation.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class ResultsSummary:
    total_loaded_evaluations: int
    total_filtered_evaluations: int
    models: List[str]
    tasks: List[str]
    reasoning_budgets: List[int]
    timestamp: str
    model_stats: Dict[str, Dict[str, Any]]


@dataclass
class ModelStats:
    total_evaluations: int = 0
    correct_evaluations: int = 0
    total_cost: float = 0.0
    accuracy: float = 0.0


class FileIOUtils:
    @staticmethod
    def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
        if not filepath.exists():
            logger.debug(f"JSONL file not found: {filepath}")
            return []

        results = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        results.append(data)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Skipping malformed JSON on line {line_num} in {filepath}: {e}. "
                            f"Line content: '{line[:100]}{'...' if len(line) > 100 else ''}'"
                        )
                        continue

        except PermissionError:
            logger.error(f"Permission denied reading file: {filepath}")
            raise
        except UnicodeDecodeError as e:
            logger.error(f"Unicode decode error in file {filepath}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error reading file {filepath}: {e}")
            raise

        logger.debug(f"Successfully loaded {len(results)} records from {filepath}")
        return results

    @staticmethod
    def save_jsonl_line(filepath: Path, data: Dict[str, Any]) -> None:
        if not data:
            logger.warning("Attempted to save empty data, skipping")
            return

        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)

            processed_data = FileIOUtils._process_data_for_json(data)
            json_string = json.dumps(processed_data, ensure_ascii=False)

            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json_string + "\n")

            logger.debug(f"Successfully saved data to {filepath}")

        except TypeError as e:
            logger.error(f"JSON serialization error for file {filepath}: {e}")
            logger.debug(f"Problematic data: {data}")
            raise ValueError(f"Cannot serialize data to JSON: {e}")
        except PermissionError:
            logger.error(f"Permission denied writing to file: {filepath}")
            raise
        except OSError as e:
            logger.error(f"OS error writing to file {filepath}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error writing to file {filepath}: {e}")
            raise

    @staticmethod
    def _process_data_for_json(data: Dict[str, Any]) -> Dict[str, Any]:
        processed = {}
        for key, value in data.items():
            if hasattr(value, "model_dump"):
                processed[key] = value.model_dump()
            elif hasattr(value, "to_dict"):
                processed[key] = value.to_dict()
            elif isinstance(value, Path):
                processed[key] = str(value)
            elif str(type(value).__name__) == "StopReason":
                processed[key] = str(value)
            else:
                processed[key] = value
        return processed

    @staticmethod
    def save_json(filepath: Path, data: Dict[str, Any]) -> None:
        if not data:
            logger.warning("Attempted to save empty data as JSON, skipping")
            return

        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Successfully saved JSON to {filepath}")

        except TypeError as e:
            logger.error(f"JSON serialization error for file {filepath}: {e}")
            raise ValueError(f"Cannot serialize data to JSON: {e}")
        except PermissionError:
            logger.error(f"Permission denied writing to file: {filepath}")
            raise
        except OSError as e:
            logger.error(f"OS error writing to file {filepath}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving JSON to {filepath}: {e}")
            raise


class ResultsManager:
    def __init__(self, results_dir: Union[str, Path], previous_results_dir: Optional[Union[str, Path]] = None):
        self.results_dir = Path(results_dir)
        self.raw_dir = self.results_dir / "raw"
        self.analysis_dir = self.results_dir / "analysis"
        self.previous_results_dir = Path(previous_results_dir) if previous_results_dir else None
        self.console = console

        self.raw_dir.mkdir(exist_ok=True, parents=True)
        self.analysis_dir.mkdir(exist_ok=True, parents=True)

        if self.previous_results_dir:
            self.console.print(
                f"Results manager initialized with directory: [cyan]{self.results_dir}[/] (resuming from [cyan]{self.previous_results_dir}[/])",
                style="green",
            )
            self._consolidate_previous_results()
        else:
            self.console.print(
                f"Results manager initialized with directory: [cyan]{self.results_dir}[/]",
                style="green",
            )

    def save_result(
        self,
        model_id: str,
        task_id: str,
        instance_id: str,
        reasoning_budget: int,
        result: Dict[str, Any],
    ) -> None:
        if not model_id or not task_id:
            raise ValueError("model_id and task_id cannot be empty")

        required_keys = ["model", "task_id", "instance_id", "reasoning_budget"]
        missing_keys = [k for k in required_keys if k not in result]
        if missing_keys:
            raise ValueError(
                f"Result dict missing required keys {missing_keys}. "
                f"Required: {required_keys}. Found: {list(result.keys())}"
            )

        result_file = self.raw_dir / model_id / f"{task_id}.jsonl"

        try:
            FileIOUtils.save_jsonl_line(result_file, result)
            logger.debug(f"Saved result for {model_id}/{task_id}/{instance_id}")
        except Exception as e:
            logger.error(f"Failed to save result for {model_id}/{task_id}/{instance_id}: {e}")
            raise

    def _filter_error_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not results:
            return results

        filtered_results = []
        error_count = 0

        for result in results:
            has_error = result.get("error") is not None
            if has_error:
                error_count += 1
                logger.debug(f"Filtering out result with error: {result.get('error')}")
            else:
                filtered_results.append(result)

        if error_count > 0:
            logger.info(f"Filtered out {error_count} results with errors, {len(filtered_results)} valid results remaining")

        return filtered_results

    def _consolidate_previous_results(self) -> None:
        if not self.previous_results_dir:
            return

        prev_raw_dir = self.previous_results_dir / "raw"
        if not prev_raw_dir.exists():
            logger.info("No previous raw results directory found, skipping consolidation")
            return

        jsonl_files = list(prev_raw_dir.rglob("*.jsonl"))
        if not jsonl_files:
            logger.info("No previous result files found, skipping consolidation")
            return

        logger.info(f"Consolidating {len(jsonl_files)} result files from previous run...")

        consolidated_count = 0
        total_results = 0
        filtered_errors = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("Consolidating results...", total=len(jsonl_files))

            for prev_file in jsonl_files:
                model_id = prev_file.parent.name
                task_id = prev_file.stem

                progress.update(task, description=f"Consolidating {model_id}/{task_id}")

                try:
                    prev_results = FileIOUtils.load_jsonl(prev_file)
                    if not prev_results:
                        progress.advance(task)
                        continue

                    valid_results = self._filter_error_results(prev_results)
                    total_results += len(prev_results)
                    filtered_errors += len(prev_results) - len(valid_results)

                    if not valid_results:
                        progress.advance(task)
                        continue

                    current_file = self.raw_dir / model_id / f"{task_id}.jsonl"
                    current_file.parent.mkdir(parents=True, exist_ok=True)

                    existing_results = []
                    if current_file.exists():
                        existing_results = FileIOUtils.load_jsonl(current_file)

                    existing_ids = set()
                    for result in existing_results:
                        result_id = (
                            result.get("instance_id"),
                            result.get("reasoning_budget"),
                            result.get("icl_enabled", False),
                            result.get("icl_num_examples", 0)
                        )
                        existing_ids.add(result_id)

                    new_results = []
                    for result in valid_results:
                        result_id = (
                            result.get("instance_id"),
                            result.get("reasoning_budget"),
                            result.get("icl_enabled", False),
                            result.get("icl_num_examples", 0)
                        )
                        if result_id not in existing_ids:
                            new_results.append(result)

                    if new_results:
                        for result in new_results:
                            FileIOUtils.save_jsonl_line(current_file, result)
                        consolidated_count += len(new_results)
                        logger.debug(f"Consolidated {len(new_results)} new results for {model_id}/{task_id}")

                except Exception as e:
                    logger.error(f"Error consolidating {prev_file}: {e}")
                    self.console.print(
                        f"Error consolidating [cyan]{prev_file}[/]: {str(e)}",
                        style="red",
                    )

                progress.advance(task)

        self.console.print(
            f"Consolidation complete: {consolidated_count} results consolidated, "
            f"{filtered_errors} error results filtered out from {total_results} total",
            style="green"
        )

    def load_raw_results_for_task(self, model_id: str, task_id: str) -> List[Dict[str, Any]]:
        jsonl_file = self.raw_dir / model_id / f"{task_id}.jsonl"
        logger.info(f"Loading raw results from: {jsonl_file}")
        if not jsonl_file.is_file():
            logger.info(f"Raw results file not found: {jsonl_file}")
            return []
        raw_results = FileIOUtils.load_jsonl(jsonl_file)
        return self._filter_error_results(raw_results)

    def check_result_exists(
        self,
        existing_results: List[Dict[str, Any]],
        instance_id: str,
        reasoning_budget: int,
        icl_enabled: bool = False,
        icl_num_examples: int = 0
    ) -> bool:
        if not existing_results:
            logger.debug("No existing results to check")
            return False

        logger.debug(
            f"Checking for result: instance_id='{instance_id}', "
            f"reasoning_budget={reasoning_budget}, icl_enabled={icl_enabled}, "
            f"icl_num_examples={icl_num_examples} in {len(existing_results)} results"
        )

        for result in existing_results:
            try:
                res_instance_id = result.get("instance_id")
                res_budget = result.get("reasoning_budget")
                res_icl_enabled = result.get("icl_enabled", False)
                res_icl_num_examples = result.get("icl_num_examples", 0)
                has_error = result.get("error") is not None

                if (
                    res_instance_id == instance_id
                    and res_budget == reasoning_budget
                    and res_icl_enabled == icl_enabled
                    and res_icl_num_examples == icl_num_examples
                ):
                    if has_error:
                        logger.debug("Found matching result but it has an error, treating as non-existent")
                        return False
                    else:
                        logger.debug("Found valid matching result")
                        return True

            except Exception as e:
                logger.warning(f"Error checking result entry: {e}")
                continue

        logger.debug("No matching valid result found")
        return False

    def load_all_results(self) -> List[Dict[str, Any]]:
        all_results = []

        if self.previous_results_dir:
            prev_results = self._load_results_from_directory(
                self.previous_results_dir / "raw", "Loading previous results"
            )
            all_results.extend(prev_results)

        current_results = self._load_results_from_directory(
            self.raw_dir, "Loading current results"
        )
        all_results.extend(current_results)

        self._display_results_summary(all_results, "All Results")
        return all_results

    def _load_results_from_directory(
        self, directory: Path, description: str
    ) -> List[Dict[str, Any]]:
        if not directory.exists():
            logger.debug(f"Directory does not exist: {directory}")
            return []

        jsonl_files = list(directory.rglob("*.jsonl"))
        if not jsonl_files:
            logger.debug(f"No JSONL files found in {directory}")
            return []

        logger.info(f"Found {len(jsonl_files)} JSONL files in {directory}")

        all_results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(description, total=len(jsonl_files))

            for result_file in jsonl_files:
                model_id = result_file.parent.name
                task_id = result_file.stem

                progress.update(
                    task,
                    description=f"{description}: {model_id}/{task_id}"
                )

                try:
                    results = FileIOUtils.load_jsonl(result_file)
                    all_results.extend(results)
                    logger.debug(f"Loaded {len(results)} results from {result_file}")
                except Exception as e:
                    self.console.print(
                        f"Error loading {result_file}: {str(e)}",
                        style="red",
                    )

                progress.advance(task)

        logger.info(f"Loaded {len(all_results)} total results from {directory}")
        return all_results

    def summarize_results(
        self, models: List[str], tasks: List[str], reasoning_budgets: List[int]
    ) -> Tuple[ResultsSummary, List[Dict[str, Any]]]:
        logger.info("Generating results summary...")

        all_results = self._load_target_files(models, tasks)
        filtered_results = self._filter_results(all_results, models, tasks, reasoning_budgets)
        model_stats = self._calculate_model_stats(filtered_results)

        summary = ResultsSummary(
            total_loaded_evaluations=len(all_results),
            total_filtered_evaluations=len(filtered_results),
            models=models,
            tasks=tasks,
            reasoning_budgets=reasoning_budgets,
            timestamp=datetime.now().isoformat(),
            model_stats=model_stats
        )

        self._display_summary_table(model_stats)

        return summary, filtered_results

    def _load_target_files(self, models: List[str], tasks: List[str]) -> List[Dict[str, Any]]:
        target_files = []
        for model_id in models:
            for task_id in tasks:
                jsonl_file = self.raw_dir / model_id / f"{task_id}.jsonl"
                if jsonl_file.is_file():
                    target_files.append((model_id, task_id, jsonl_file))

        logger.info(f"Identified {len(target_files)} result files to load.")

        all_results = []
        if not target_files:
            logger.warning("No result files found for the specified criteria.")
            return all_results

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            load_task = progress.add_task("Loading results...", total=len(target_files))

            for model_id, task_id, jsonl_file in target_files:
                progress.update(load_task, description=f"Loading {model_id}/{task_id}")
                results = FileIOUtils.load_jsonl(jsonl_file)
                all_results.extend(results)
                progress.advance(load_task)

        logger.info(f"Loaded {len(all_results)} results from {len(target_files)} files.")
        return all_results

    def _filter_results(
        self,
        all_results: List[Dict[str, Any]],
        models: List[str],
        tasks: List[str],
        reasoning_budgets: List[int]
    ) -> List[Dict[str, Any]]:
        filtered = [
            result for result in all_results
            if (result.get("model") in models and
                result.get("task_id") in tasks and
                result.get("reasoning_budget") in reasoning_budgets)
        ]
        logger.info(f"Filtered to {len(filtered)} relevant results.")
        return filtered

    def _calculate_model_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        model_stats = defaultdict(lambda: ModelStats().__dict__)

        for result in results:
            model_id = result.get("model")
            if not model_id:
                continue

            stats = model_stats[model_id]
            stats["total_evaluations"] += 1
            stats["total_cost"] += result.get("cost", 0) or 0
            if result.get("correct", False):
                stats["correct_evaluations"] += 1

        for stats in model_stats.values():
            total = stats["total_evaluations"]
            stats["accuracy"] = stats["correct_evaluations"] / total if total > 0 else 0.0

        return dict(model_stats)

    def _display_summary_table(self, model_stats: Dict[str, Dict[str, Any]]) -> None:
        table = Table(title="Results Summary", show_header=True)
        table.add_column("Model", style="cyan")
        table.add_column("Evaluations", justify="right", style="green")
        table.add_column("Accuracy", justify="right", style="yellow")
        table.add_column("Cost ($)", justify="right", style="blue")

        for model_id, stats in model_stats.items():
            table.add_row(
                model_id,
                str(stats["total_evaluations"]),
                f"{stats['accuracy']*100:.1f}%",
                f"{stats['total_cost']:.4f}",
            )

        self.console.print(table)

    def save_summary(self, summary: Union[ResultsSummary, Dict[str, Any]]) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.analysis_dir / f"summary_{timestamp}.json"

        if isinstance(summary, ResultsSummary):
            summary_dict = {
                "total_loaded_evaluations": summary.total_loaded_evaluations,
                "total_filtered_evaluations": summary.total_filtered_evaluations,
                "models": summary.models,
                "tasks": summary.tasks,
                "reasoning_budgets": summary.reasoning_budgets,
                "timestamp": summary.timestamp,
                "model_stats": summary.model_stats
            }
        else:
            summary_dict = summary

        FileIOUtils.save_json(summary_file, summary_dict)
        self.console.print(
            f"Saved summary to [cyan]{summary_file}[/]", style="green"
        )
        return summary_file

    def create_dataframe(
        self, results_list: Optional[List[Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        if results_list is not None:
            logger.info("Creating DataFrame from provided results list...")
            results = results_list
        else:
            logger.info("No results list provided, loading all results from disk...")
            results = self.load_all_results()

        if not results:
            logger.warning("Attempting to create DataFrame from empty results list.")
            return pd.DataFrame([])

        df = pd.DataFrame(results)

        with pd.option_context("display.max_rows", 10):
            self.console.print("\n[bold cyan]DataFrame Preview:[/]")
            self.console.print(df.head())
            self.console.print("\n[bold cyan]DataFrame Info:[/]")
            df.info(buf=self.console.file)

        return df

    def _display_results_summary(self, results: List[Dict[str, Any]], title: str) -> None:
        table = Table(title=f"{title}", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Total Results", str(len(results)))

        if results:
            unique_models = {r.get("model") for r in results if r.get("model")}
            unique_tasks = {r.get("task_id") for r in results if r.get("task_id")}

            if len(unique_models) > 1:
                table.add_row("Unique Models", str(len(unique_models)))
            if len(unique_tasks) > 1:
                table.add_row("Unique Tasks", str(len(unique_tasks)))

            correct_count = sum(1 for r in results if r.get("correct", False))
            success_rate = f"{(correct_count / len(results)) * 100:.1f}%"
            table.add_row("Success Rate", success_rate)

        self.console.print(table)
