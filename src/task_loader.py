"""
Task loader for inverse scaling evaluation tasks.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import ast

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


class TaskLoader:
    """Loader for inverse scaling evaluation tasks."""

    def __init__(self, tasks_config: Dict[str, Any]):
        self.tasks_config = tasks_config
        self.tasks_data = {}
        self.console = console

        table = Table(title="Available Tasks", show_header=True)
        table.add_column("Task ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Category", style="yellow")

        for task_id, config in tasks_config.items():
            table.add_row(
                task_id, config.get("name", task_id), config.get("category", "N/A")
            )

        self.console.print(table)
        self.console.print("Task loader initialized successfully", style="green")

    def load_task(self, task_id: str, external_progress=None) -> List[Dict[str, Any]]:
        if task_id in self.tasks_data:
            return self.tasks_data[task_id]

        if task_id not in self.tasks_config:
            self.console.print(
                f"Task '{task_id}' not found in tasks configuration", style="red"
            )
            raise ValueError(f"Task '{task_id}' not found in tasks configuration")

        task_config = self.tasks_config[task_id]
        task_file = Path(task_config["file_path"])

        if not task_file.exists():
            self.console.print(f"Task file '{task_file}' not found", style="red")
            raise FileNotFoundError(f"Task file '{task_file}' not found")

        self.console.print(
            f"Loading task [cyan]{task_id}[/] from [yellow]{task_file}[/]"
        )

        task_data = []
        line_count = sum(1 for _ in open(task_file))

        if external_progress is None:
            progress_context = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=self.console,
            )
        else:
            progress_context = external_progress

        with progress_context as progress:
            if external_progress is None:
                task = progress.add_task(
                    f"Loading instances for {task_id}", total=line_count
                )

            with open(task_file, "r") as f:
                for i, line in enumerate(f):
                    try:
                        instance = json.loads(line)
                        has_prompt = "prompt" in instance
                        has_mcq_fields = ("classes" in instance and "answer_index" in instance)
                        has_open_ended_field = "answer" in instance

                        if not has_prompt or not (has_mcq_fields or has_open_ended_field):
                            missing_fields = []
                            if not has_prompt: missing_fields.append('prompt')
                            if not has_mcq_fields and not has_open_ended_field:
                                missing_fields.append("('classes' & 'answer_index') or 'answer'")

                            self.console.print(
                                f"Missing required fields ({', '.join(missing_fields)}) in task [cyan]{task_id}[/], instance {i}. Skipping.",
                                style="yellow",
                            )
                            continue

                        if "classes" in instance and isinstance(instance["classes"], str):
                            try:
                                evaluated_classes = ast.literal_eval(
                                    instance["classes"]
                                )
                                if not isinstance(evaluated_classes, list):
                                    raise ValueError("Parsed 'classes' is not a list")
                                instance["classes"] = [
                                    c.strip() for c in evaluated_classes
                                ]
                            except (
                                ValueError,
                                SyntaxError,
                                TypeError,
                            ) as e:
                                self.console.print(
                                    f"Could not parse 'classes' field as a list literal in task [cyan]{task_id}[/], instance {i}: {e}. String was: {instance['classes'][:100]}... Skipping.",
                                    style="yellow",
                                )
                                continue

                        instance["id"] = f"{task_id}_{i}"
                        instance["task_id"] = task_id
                        instance["metric"] = task_config.get("metric", "accuracy")
                        instance["task_metadata"] = {
                            "name": task_config.get("name", task_id),
                            "description": task_config.get("description", ""),
                            "category": task_config.get("category", ""),
                        }

                        task_data.append(instance)
                    except json.JSONDecodeError:
                        self.console.print(
                            f"Error parsing JSON in task [cyan]{task_id}[/], line {i}. Skipping.",
                            style="yellow",
                        )
                    except Exception as e:
                        self.console.print(
                            f"Error processing task [cyan]{task_id}[/], line {i}: {e}. Skipping.",
                            style="yellow",
                        )

                    if external_progress is None:
                        progress.advance(task)

        table = Table(title=f"Task Summary: {task_id}", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Total Instances", str(len(task_data)))
        table.add_row("Name", task_config.get("name", task_id))
        table.add_row("Category", task_config.get("category", "N/A"))
        if task_config.get("description"):
            table.add_row("Description", task_config["description"])
        self.console.print(table)

        self.console.print(
            f"Successfully loaded [cyan]{len(task_data)}[/] instances for task [green]{task_id}[/]",
            style="green",
        )
        self.tasks_data[task_id] = task_data
        return task_data

    def get_task_data(self, task_id: str, external_progress=None) -> Optional[List[Dict[str, Any]]]:
        if task_id in self.tasks_data:
            return self.tasks_data[task_id]

        try:
            return self.load_task(task_id, external_progress=external_progress)
        except (ValueError, FileNotFoundError) as e:
            logger.error(f"Failed to load task data for '{task_id}': {e}")
            return None

    def get_task_instance(
        self, task_id: str, instance_idx: int
    ) -> Optional[Dict[str, Any]]:
        task_data = self.load_task(task_id)
        if 0 <= instance_idx < len(task_data):
            return task_data[instance_idx]
        logger.warning(
            f"Instance index [yellow]{instance_idx}[/] out of range for task [cyan]{task_id}[/]"
        )
        return None

    def get_all_task_ids(self) -> List[str]:
        return list(self.tasks_config.keys())

    def get_task_metadata(self, task_id: str) -> Dict[str, Any]:
        if task_id not in self.tasks_config:
            logger.error(f"Task '{task_id}' not found in tasks configuration")
            raise ValueError(f"Task '{task_id}' not found in tasks configuration")

        task_config = self.tasks_config[task_id]
        metadata = {
            "name": task_config.get("name", task_id),
            "description": task_config.get("description", ""),
            "category": task_config.get("category", ""),
            "num_instances": len(self.load_task(task_id)),
        }

        table = Table(title=f"Task Metadata: {task_id}", show_header=True)
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")
        for key, value in metadata.items():
            table.add_row(key.title(), str(value))
        self.console.print(table)

        return metadata
