"""
Interface for accessing DeepSeek API through the safetytooling APIs.
"""

import asyncio
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
    BarColumn, MofNCompleteColumn, TimeRemainingColumn,
)
from safetytooling.apis import InferenceAPI
from safetytooling.data_models import Prompt
from safetytooling.utils import utils

from src.utils.model_helpers import (
    _calculate_max_tokens,
    _extract_answer_tag,
    _get_stage1_interrupted_prompt,
    _get_stage1_natural_prompt,
    _prepare_prompt,
    _prepare_time_up_prompt,
    import_time,
)

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_BACKOFF = 1.0
DEFAULT_MAX_BACKOFF = 60.0
DEFAULT_CONCURRENCY_LIMIT = 20


class ModelInterface:
    """Interface for accessing DeepSeek reasoning model via API."""

    def __init__(
        self,
        models_config: Dict,
        use_cache: bool = True,
        anthropic_api_key_tag: str = "ANTHROPIC_API_KEY",
        evaluation_config: Optional[Dict[str, Any]] = None,
    ):
        self.models_config = models_config
        self.anthropic_api_key_tag = anthropic_api_key_tag
        self.console = console
        self.evaluation_config = evaluation_config or {}

        self.cache_dir = Path.home() / ".cache" / "deepseek-zebra-eval"
        self.prompt_history_dir = (
            Path.home() / ".prompt_history" / "deepseek-zebra-eval"
        )
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.prompt_history_dir.mkdir(exist_ok=True, parents=True)

        utils.setup_environment(anthropic_tag=anthropic_api_key_tag)

        self.api = InferenceAPI(
            cache_dir=self.cache_dir if use_cache else None,
            prompt_history_dir=self.prompt_history_dir,
            print_prompt_and_response=False,
            empty_completion_threshold=1.0,
        )

        self.console.print("Model interface initialized successfully", style="green")

    async def _single_api_call(
        self,
        model_id: str,
        model_config: Dict,
        prompt: Prompt,
        max_tokens: int,
        reasoning_budget: int,
        reasoning_effort: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Make a single API call with retry logic. Returns processed result dict."""
        kwargs = {**model_config.get("api_params", {})}

        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        if model_config.get("type", None) == "openai":
            kwargs["max_completion_tokens"] = max_tokens
            api_max_tokens = max_tokens
        else:
            api_max_tokens = max_tokens

        max_retries = model_config.get("max_retries", DEFAULT_MAX_RETRIES)
        current_backoff = model_config.get("initial_backoff", DEFAULT_INITIAL_BACKOFF)
        max_backoff = model_config.get("max_backoff", DEFAULT_MAX_BACKOFF)

        for attempt in range(max_retries):
            try:
                logger.debug(f"Attempt {attempt+1}/{max_retries} for {model_id} (budget: {reasoning_budget})")
                responses = await self.api(
                    model_id=model_config["model_name"],
                    prompt=prompt,
                    temperature=model_config.get("temperature", 0.0),
                    max_tokens=api_max_tokens,
                    **kwargs,
                )

                response = responses[0] if responses else None
                processed_result = await self.process_response(response, model_id, reasoning_budget, import_time())

                if processed_result.get("error") is None:
                    return processed_result

                logger.warning(f"Processing error on attempt {attempt+1} for {model_id}: {processed_result['error']}. Retrying...")

            except Exception as e:
                logger.warning(f"Exception on attempt {attempt + 1} for {model_id}: {str(e)}. Retrying...")

            if attempt < max_retries - 1:
                jitter = random.uniform(0, current_backoff * 0.1)
                sleep_time = current_backoff + jitter
                logger.info(f"Sleeping for {sleep_time:.2f} seconds before retry {attempt+2}")
                await asyncio.sleep(sleep_time)
                current_backoff = min(current_backoff * 2, max_backoff)

        logger.error(f"Max retries ({max_retries}) reached for {model_id}.")
        return {
            "model": model_id,
            "reasoning_budget": reasoning_budget,
            "latency": 0,
            "error": f"Max retries ({max_retries}) reached",
            "response": None,
            "extracted_answer": None,
            "cost": 0,
            "reasoning_content": None,
        }

    async def evaluate_prompt(
        self,
        model_id: str,
        prompt_text: str,
        possible_answers: List[str],
        reasoning_budget: int = 0,
        icl_examples: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if model_id not in self.models_config:
            logger.error(f"Model ID '{model_id}' not found in configuration.")
            raise ValueError(f"Invalid model_id: {model_id}")

        return await self._evaluate_prompt_two_stage(
            model_id, prompt_text, possible_answers, reasoning_budget, icl_examples
        )

    async def _evaluate_prompt_two_stage(
        self,
        model_id: str,
        prompt_text: str,
        possible_answers: List[str],
        reasoning_budget: int,
        icl_examples: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Two-stage 'Time's Up!' early stopping.

        budget > 0: interrupted mode — model is warned it will be cut off.
        budget < 0: natural mode — model reasons naturally, truncated without warning.
        budget = 0: no-thinking mode."""
        model_config = self.models_config[model_id]
        is_multiple_choice = bool(possible_answers)
        base_system_prompt = model_config.get("system_prompt", "")
        stage2_max_tokens = self.evaluation_config.get("stage2_max_tokens", 4096)

        is_natural = reasoning_budget < 0
        abs_budget = abs(reasoning_budget) if is_natural else reasoning_budget

        # --- Stage 1 prompt ---
        if is_natural:
            stage1_system_prompt = _get_stage1_natural_prompt(abs_budget, is_multiple_choice)
        else:
            stage1_system_prompt = _get_stage1_interrupted_prompt(abs_budget, is_multiple_choice)
        full_system_prompt = base_system_prompt + stage1_system_prompt

        stage1_prompt = _prepare_prompt(
            prompt_text,
            possible_answers if possible_answers else [],
            model_id,
            full_system_prompt,
            icl_examples=icl_examples,
            models_config=self.models_config,
        )

        BASE_TOKENS = 16384
        if is_natural:
            stage1_max_tokens = BASE_TOKENS + (abs_budget - 1) * 4096
        else:
            stage1_max_tokens = BASE_TOKENS

        reasoning_effort = "high"

        stage1_start = import_time()
        stage1_result = await self._single_api_call(
            model_id, model_config, stage1_prompt, stage1_max_tokens, reasoning_budget,
            reasoning_effort=reasoning_effort,
        )
        stage1_latency = import_time() - stage1_start

        stage1_completion = stage1_result.get("response")
        stage1_reasoning = stage1_result.get("reasoning_content") or ""

        # Check if Stage 1 already produced a valid answer
        stage1_extracted = _extract_answer_tag(stage1_completion) if stage1_completion else None
        if stage1_extracted is not None:
            return {
                "model": model_id,
                "reasoning_budget": reasoning_budget,
                "mode": "natural" if is_natural else "interrupted",
                "response": stage1_completion,
                "extracted_answer": stage1_extracted,
                "latency": stage1_latency,
                "cost": stage1_result.get("cost", 0),
                "error": None,
                "reasoning_content": stage1_reasoning,
                "stop_reason": stage1_result.get("stop_reason"),
                "usage": stage1_result.get("usage"),
                "two_stage": True,
                "stage2_skipped": True,
                "stage1_max_tokens": stage1_max_tokens,
            }

        # --- Stage 2: Time's Up! ---
        stage2_prompt = _prepare_time_up_prompt(
            truncated_reasoning=stage1_reasoning,
            original_question=prompt_text,
            possible_answers=possible_answers if possible_answers else [],
            budget=abs_budget,
            base_system_prompt=base_system_prompt,
        )

        stage2_start = import_time()
        stage2_result = await self._single_api_call(
            model_id, model_config, stage2_prompt, stage2_max_tokens, reasoning_budget,
            reasoning_effort=reasoning_effort,
        )
        stage2_latency = import_time() - stage2_start

        stage2_completion = stage2_result.get("response")
        if stage2_completion and not stage2_completion.startswith("<answer>"):
            stage2_completion = f"<answer>{stage2_completion}"
        stage2_extracted = _extract_answer_tag(stage2_completion) if stage2_completion else None

        total_latency = stage1_latency + stage2_latency
        total_cost = stage1_result.get("cost", 0) + stage2_result.get("cost", 0)

        error = None
        if stage2_result.get("error"):
            error = f"Stage 2 failed: {stage2_result['error']}"
        elif stage2_extracted is None:
            error = "Stage 2 produced no valid answer"

        result = {
            "model": model_id,
            "reasoning_budget": reasoning_budget,
            "mode": "natural" if is_natural else "interrupted",
            "response": stage2_completion,
            "extracted_answer": stage2_extracted,
            "latency": total_latency,
            "cost": total_cost,
            "error": error,
            "reasoning_content": stage1_reasoning,
            "stop_reason": stage2_result.get("stop_reason"),
            "usage": None,
            "two_stage": True,
            "stage2_skipped": False,
            "stage1_max_tokens": stage1_max_tokens,
            "stage2_max_tokens": stage2_max_tokens,
        }

        if result["error"]:
            logger.error(f"Two-stage evaluation failed for {model_id} (budget: {reasoning_budget}): {result['error']}")
        else:
            logger.debug(f"Two-stage evaluation succeeded for {model_id} (budget: {reasoning_budget})")
            if result['extracted_answer']:
                logger.debug(f"Answer: {result['extracted_answer']}")

        return result

    async def process_response(
        self,
        response,
        model_id: str,
        reasoning_budget: int,
        start_time: float,
    ) -> Dict[str, Any]:
        end_time = import_time()
        latency = end_time - start_time

        if response is None:
            logger.warning(
                f"Received None response object for budget {reasoning_budget}."
            )
            return {
                "model": model_id,
                "reasoning_budget": reasoning_budget,
                "latency": latency,
                "error": "No response received from API",
                "response": None,
                "extracted_answer": None,
                "cost": 0,
                "reasoning_content": None,
            }

        result = {
            "model": getattr(response, "model_id", model_id),
            "reasoning_budget": reasoning_budget,
            "latency": getattr(response, "duration", latency),
            "error": None,
            "response": None,
            "extracted_answer": None,
            "cost": getattr(response, "cost", 0),
            "stop_reason": getattr(response, "stop_reason", None),
            "reasoning_content": getattr(response, "reasoning_content", None),
            "usage": getattr(response, "usage", None),
        }
        # Extract token counts from usage dict (DeepSeek stores them there)
        usage = result["usage"]
        if isinstance(usage, dict):
            result["input_tokens"] = usage.get("input_tokens")
            result["output_tokens"] = usage.get("output_tokens")
            result["total_tokens"] = usage.get("total_tokens")
        else:
            result["input_tokens"] = getattr(response, "input_tokens", None)
            result["output_tokens"] = getattr(response, "output_tokens", None)
            result["total_tokens"] = getattr(response, "total_tokens", None)

        if hasattr(response, "error") and response.error:
            result["error"] = response.error
            return result

        completion_text = getattr(response, "completion", None)
        if completion_text is None:
            result["error"] = "No completion text found in response"
        elif not isinstance(completion_text, str):
            try:
                if hasattr(completion_text, "text"):
                    completion_text = completion_text.text
                elif hasattr(completion_text, "thinking"):
                    completion_text = str(completion_text.thinking)
                    result["reasoning_content"] = completion_text
                else:
                    result["error"] = f"Completion is not a string and has no text/thinking attribute: {type(completion_text)}"
                    completion_text = str(completion_text)
            except Exception as e:
                result["error"] = f"Error processing completion object: {str(e)}"
                completion_text = None
        else:
            completion_text = completion_text.strip()

        result["response"] = completion_text

        if isinstance(completion_text, str):
            if completion_text:
                result["extracted_answer"] = _extract_answer_tag(completion_text)
        elif completion_text is not None:
            logger.warning(
                f"Completion content is not a string: {type(completion_text)}. Cannot extract answer."
            )

        return result

    async def _evaluate_prompt_with_semaphore(
        self, semaphore: asyncio.Semaphore, model_id: str, prompt_text: str,
        possible_answers: List[str], reasoning_budget: int,
        icl_examples: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        async with semaphore:
            return await self.evaluate_prompt(
                model_id, prompt_text, possible_answers,
                reasoning_budget, icl_examples
            )

    async def evaluate_prompts_batch(
        self,
        model_id: str,
        prompt_texts: List[str],
        possible_answers_list: List[List[str]],
        reasoning_budgets: List[int],
        external_progress: Optional[Progress] = None,
        icl_examples_list: Optional[List[Optional[List[Dict[str, Any]]]]] = None,
    ) -> List[Dict[str, Any]]:
        if not (len(prompt_texts) == len(possible_answers_list) == len(reasoning_budgets)):
            raise ValueError("Input lists must have the same length")

        if model_id not in self.models_config:
             raise ValueError(f"Invalid model_id: {model_id}")
        model_config = self.models_config[model_id]

        if icl_examples_list is not None and len(icl_examples_list) != len(prompt_texts):
            raise ValueError("Length of icl_examples_list must match length of prompt_texts")

        if icl_examples_list is None:
            icl_examples_list = [None] * len(prompt_texts)

        concurrency_limit = model_config.get("concurrency_limit", DEFAULT_CONCURRENCY_LIMIT)
        semaphore = asyncio.Semaphore(concurrency_limit)

        tasks = []
        for i in range(len(prompt_texts)):
            task = asyncio.create_task(
                self._evaluate_prompt_with_semaphore(
                    semaphore, model_id, prompt_texts[i],
                    possible_answers_list[i], reasoning_budgets[i],
                    icl_examples_list[i]
                )
            )
            tasks.append(task)

        if external_progress is None:
            progress_context = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
                transient=False
            )
            use_external_progress = False
        else:
            progress_context = external_progress
            use_external_progress = True

        all_results = [None] * len(tasks)
        with progress_context as progress:
            if not use_external_progress:
                batch_task = progress.add_task(
                    f"Evaluating {len(tasks)} prompts concurrently for {model_id} (limit: {concurrency_limit})",
                    total=len(tasks)
                )

            gathered_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result_or_exception in enumerate(gathered_results):
                if isinstance(result_or_exception, Exception):
                    logger.error(f"Error evaluating prompt index {i} for {model_id}: {result_or_exception}", exc_info=True)
                    all_results[i] = {
                        "model": model_id,
                        "reasoning_budget": reasoning_budgets[i],
                        "error": f"Concurrency failure: {str(result_or_exception)}",
                        "response": None,
                        "extracted_answer": None,
                        "latency": 0,
                        "cost": 0,
                        "reasoning_content": None,
                        "stop_reason": None,
                        "input_tokens": None,
                        "output_tokens": None,
                        "usage": None,
                    }
                else:
                    all_results[i] = result_or_exception

                if not use_external_progress:
                    progress.advance(batch_task)

            if not use_external_progress:
                progress.update(batch_task, completed=len(tasks))

        logger.info(f"Finished concurrent evaluation for {model_id}. Got {len(all_results)} results.")
        return all_results
