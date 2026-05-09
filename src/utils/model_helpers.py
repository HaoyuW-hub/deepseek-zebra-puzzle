import re
import time
from typing import Any, Dict, List, Optional

from safetytooling.data_models import ChatMessage, MessageRole, Prompt

USE_DEVELOPER_ROLE_MODELS = ["o1-mini-2024-09-12", "o1-2024-12-17", "o3-mini-2025-01-31", "o4-mini-2025-04-16", "o3-2025-04-16"]

def import_time():
    return time.time()

def _format_prompt_content(prompt_text: str, possible_answers: Optional[List[str]]) -> str:
    content = prompt_text
    is_multiple_choice = bool(possible_answers)

    if is_multiple_choice:
        content += "\n\n"
        for i, answer in enumerate(possible_answers):
            letter = chr(65 + i)
            content += f"{letter}. {answer}\n"

    if is_multiple_choice:
        content += "\n\nProvide your final answer as a single letter in the format <answer>X</answer>, where X is your chosen option."
    else:
        content += "\n\nProvide your final answer in the format <answer>X</answer>, where X is the final answer."

    return content

def _prepare_prompt(
    prompt_text: str,
    possible_answers: List[str],
    model_id: str,
    system_prompt: Optional[str] = "",
    icl_examples: Optional[List[Dict[str, Any]]] = None,
    models_config: Optional[Dict] = None,
    prefill_no_think: bool = False,
) -> Prompt:
    messages = []
    processed_system_prompt = system_prompt if system_prompt else ""

    model_type = "default"
    if models_config and model_id in models_config:
        model_type = models_config[model_id].get("type", "default")

    if icl_examples:
        example_str_parts = []
        for example in icl_examples:
            try:
                example_prompt_text = example.get("prompt", "")
                example_classes = example.get("classes")
                example_user_content = _format_prompt_content(example_prompt_text, example_classes)

                example_assistant_content = ""
                if example_classes and "answer_index" in example:
                    correct_index = int(example["answer_index"])
                    if 0 <= correct_index < len(example_classes):
                        correct_letter = chr(65 + correct_index)
                        example_assistant_content = f"<answer>{correct_letter}</answer>"
                    else:
                        continue
                elif "answer" in example:
                    example_assistant_content = f"<answer>{example['answer']}</answer>"
                else:
                    continue

                example_str_parts.append(f"Example:\nUser: {example_user_content}\nAssistant: <thinking>Your thinking process...</thinking> {example_assistant_content}")

            except (ValueError, TypeError):
                continue

        if example_str_parts:
            processed_system_prompt += "\n\n" + "\n\n".join(example_str_parts)

    if processed_system_prompt:
        role = MessageRole.developer if model_id in USE_DEVELOPER_ROLE_MODELS else MessageRole.system
        messages.append(ChatMessage(role=role, content=processed_system_prompt))

    final_task_prompt = _format_prompt_content(prompt_text, possible_answers)
    messages.append(ChatMessage(role=MessageRole.user, content=final_task_prompt))

    if prefill_no_think:
        messages.append(ChatMessage(role=MessageRole.assistant, content="<think></think>", is_prefix=True))

    return Prompt(messages=messages)

def _extract_answer_tag(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None

    pattern = r"<answer>(?:(?!</?answer>).)*?</answer>"
    matches = list(re.finditer(pattern, text, re.IGNORECASE | re.DOTALL))
    if matches:
        last_match = matches[-1]
        answer = re.search(r"<answer>(.*?)</answer>", last_match.group(0), re.IGNORECASE | re.DOTALL)
        if answer:
            answer = answer.group(1).strip()
            if len(answer) == 1 and answer.isalpha():
                return answer.upper()
            elif answer:
                return answer
            else:
                return None

    match = re.search(
        r"(?:answer|option|choose|select|pick)(?:\s+is|\s*:\s*)?\s+([A-Z])(?:\.|,|\s|$)",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()

    match = re.search(r"\b([A-Z])\b(?![.-])", text)
    if match:
        preceding_text = text[:match.start()].lower()
        if not any(kw in preceding_text[-20:] for kw in ["section", "part", "appendix", "figure", "table"]):
             return match.group(1).upper()

    return None


def _get_stage1_interrupted_prompt(
    reasoning_budget: int,
    is_multiple_choice: bool,
) -> str:
    """Constructs Stage 1 system prompt that tells the model its reasoning will be interrupted.

    Corresponds to the paper's approach: the model is informed upfront about the token
    budget and that it will be cut off, so it can plan its reasoning accordingly."""
    if reasoning_budget == 0:
        return (
            f"\nCRITICAL INSTRUCTION: DO NOT THINK. DO NOT output any reasoning, analysis, or thinking process whatsoever. "
            f"You must output ONLY your final answer immediately using the <answer>X</answer> format. "
            f"Do NOT generate any text before or after the <answer> tag."
        )

    return (
        f"\nYou will analyze this problem step by step. "
        f"CRITICAL: Your thinking process will be INTERRUPTED after approximately {reasoning_budget} tokens. "
        f"If you reach a conclusion before the interruption, provide your final answer using the <answer>X</answer> format. "
        f"If the interruption occurs before you finish, you will be given a separate opportunity to provide your final answer."
    )


def _get_stage1_natural_prompt(
    reasoning_budget: int,
    is_multiple_choice: bool,
) -> str:
    """Constructs Stage 1 system prompt for natural mode: no interruption warning.

    The model reasons naturally without knowing it will be truncated. The API-level
    max_tokens still enforces truncation, but the model receives no advance notice."""
    if reasoning_budget == 0:
        return (
            f"\nCRITICAL INSTRUCTION: DO NOT THINK. DO NOT output any reasoning, analysis, or thinking process whatsoever. "
            f"You must output ONLY your final answer immediately using the <answer>X</answer> format. "
            f"Do NOT generate any text before or after the <answer> tag."
        )

    return (
        f"\nYou will analyze this problem step by step. "
        f"Provide your final answer when you are ready, using the <answer>X</answer> format."
    )


def _prepare_time_up_prompt(
    truncated_reasoning: str,
    original_question: str,
    possible_answers: list,
    budget: int,
    base_system_prompt: str,
) -> Prompt:
    """Constructs the Stage 2 prompt with 'Time's Up!' transition and answer prefill.

    Reconstructs the conversation structure to match the paper's context flow:
    User asks the question -> Assistant begins reasoning -> Reasoning is interrupted
    -> User says 'Time's Up!' -> Assistant prefills <answer> to conclude directly."""
    messages = []

    if base_system_prompt:
        messages.append(ChatMessage(role=MessageRole.system, content=base_system_prompt))

    reasoning_block = truncated_reasoning if truncated_reasoning else "(No reasoning was produced before the interruption.)"

    is_multiple_choice = bool(possible_answers)
    if is_multiple_choice:
        options_text = "\n".join(f"{chr(65 + i)}. {answer}" for i, answer in enumerate(possible_answers))
        question_content = f"{original_question}\n\n{options_text}\n\nProvide your final answer as a single letter in the format <answer>X</answer>, where X is your chosen option."
    else:
        question_content = f"{original_question}\n\nProvide your final answer in the format <answer>X</answer>, where X is the final answer."

    # Step 1: User asks the question (just like a fresh conversation)
    messages.append(ChatMessage(role=MessageRole.user, content=question_content))

    # Step 2: Assistant's partial reasoning (what the model "just generated" in Stage 1)
    messages.append(ChatMessage(role=MessageRole.assistant, content=reasoning_block))

    # Step 3: Time's Up! — the interruption
    messages.append(ChatMessage(
        role=MessageRole.user,
        content=f"Time's Up! Your reasoning was interrupted after {budget} tokens. Based on your reasoning above, output your final answer now.",
    ))

    # Step 4: Force the model to start its response from <answer>
    messages.append(ChatMessage(role=MessageRole.assistant, content="<answer>", is_prefix=True))
    return Prompt(messages=messages)

def _get_reasoning_params(
    model_id: str, reasoning_budget: int, models_config: Dict
) -> Dict[str, Any]:
    if model_id not in models_config:
        return {}

    model_config = models_config[model_id]
    model_type = model_config.get("type", "default")

    if reasoning_budget == 0:
        return {}
    elif reasoning_budget < 0:
        if model_type == "anthropic":
            return {"thinking": {"type": "enabled", "budget_tokens": 16384}}
        elif model_type == "openai" and not model_id.startswith("o1-mini") and not model_id.startswith("o1-preview") and not model_id.startswith("gpt-4.1-mini"):
            return {"reasoning_effort": "high"}
        else:
            return {}
    else:
        if model_type == "anthropic":
            if model_id in ["claude-3-7-sonnet-20250219", "claude-sonnet-4-20250514", "claude-opus-4-20250514"]:
                return {"thinking": {"type": "enabled", "budget_tokens": reasoning_budget}}
            else:
                return {}
        elif model_type == "openai" and not model_id.startswith("o1-mini") and not model_id.startswith("o1-preview") and not model_id.startswith("gpt-4.1-mini"):
            if 1024 <= reasoning_budget < 4096:
                reasoning_effort = "low"
            elif 4096 <= reasoning_budget < 8192:
                reasoning_effort = "medium"
            elif reasoning_budget >= 8192:
                reasoning_effort = "high"
            else:
                raise ValueError(f"Invalid reasoning budget: {reasoning_budget}")
            return {"reasoning_effort": reasoning_effort}
        else:
            return {}

def _calculate_max_tokens(reasoning_budget: int, base_tokens: int = 1024) -> int:
    if reasoning_budget < 0:
        return 16384 + base_tokens
    else:
        return reasoning_budget + base_tokens
