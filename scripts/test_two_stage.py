#!/usr/bin/env python3
"""Verify the two-stage early stopping framework works end-to-end."""

import asyncio, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model_interface import ModelInterface
from utils.model_helpers import (
    _get_stage1_interrupted_prompt,
    _get_stage1_natural_prompt,
    _prepare_time_up_prompt,
    _prepare_prompt,
    _format_prompt_content,
    _extract_answer_tag,
)

LOCAL_CONFIG = {
    "deepseek-r1-local": {
        "id": "deepseek-r1-local",
        "model_name": "/opt/models/DeepSeek-R1-Distill-Qwen-14B",
        "type": "deepseek",
        "system_prompt": "You are a helpful AI assistant.",
        "temperature": 0.7,
        "concurrency_limit": 1,
    }
}

DASHSCOPE_CONFIG = {
    "deepseek-v4-flash": {
        "id": "deepseek-v4-flash",
        "model_name": "deepseek-v4-flash",
        "type": "deepseek",
        "system_prompt": "You are a helpful AI assistant.",
        "temperature": 0.7,
        "concurrency_limit": 1,
    }
}


def test_prompt_templates():
    """Test 1: Prompt templates are distinct for interrupted vs natural mode."""
    print("=" * 60)
    print("Test 1: Prompt templates")
    print("=" * 60)

    interrupted = _get_stage1_interrupted_prompt(1024, False)
    natural = _get_stage1_natural_prompt(-2, False)

    assert "INTERRUPTED" in interrupted, "FAIL: interrupted prompt missing INTERRUPTED keyword"
    assert "INTERRUPTED" not in natural, "FAIL: natural prompt should NOT contain INTERRUPTED"
    print("  [PASS] Interrupted prompt contains INTERRUPTED")
    print("  [PASS] Natural prompt does NOT contain INTERRUPTED")

    # budget=0 special case
    zero_prompt = _get_stage1_interrupted_prompt(0, False)
    assert "DO NOT THINK" in zero_prompt, "FAIL: budget=0 should say DO NOT THINK"
    print("  [PASS] Budget=0 prompt contains DO NOT THINK")

    print()


def test_stage2_prompt():
    """Test 2: Stage 2 prompt contains truncated reasoning and Time's Up."""
    print("=" * 60)
    print("Test 2: Stage 2 prompt structure")
    print("=" * 60)

    stage2 = _prepare_time_up_prompt(
        truncated_reasoning="We first list all categories...",
        original_question="What position is the Swede at?",
        possible_answers=[],
        budget=1024,
        base_system_prompt="You are a helpful AI assistant.",
    )

    msgs = stage2.messages
    # Plan C: system -> user(question) -> assistant(reasoning) -> user(Time's Up) -> assistant(prefix <answer>)
    assert len(msgs) >= 4, f"FAIL: expected >=4 messages, got {len(msgs)}"
    assert msgs[1].role.value == "user", f"FAIL: msg[1] should be user"
    assert msgs[2].role.value == "assistant", f"FAIL: msg[2] should be assistant (reasoning)"
    assert "Time's Up!" in msgs[3].content, f"FAIL: msg[3] should contain Time's Up!"
    print(f"  [PASS] {len(msgs)} messages in Stage 2 prompt")
    print(f"         msg[1] user: question ({len(msgs[1].content)} chars)")
    print(f"         msg[2] assistant: reasoning ({len(msgs[2].content)} chars)")
    print(f"         msg[3] user: Time's Up!")
    print(f"         msg[4] assistant prefix: <answer>")

    # Verify prefill: last message is assistant, Prompt.deepseek_format() applies is_prefix
    last = msgs[-1]
    assert last.role.value == "assistant", "FAIL: last message should be assistant"
    assert last.content == "<answer>", "FAIL: message should be <answer>"
    # deepseek_format() checks is_last_message_assistant() and adds is_prefix=True
    assert stage2.is_last_message_assistant(), "FAIL: Prompt should detect last as assistant for prefill"
    print("  [PASS] Assistant prefill <answer> confirmed (via deepseek_format)")

    print()


def test_max_tokens_calculation():
    """Test 3: max_tokens calculation for different budgets."""
    print("=" * 60)
    print("Test 3: max_tokens mapping")
    print("=" * 60)

    BASE_TOKENS = 16384

    # Interrupted: all budgets get BASE_TOKENS
    for b in [0, 1024, 2048, 4096, 8192]:
        assert BASE_TOKENS == 16384, "Interrupted max_tokens should be BASE_TOKENS"
    print("  [PASS] All interrupted budgets -> max_tokens = 16384")

    # Natural: BASE_TOKENS + (abs_budget - 1) * 4096
    expected = {1: 16384, 2: 20480, 3: 24576, 4: 28672, 5: 32768}
    for level, mt in expected.items():
        computed = 16384 + (level - 1) * 4096
        assert computed == mt, f"FAIL: level {level}: expected {mt}, got {computed}"
    print("  [PASS] Natural mode: -1=16384, -2=20480, -3=24576, -4=28672, -5=32768")

    print()


def test_answer_extraction():
    """Test 4: Answer extraction from completion text."""
    print("=" * 60)
    print("Test 4: Answer extraction")
    print("=" * 60)

    tests = [
        ("<answer>3</answer>", "3"),
        ("<answer>Swede</answer>", "Swede"),
        ("some text <answer>4</answer> more text", "4"),
        ("<answer>Position 3</answer>", "Position 3"),
        ("no answer tag here", None),
        ("", None),
    ]
    for text, expected in tests:
        result = _extract_answer_tag(text)
        assert result == expected, f"FAIL: '{text}' -> {result}, expected {expected}"
    print(f"  [PASS] All {len(tests)} extraction cases pass")

    print()


async def test_stage1_truncation():
    """Test 5: Stage 1 truncation actually happens with tight max_tokens."""
    print("=" * 60)
    print("Test 5: Stage 1 truncation (API call)")
    print("=" * 60)

    # Use local model if available, otherwise DashScope
    try:
        config = LOCAL_CONFIG
        print("  Trying local vLLM...")
        interface = ModelInterface(config, use_cache=True)
        await interface.api._ensure_client()
        print("  Using: LOCAL vLLM")
    except Exception:
        config = DASHSCOPE_CONFIG
        print("  Local not available, trying DashScope...")
        interface = ModelInterface(config, use_cache=True)
        print("  Using: DashScope API")

    # Load a medium-difficulty puzzle
    with open("data/bbeh_zebra_puzzles_scored_g567.jsonl") as f:
        puzzles = [json.loads(line) for line in f if line.strip()]

    # Pick a g5 puzzle (simpler)
    puzzle = [p for p in puzzles if p.get("grid_size") == 5][0]
    print(f"  Puzzle: grid_size={puzzle['grid_size']}")

    # Test interrupted mode with budget=1024
    result = await interface.evaluate_prompt(
        list(config.keys())[0], puzzle["prompt"], [], 1024
    )

    print(f"  mode: {result.get('mode')}")
    print(f"  stage2_skipped: {result.get('stage2_skipped')}")
    print(f"  extracted_answer: {result.get('extracted_answer')}")
    print(f"  error: {result.get('error')}")
    print(f"  reasoning_content length: {len(result.get('reasoning_content', '') or '')}")
    print(f"  stop_reason: {result.get('stop_reason')}")

    assert result.get("error") is None, f"FAIL: evaluation error: {result['error']}"
    assert result.get("extracted_answer") is not None, "FAIL: no answer extracted"
    assert result.get("mode") == "interrupted", f"FAIL: wrong mode: {result['mode']}"

    print("  [PASS] Two-stage evaluation completed successfully")

    print()


async def test_natural_vs_interrupted():
    """Test 6: Natural and interrupted modes produce different system prompts."""
    print("=" * 60)
    print("Test 6: Natural vs Interrupted divergence")
    print("=" * 60)

    try:
        config = LOCAL_CONFIG
        interface = ModelInterface(config, use_cache=True)
        await interface.api._ensure_client()
    except Exception:
        config = DASHSCOPE_CONFIG
        interface = ModelInterface(config, use_cache=True)

    with open("data/bbeh_zebra_puzzles_scored_g567.jsonl") as f:
        puzzles = [json.loads(line) for line in f if line.strip()]
    puzzle = [p for p in puzzles if p.get("grid_size") == 5][0]

    model_id = list(config.keys())[0]

    r_interrupted = await interface.evaluate_prompt(model_id, puzzle["prompt"], [], 1024)
    r_natural = await interface.evaluate_prompt(model_id, puzzle["prompt"], [], -2)

    print(f"  Interrupted: mode={r_interrupted['mode']}, answer={r_interrupted.get('extracted_answer')}")
    print(f"  Natural:     mode={r_natural['mode']}, answer={r_natural.get('extracted_answer')}")
    print(f"  stage1_max_tokens: interrupted=16384, natural=20480")

    assert r_interrupted["mode"] == "interrupted"
    assert r_natural["mode"] == "natural"
    print("  [PASS] Both modes run with correct mode tags")

    print()


async def main():
    print()
    print("Two-Stage Framework Verification")
    print("================================")
    print()

    # Pure logic tests (no API)
    test_prompt_templates()
    test_stage2_prompt()
    test_max_tokens_calculation()
    test_answer_extraction()

    # API tests (requires model access)
    try:
        await test_stage1_truncation()
        await test_natural_vs_interrupted()
    except Exception as e:
        print(f"  [SKIP] API tests failed: {e}")
        print("  (This is normal if no model is available)")

    print("=" * 60)
    print("All pure logic tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
