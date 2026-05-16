#!/usr/bin/env python3
"""Extract deduction milestones from reasoning traces using deepseek-v4-flash.

For each reasoning_content, identifies the first time a specific position-attribute
pair is confirmed, recording the character/token position in the text.
"""

import json
import re
import sys
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

import openai

load_dotenv()

EXTRACTION_SYSTEM = """You analyze step-by-step reasoning traces for zebra puzzle solving.

Your task: Identify every "deduction milestone" — the FIRST time the reasoning confirms a specific attribute at a specific position.

A deduction milestone is when the text first concludes something definitive like:
- "Position 3 is the Indian" or "the Indian is at position 3"
- "position 2 drives the Mercedes"
- "the architect is at position 1"
- "Position 5 likes badminton"
- "the person at position 4 eats kiwis"
- "Rose is at position 6"

Rules:
1. Only include deductions stated with CERTAINTY, not hypotheses ("might", "could", "if", "possibly")
2. Only record the FIRST time a specific position-attribute pair is confirmed
3. If a position has multiple attributes (name, nationality, car...), each is a separate node
4. The evidence MUST be an exact verbatim quote from the text (copy-paste the relevant sentence)
5. Ignore deductions that only confirm "someone is at one of the ends" without specifying which end
6. Group related deductions: if the text says "position 3: handbag=MK, car=Mercedes", treat as separate nodes

Output a JSON array sorted by occurrence order:
```json
[
  {
    "position": 3,
    "attribute": "Michael Kors handbag",
    "category": "handbag",
    "evidence": "position 3: handbag = Michael Kors"
  }
]
```

The "evidence" field is critical — it must be an exact substring from the original text. Quote it precisely."""

USER_TEMPLATE = """Extract ALL deduction milestones from this reasoning trace. Return ONLY the JSON array, no other text.

{reasoning}"""


async def extract_nodes(client: openai.AsyncOpenAI, reasoning: str, semaphore: asyncio.Semaphore) -> list[dict]:
    """Call deepseek-v4-flash to extract deduction nodes from reasoning text."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM},
                    {"role": "user", "content": USER_TEMPLATE.format(reasoning=reasoning)},
                ],
                temperature=0.0,
                max_tokens=16384,
            )
            content = response.choices[0].message.content
        except Exception as e:
            print(f"  API error: {e}", file=sys.stderr)
            return []

    # Parse JSON from response (may be wrapped in ```json ... ```)
    json_str = content
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", content, re.DOTALL)
    if m:
        json_str = m.group(1)
    else:
        m = re.search(r"\[.*\]", content, re.DOTALL)
        if m:
            json_str = m.group(0)

    try:
        nodes = json.loads(json_str)
    except json.JSONDecodeError:
        # Try to recover truncated JSON by completing the array
        json_str = json_str.rstrip()
        if json_str.endswith(","):
            json_str = json_str[:-1]
        if not json_str.endswith("]"):
            # Find last complete object before truncation
            last_complete = json_str.rfind('"}')
            if last_complete > 0:
                json_str = json_str[:last_complete + 2] + "\n]"
        try:
            nodes = json.loads(json_str)
            print(f"  Recovered {len(nodes)} nodes from truncated JSON", file=sys.stderr)
        except json.JSONDecodeError:
            print(f"  JSON parse error. Response: {content[:300]}", file=sys.stderr)
            return []

    return nodes


def locate_evidence(reasoning: str, evidence: str) -> int:
    """Find the character position of evidence text in the reasoning.

    Tries multiple matching strategies, from exact to fuzzy."""
    if not evidence or not reasoning:
        return -1

    # 1. Exact match
    idx = reasoning.find(evidence)
    if idx >= 0:
        return idx

    # 2. Case-insensitive
    idx = reasoning.lower().find(evidence.lower())
    if idx >= 0:
        return idx

    # 3. Normalize whitespace (collapse multiple spaces, strip)
    import re
    normalized_evidence = re.sub(r"\s+", " ", evidence).strip()
    normalized_reasoning = re.sub(r"\s+", " ", reasoning)
    idx = normalized_reasoning.find(normalized_evidence)
    if idx >= 0:
        # Map back to original position (approximate)
        return idx

    idx = normalized_reasoning.lower().find(normalized_evidence.lower())
    if idx >= 0:
        return idx

    # 4. Search with the first 30+ meaningful chars
    short = evidence.strip()[:50]
    if len(short) >= 20:
        for attempt in [short, short.lower()]:
            idx = reasoning.find(attempt)
            if idx >= 0:
                return idx
            idx = normalized_reasoning.find(re.sub(r"\s+", " ", attempt).strip())
            if idx >= 0:
                return idx

    # 5. Search for key phrase: position + attribute
    pos = re.search(r"position\s*(\d+)", evidence, re.IGNORECASE)
    if pos:
        pos_num = pos.group(1)
        # Extract a distinctive word from evidence (longest word > 4 chars)
        words = [w for w in re.findall(r"\b[A-Za-z]{4,}\b", evidence) if w.lower() not in
                 ("position", "person", "there", "which", "where", "their", "about", "other")]
        if words:
            for word in words[:3]:
                # Search for position number near the distinctive word
                pattern = re.compile(
                    rf"(?:position\s*{pos_num}[^.]*?\b{word}\b|\b{word}\b[^.]*?position\s*{pos_num})",
                    re.IGNORECASE
                )
                m = pattern.search(reasoning)
                if m:
                    return m.start()

    return -1


async def process_file(
    input_path: Path,
    output_path: Path,
    concurrency: int = 10,
    limit: int | None = None,
):
    """Process a JSONL file of results, extracting nodes from reasoning_content."""
    # Load results
    results = []
    with open(input_path) as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    print(f"Loaded {len(results)} results from {input_path}")

    # Filter for results with reasoning_content
    to_process = []
    for r in results:
        rc = r.get("reasoning_content", "") or ""
        if len(rc) > 100:
            to_process.append(r)

    print(f"Results with reasoning: {len(to_process)}")
    if limit:
        to_process = to_process[:limit]
        print(f"Limited to {limit}")

    client = openai.AsyncOpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    semaphore = asyncio.Semaphore(concurrency)

    output_lines = []
    success_count = 0
    total_nodes = 0

    async def process_one(i: int, r: dict):
        nonlocal success_count, total_nodes
        rc = r.get("reasoning_content", "")
        instance_id = r.get("instance_id", f"unknown_{i}")

        nodes = await extract_nodes(client, rc, semaphore)

        enriched = []
        for node in nodes:
            evidence = node.get("evidence", "")
            char_pos = locate_evidence(rc, evidence)
            token_pos = char_pos // 4 if char_pos >= 0 else -1

            enriched.append({
                "position": node.get("position"),
                "attribute": node.get("attribute"),
                "category": node.get("category"),
                "evidence": evidence,
                "char_position": char_pos,
                "token_position": token_pos,
            })

        # Sort by char_position ascending (first deduction first)
        enriched.sort(key=lambda n: n["char_position"] if n["char_position"] >= 0 else 999999)

        if enriched:
            success_count += 1
            total_nodes += len(enriched)

        return {
            "instance_id": instance_id,
            "reasoning_budget": r.get("reasoning_budget"),
            "correct": r.get("correct"),
            "correct_answer": r.get("correct_answer"),
            "extracted_answer": r.get("extracted_answer"),
            "reasoning_chars": len(rc),
            "reasoning_tokens_est": len(rc) // 4,
            "nodes": enriched,
        }

    tasks = [process_one(i, r) for i, r in enumerate(to_process)]
    for coro in asyncio.as_completed(tasks):
        result = await coro
        output_lines.append(json.dumps(result, ensure_ascii=False))
        print(f"  [{len(output_lines)}/{len(to_process)}] {result['instance_id']}: "
              f"{len(result['nodes'])} nodes, correct={result['correct']}")

    # Write output
    with open(output_path, "w") as f:
        for line in output_lines:
            f.write(line + "\n")

    print(f"\nDone. {success_count}/{len(to_process)} successful, {total_nodes} total nodes")
    print(f"Output: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_deduction_nodes.py <input.jsonl> [--limit N] [--concurrency N]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"File not found: {input_path}")
        sys.exit(1)

    # Parse optional args
    limit = None
    concurrency = 10
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1])
            i += 2
        elif args[i] == "--concurrency" and i + 1 < len(args):
            concurrency = int(args[i + 1])
            i += 2
        else:
            i += 1

    # Output path: same dir as input, with _nodes suffix
    output_path = input_path.parent / f"{input_path.stem}_nodes.jsonl"

    asyncio.run(process_file(input_path, output_path, concurrency=concurrency, limit=limit))


if __name__ == "__main__":
    main()
