#!/usr/bin/env python3
"""Use deepseek-chat to extract position-property deductions from reasoning chains."""

import asyncio, json, sys
from pathlib import Path
from collections import Counter, defaultdict

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils

SYSTEM_PROMPT = """You are a precise data extraction assistant. Extract ALL confirmed position-property deductions from the reasoning trace below.

A deduction is when the solver CONFIRMS a specific attribute at a specific position, e.g.:
- "position 3 is the Swede" or "Pos3: Swede"
- "the architect is at position 1"
- "position 8: physical activity = weight lifting"
- "Pos4: book=fantasy, NBA=Knicks"
- Summary blocks assigning multiple properties per position

Do NOT extract:
- Speculations: "could be position 3", "maybe the Swede", "possibly at pos1"
- Clue restatements that don't assign to a position
- Category listings or descriptions of possible values

Output ONLY a JSON array with this exact format:
[{"position": N, "category": "category_name", "value": "the_value", "snippet": "exact text from reasoning"}]

Rules:
- position: integer (1-8)
- category: standardize to one of: "physical activity", "genre of books", "musical instrument", "NBA team", "favorite hobby", "profession", "nationality", "sport"
- value: the specific attribute value (e.g., "Warriors", "Swede", "running")
- snippet: copy the EXACT sentence or phrase from the reasoning that confirms this deduction
"""


async def extract_chunk(api: InferenceAPI, chunk: str, chunk_idx: int, total: int) -> list[dict]:
    prompt = Prompt(messages=[
        ChatMessage(role=MessageRole.system, content=SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.user, content=f"Fragment {chunk_idx+1}/{total}:\n\n{chunk}"),
    ])
    resp = await api(model_id="deepseek-chat", prompt=prompt, temperature=0.0, max_tokens=4096)
    text = resp[0].completion.strip()
    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return []


def locate_snippet(snippet: str, full_reasoning: str, chunk_offset: int) -> int | None:
    """Find the character offset of a snippet in the full reasoning text."""
    if not snippet:
        return None
    idx = full_reasoning.find(snippet.strip())
    if idx >= 0:
        return idx
    # Try partial match
    words = snippet.strip().split()
    if len(words) > 3:
        short = " ".join(words[:len(words)//2])
        idx = full_reasoning.find(short)
        if idx >= 0:
            return idx
    return None


def normalize(deductions: list[dict], full_reasoning: str, chunk_offset: int) -> list[dict]:
    """Normalize categories, locate snippets, remove duplicates."""
    category_map = {
        "book": "genre of books", "book genre": "genre of books", "books": "genre of books",
        "instrument": "musical instrument", "instruments": "musical instrument",
        "nba": "NBA team", "physical": "physical activity", "activity": "physical activity",
        "hobby": "favorite hobby", "sport": "sport", "profession": "profession",
        "nationality": "nationality",
    }
    seen = set()
    result = []
    for d in deductions:
        cat = category_map.get(d.get("category", "").lower(), d.get("category", ""))
        pos = d.get("position")
        val = d.get("value", "").strip()
        snippet = d.get("snippet", "")
        if not pos or not cat or not val or val.lower() == "unknown":
            continue
        key = (pos, cat.lower(), val.lower())
        if key not in seen:
            seen.add(key)
            offset = locate_snippet(snippet, full_reasoning, chunk_offset)
            result.append({
                "position": pos,
                "category": cat,
                "value": val,
                "offset": offset,
                "snippet": snippet,
            })
    return result


async def main():
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        base = Path("results/deepseek-reasoner__bbeh_zebra_puzzles")
        dirs = sorted(base.glob("2*"), reverse=True)
        if not dirs:
            print("No results found"); return
        path = dirs[0] / "raw" / "deepseek-reasoner" / "bbeh_zebra_puzzles.jsonl"
        print(f"Using: {path}")

    utils.setup_environment()
    api = InferenceAPI(cache_dir=Path.home() / ".cache" / "deepseek-zebra-eval")

    with open(path) as f:
        results = [json.loads(line) for line in f]

    all_deductions = defaultdict(list)
    total = 0

    for i, r in enumerate(results):
        iid = r.get("instance_id", "?")
        reasoning = r.get("reasoning_content", "")
        if not reasoning:
            continue

        chunk_size = 15000
        chunks = [reasoning[j:j+chunk_size] for j in range(0, len(reasoning), chunk_size)]

        print(f"[{i+1}/{len(results)}] {iid} ({len(reasoning)} chars, {len(chunks)} chunks)...")
        all_ds = []
        for ci, chunk in enumerate(chunks):
            chunk_offset = ci * chunk_size
            ds = await extract_chunk(api, chunk, ci, len(chunks))
            normalized = normalize(ds, reasoning, chunk_offset)
            all_ds.extend(normalized)
            print(f"  chunk {ci+1}: {len(ds)} raw -> {len(normalized)} normalized")

        total += len(all_ds)
        all_deductions[iid] = all_ds
        print(f"  => {len(all_ds)} total")

    print(f"\nTotal: {total} deductions from {len(results)} results")
    cats = Counter(d["category"] for dd in all_deductions.values() for d in dd)
    print("Categories:")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")

    out_path = path.parent / f"{path.stem}_llm_deductions.json"
    with open(out_path, "w") as f:
        json.dump({iid: ds for iid, ds in all_deductions.items()}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
