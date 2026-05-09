#!/usr/bin/env python3
"""Extract <|position X|category|value|> markers from reasoning content."""

import json, re, sys
from pathlib import Path
from collections import Counter, defaultdict

PATTERN = re.compile(r"<deduction>\s*(\d+)\s+(\S+)\s+(.+?)</deduction>")


def extract_markers(text: str) -> list[dict]:
    if not text:
        return []
    markers = []
    for m in PATTERN.finditer(text):
        markers.append({
            "position": int(m.group(1)),
            "category": m.group(2).strip(),
            "value": m.group(3).strip(),
        })
    return markers


def main():
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        base = Path("results/deepseek-reasoner__bbeh_zebra_puzzles")
        dirs = sorted(base.glob("2*"), reverse=True)
        if not dirs:
            print("No results found")
            return
        path = dirs[0] / "raw" / "deepseek-reasoner" / "bbeh_zebra_puzzles.jsonl"
        print(f"Using: {path}")

    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))

    total_markers = 0
    by_instance = defaultdict(list)

    for r in results:
        iid = r.get("instance_id", "?")
        reasoning = r.get("reasoning_content", "")
        markers = extract_markers(reasoning)
        total_markers += len(markers)
        if markers:
            by_instance[iid].extend(markers)

    print(f"Total results: {len(results)}")
    print(f"Total markers extracted: {total_markers}")
    print(f"Instances with markers: {len(by_instance)}/{len(results)}")

    categories = Counter(m["category"] for markers in by_instance.values() for m in markers)
    print(f"\nCategory distribution:")
    for cat, count in categories.most_common():
        print(f"  {cat}: {count}")

    print(f"\nPer-instance marker counts (top 10):")
    counts = [(iid, len(ms)) for iid, ms in by_instance.items()]
    counts.sort(key=lambda x: x[1], reverse=True)
    for iid, count in counts[:10]:
        print(f"  {iid}: {count} markers")

    out_path = path.parent / f"{path.stem}_markers.json"
    with open(out_path, "w") as f:
        json.dump({iid: markers for iid, markers in by_instance.items()}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
