#!/usr/bin/env python3
"""Extract position-property deductions from natural-language reasoning without markers."""

import json, re, sys
from pathlib import Path
from collections import Counter, defaultdict


def parse_categories(prompt: str) -> dict[str, set[str]]:
    """Extract category -> values mapping from the puzzle prompt."""
    categories = {}
    # Pattern: "Everyone <verbs> a different <category>: val1, val2, ..."
    pattern = re.compile(
        r"Everyone\s+(?:\w+\s+)*a\s+different\s+(.+?):\s*(.+?)\.",
        re.IGNORECASE,
    )
    for m in pattern.finditer(prompt):
        cat = m.group(1).strip().lower()
        vals = {v.strip() for v in m.group(2).split(",")}
        categories[cat] = vals

    # Also capture name-like categories: "Everyone is a different nationality: ..."
    pattern2 = re.compile(
        r"Everyone\s+is\s+a\s+different\s+(.+?):\s*(.+?)\.",
        re.IGNORECASE,
    )
    for m in pattern2.finditer(prompt):
        cat = m.group(1).strip().lower()
        vals = {v.strip() for v in m.group(2).split(",")}
        categories[cat] = vals

    return categories


def build_value_to_category(categories: dict[str, set[str]]) -> dict[str, str]:
    """Reverse mapping: value -> category name."""
    v2c = {}
    for cat, vals in categories.items():
        for v in vals:
            v2c[v.lower()] = cat
    return v2c


def extract_deductions(text: str, value_to_category: dict[str, str]) -> list[dict]:
    """Extract (position, category, value) from natural language reasoning."""
    deductions = []
    seen = set()
    pos_words = r"(?:pos(?:ition)?\s*)"

    # Pattern 1: "Position N: category = value" or "position N: category value"
    # e.g., "position 8: physical activity = weight lifting", "Pos2: book=sci-fi"
    for m in re.finditer(
        r"(?:^|\n|\.\s+|;\s*)\s*" + pos_words + r"(\d+)\s*[:;]\s*(.+?)(?:\n|$)",
        text, re.IGNORECASE | re.MULTILINE,
    ):
        pos = int(m.group(1))
        rest = m.group(2).strip()
        # Try "category = value, category = value" format
        for part in re.split(r"\s*,\s*(?=\w+\s*=)", rest):
            eq = re.match(r"(.+?)\s*=\s*(.+)", part)
            if eq:
                cat_key = eq.group(1).strip().lower()
                val = eq.group(2).strip().rstrip(".,;")
                cat = _match_category(cat_key, value_to_category)
                if cat and val.lower() in {v.lower() for v in _get_category_values(value_to_category, cat)}:
                    key = (pos, cat, val)
                    if key not in seen:
                        seen.add(key)
                        deductions.append({"position": pos, "category": cat, "value": val})

    # Pattern 2: "value at posN" / "value at position N"
    for m in re.finditer(r"(\S[^,.;\n]{0,50}?)\s+at\s+" + pos_words + r"(\d+)", text, re.IGNORECASE):
        val = m.group(1).strip().rstrip(".,;")
        pos = int(m.group(2))
        _try_add(val, pos, value_to_category, seen, deductions)

    # Pattern 3: "posN value" where value is a known entity
    # e.g., "pos1 architect", "pos8 Swede", "position 4 running"
    for m in re.finditer(r"\b" + pos_words + r"(\d+)\s+(?:is\s+)?(\S[^,.;\n]{0,50}?)(?:\s*[,.;]|\s*$|\s+and\b|\s+but\b|\s+with\b|\s+so\b|\s+the\b|\s+this\b|\s+also\b)", text, re.IGNORECASE):
        pos = int(m.group(1))
        val = m.group(2).strip().rstrip(".,;")
        _try_add(val, pos, value_to_category, seen, deductions)

    # Pattern 4: "X is at position N" / "X is at pos N"
    for m in re.finditer(r"(\S[^,.;\n]{0,50}?)\s+is\s+at\s+" + pos_words + r"(\d+)", text, re.IGNORECASE):
        val = m.group(1).strip().rstrip(".,;")
        pos = int(m.group(2))
        _try_add(val, pos, value_to_category, seen, deductions)

    return deductions


# Common abbreviations the model uses for category names
CATEGORY_ALIASES = {
    "book": "genre of books",
    "books": "genre of books",
    "instrument": "musical instrument",
    "instruments": "musical instrument",
    "nba": "nba team",
    "hobby": "favorite hobby",
    "hobbies": "favorite hobby",
    "physical": "physical activity",
    "activity": "physical activity",
    "sport": "sport",
    "profession": "profession",
    "nationality": "nationality",
}


def _match_category(cat_key: str, value_to_category: dict[str, str]) -> str | None:
    """Match a model-written category name to the canonical category."""
    cat_lower = cat_key.strip().lower().rstrip("s")
    canonicals = set(value_to_category.values())

    # Direct alias lookup
    if cat_lower in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[cat_lower]
    if cat_key.strip().lower() in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[cat_key.strip().lower()]

    # Direct canonical match
    if cat_lower in canonicals:
        return cat_lower
    if cat_key.strip().lower() in canonicals:
        return cat_key.strip().lower()

    # Word overlap fuzzy match
    key_words = set(cat_lower.replace("_", " ").replace("-", " ").split())
    for cat in canonicals:
        cat_words = set(cat.lower().split())
        if len(key_words & cat_words) >= 2:
            return cat
        # Single-word match with stemming
        for kw in key_words:
            for cw in cat_words:
                if (len(kw) >= 4 and len(cw) >= 4 and
                    (kw.startswith(cw) or cw.startswith(kw))):
                    return cat

    return None


def _get_category_values(value_to_category: dict[str, str], cat: str) -> set[str]:
    vals = set()
    for v, c in value_to_category.items():
        if c == cat:
            vals.add(v.lower())
    return vals


def _try_add(val: str, pos: int, v2c: dict, seen: set, deductions: list):
    val_lower = val.lower()
    if val_lower in v2c:
        key = (pos, v2c[val_lower], val)
        if key not in seen:
            seen.add(key)
            deductions.append({"position": pos, "category": v2c[val_lower], "value": val})


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

    all_deductions = defaultdict(list)
    total = 0

    for r in results:
        iid = r.get("instance_id", "?")
        prompt = r.get("prompt", "")
        reasoning = r.get("reasoning_content", "")

        categories = parse_categories(prompt)
        v2c = build_value_to_category(categories)
        deductions = extract_deductions(reasoning, v2c)

        total += len(deductions)
        if deductions:
            all_deductions[iid].extend(deductions)

    print(f"Total results: {len(results)}")
    print(f"Total deductions extracted: {total}")
    print(f"Instances with deductions: {len(all_deductions)}/{len(results)}")

    cats = Counter(d["category"] for dd in all_deductions.values() for d in dd)
    print(f"\nCategory distribution:")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")

    print(f"\nPer-instance deduction counts (top 10):")
    counts = [(iid, len(ds)) for iid, ds in all_deductions.items()]
    counts.sort(key=lambda x: x[1], reverse=True)
    for iid, count in counts[:10]:
        print(f"  {iid}: {count} deductions")

    out_path = path.parent / f"{path.stem}_deductions.json"
    with open(out_path, "w") as f:
        json.dump({iid: ds for iid, ds in all_deductions.items()}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
