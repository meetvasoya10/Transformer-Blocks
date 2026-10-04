#!/usr/bin/env python3
# make_evalsets.py - Generate the deterministic eval sets and their SHA256 manifest.

import argparse
import hashlib
import json
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)
from thinker.tasks import TASKS, generate

DIFFICULTIES = range(1, 33)
COUNT = 200


def main():
    parser = argparse.ArgumentParser(description="Generate eval sets (200 examples per task and difficulty 1-32).")
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "data", "evalsets"))
    parser.add_argument("--manifest", default=os.path.join(ROOT, "evalsets", "manifest.json"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)

    entries = []
    for task_index, task in enumerate(TASKS):
        for difficulty in DIFFICULTIES:
            seed = 10000 * task_index + difficulty
            rng = random.Random(seed)
            lines = [json.dumps(generate(task, difficulty, rng), sort_keys=True) for _ in range(COUNT)]
            data = ("\n".join(lines) + "\n").encode("utf-8")
            name = f"{task}_d{difficulty}.jsonl"
            with open(os.path.join(args.out_dir, name), "wb") as f:
                f.write(data)
            entries.append({
                "task": task,
                "difficulty": difficulty,
                "count": COUNT,
                "seed": seed,
                "file": name,
                "sha256": hashlib.sha256(data).hexdigest(),
            })

    with open(args.manifest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(entries, f, indent=2)
        f.write("\n")
    print(f"wrote {len(entries)} files to {args.out_dir} and manifest to {args.manifest}")


if __name__ == "__main__":
    main()
