#!/bin/bash

assignment_scene_tokens_and_count() {
    local assignment_json="${1:?assignment json path is required}"
    local assignment_splits="${2:-all}"

    python3 - "$assignment_json" "$assignment_splits" <<'PY'
import json
import sys
from pathlib import Path

assignment_path = Path(sys.argv[1])
split_arg = sys.argv[2].strip()

if not assignment_path.is_file():
    print(f"ERROR: assignment JSON not found: {assignment_path}", file=sys.stderr)
    sys.exit(2)

split_aliases = {
    "all": None,
    "*": None,
    "navtest": {"test"},
    "test": {"test"},
    "navtrain": {"train", "val"},
    "trainval": {"train", "val"},
}

if not split_arg:
    wanted_splits = None
else:
    lowered = split_arg.lower()
    if lowered in split_aliases:
        wanted_splits = split_aliases[lowered]
    else:
        wanted_splits = {part.strip() for part in split_arg.split(",") if part.strip()}

with assignment_path.open("r") as f:
    payload = json.load(f)

tokens = []
seen = set()
for assignment in payload.get("assignments", []):
    if assignment.get("dataset") != "navsim":
        continue
    if wanted_splits is not None and assignment.get("split") not in wanted_splits:
        continue
    token = assignment.get("scene_token")
    if token and token not in seen:
        seen.add(token)
        tokens.append(token)

if not tokens:
    print(
        f"ERROR: no navsim scene_token entries found in {assignment_path} "
        f"for ASSIGNMENT_SPLITS={split_arg!r}",
        file=sys.stderr,
    )
    sys.exit(3)

print(f"{len(tokens)} {json.dumps(tokens, separators=(',', ':'))}")
PY
}
