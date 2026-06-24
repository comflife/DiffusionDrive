#!/bin/bash

assign_plus_random_scene_tokens_and_count() {
    local assignment_json="${1:?assignment json path is required}"
    local assignment_splits="${2:?assignment split is required}"
    local random_seed="${3:-42}"
    local random_data_split="${4:-val}"

    local assignment_token_output
    assignment_token_output="$(assignment_scene_tokens_and_count "$assignment_json" "$assignment_splits")"
    local assignment_scene_count="${assignment_token_output%% *}"
    local assignment_scene_tokens="${assignment_token_output#* }"

    export RANDOM_SCENE_DATA_SPLIT="$random_data_split"
    local random_token_output
    random_token_output="$(random_scene_tokens_and_count "$assignment_json" "$random_seed" "$assignment_scene_count")"
    local random_scene_tokens="${random_token_output#* }"

    python3 - "$assignment_scene_tokens" "$random_scene_tokens" "$assignment_splits" "$random_seed" "$random_data_split" <<'PY'
import json
import sys
from pathlib import Path

assignment_tokens = json.loads(sys.argv[1])
random_tokens = json.loads(sys.argv[2])
assignment_splits = sys.argv[3]
random_seed = int(sys.argv[4])
random_data_split = sys.argv[5]

assignment_set = set(assignment_tokens)
random_set = set(random_tokens)
overlap = assignment_set & random_set
if overlap:
    print(
        f"ERROR: assign+random overlap detected ({len(overlap)} tokens), e.g. {next(iter(overlap))}",
        file=sys.stderr,
    )
    sys.exit(7)

merged = assignment_tokens + [token for token in random_tokens if token not in assignment_set]
cache_dir = Path("/home/byounggun/DiffusionDrive/train_eval/.cache")
cache_dir.mkdir(parents=True, exist_ok=True)
cache_path = (
    cache_dir
    / f"assign_plus_random_scene_tokens_{random_data_split}_split-{assignment_splits}_seed{random_seed}_n{len(assignment_tokens)}.json"
)
cache_path.write_text(
    json.dumps(
        {
            "assignment_splits": assignment_splits,
            "random_data_split": random_data_split,
            "seed": random_seed,
            "assignment_scene_count": len(assignment_tokens),
            "random_scene_count": len(random_tokens),
            "total_scene_count": len(merged),
            "assignment_tokens": assignment_tokens,
            "random_tokens": random_tokens,
            "tokens": merged,
        },
        indent=2,
    )
)

print(f"{len(merged)} {json.dumps(merged, separators=(',', ':'))}")
PY
}
