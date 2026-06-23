#!/bin/bash

random_scene_tokens_and_count() {
    local assignment_json="${1:?assignment json path is required}"
    local random_seed="${2:-42}"
    local scene_count="${3:-}"

    python3 - "$assignment_json" "$random_seed" "$scene_count" <<'PY'
import json
import os
import random
import sys
from pathlib import Path

assignment_path = Path(sys.argv[1])
random_seed = int(sys.argv[2])
scene_count_arg = sys.argv[3].strip()

split = os.environ.get("RANDOM_SCENE_DATA_SPLIT", "all").strip().lower()
if split not in ("all", "trainval", "test", "val"):
    print(
        f"ERROR: RANDOM_SCENE_DATA_SPLIT must be one of all/trainval/test/val, got {split!r}",
        file=sys.stderr,
    )
    sys.exit(6)

if not assignment_path.is_file():
    print(f"ERROR: assignment JSON not found: {assignment_path}", file=sys.stderr)
    sys.exit(2)

with assignment_path.open("r") as f:
    payload = json.load(f)

assignment_tokens = []
assignment_seen = set()
for assignment in payload.get("assignments", []):
    if assignment.get("dataset") != "navsim":
        continue
    token = assignment.get("scene_token")
    if token and token not in assignment_seen:
        assignment_seen.add(token)
        assignment_tokens.append(token)

if scene_count_arg:
    scene_count = int(scene_count_arg)
else:
    scene_count = len(assignment_tokens)

if scene_count <= 0:
    print("ERROR: scene_count must be positive", file=sys.stderr)
    sys.exit(3)

script_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
cache_dir = Path("/home/byounggun/DiffusionDrive/train_eval/.cache")
cache_dir.mkdir(parents=True, exist_ok=True)
selection_cache = cache_dir / f"random_scene_tokens_{split}_seed{random_seed}_n{scene_count}.json"

if selection_cache.is_file():
    cached = json.loads(selection_cache.read_text())
    tokens = cached["tokens"]
    if len(tokens) != scene_count:
        print(
            f"ERROR: cached token count mismatch for {selection_cache}: "
            f"expected {scene_count}, got {len(tokens)}",
            file=sys.stderr,
        )
        sys.exit(4)
    print(f"{len(tokens)} {json.dumps(tokens, separators=(',', ':'))}")
    sys.exit(0)

if split == "val":
    pool_cache = cache_dir / "navtrain_val_scene_tokens.json"
    if pool_cache.is_file():
        all_scene_tokens = set(json.loads(pool_cache.read_text()))
    else:
        import subprocess

        repo_root = Path("/home/byounggun/DiffusionDrive")
        helper = repo_root / "train_eval/_val_scene_tokens.sh"
        proc = subprocess.run(
            ["bash", "-c", f"source {helper} && all_val_scene_tokens_and_count"],
            capture_output=True,
            text=True,
            check=True,
        )
        count_and_tokens = proc.stdout.strip().split(" ", 1)
        all_scene_tokens = set(json.loads(count_and_tokens[1]))
elif split == "all":
    pool_cache = cache_dir / "all_navsim_scene_tokens.json"
    data_roots = [
        Path("/data/navsim/dataset/navsim_logs/test"),
        Path("/data/navsim/dataset/navsim_logs/trainval"),
    ]
    if pool_cache.is_file():
        all_scene_tokens = set(json.loads(pool_cache.read_text()))
    else:
        import pickle

        all_scene_tokens = set()
        for data_root in data_roots:
            if not data_root.is_dir():
                print(f"WARNING: navsim log root not found: {data_root}", file=sys.stderr)
                continue
            for log_path in data_root.iterdir():
                if not log_path.is_file():
                    continue
                with log_path.open("rb") as f:
                    frames = pickle.load(f)
                for frame in frames:
                    scene_token = frame.get("scene_token")
                    if scene_token:
                        all_scene_tokens.add(scene_token)

        pool_cache.write_text(json.dumps(sorted(all_scene_tokens)))
elif split == "test":
    pool_cache = cache_dir / "test_navsim_scene_tokens.json"
    data_roots = [Path("/data/navsim/dataset/navsim_logs/test")]
    if pool_cache.is_file():
        all_scene_tokens = set(json.loads(pool_cache.read_text()))
    else:
        import pickle

        all_scene_tokens = set()
        for data_root in data_roots:
            if not data_root.is_dir():
                print(f"WARNING: navsim log root not found: {data_root}", file=sys.stderr)
                continue
            for log_path in data_root.iterdir():
                if not log_path.is_file():
                    continue
                with log_path.open("rb") as f:
                    frames = pickle.load(f)
                for frame in frames:
                    scene_token = frame.get("scene_token")
                    if scene_token:
                        all_scene_tokens.add(scene_token)

        pool_cache.write_text(json.dumps(sorted(all_scene_tokens)))
else:  # trainval
    pool_cache = cache_dir / "trainval_navsim_scene_tokens.json"
    data_roots = [Path("/data/navsim/dataset/navsim_logs/trainval")]
    if pool_cache.is_file():
        all_scene_tokens = set(json.loads(pool_cache.read_text()))
    else:
        import pickle

        all_scene_tokens = set()
        for data_root in data_roots:
            if not data_root.is_dir():
                print(f"WARNING: navsim log root not found: {data_root}", file=sys.stderr)
                continue
            for log_path in data_root.iterdir():
                if not log_path.is_file():
                    continue
                with log_path.open("rb") as f:
                    frames = pickle.load(f)
                for frame in frames:
                    scene_token = frame.get("scene_token")
                    if scene_token:
                        all_scene_tokens.add(scene_token)

        pool_cache.write_text(json.dumps(sorted(all_scene_tokens)))

candidate_pool = sorted(all_scene_tokens - assignment_seen)
if len(candidate_pool) < scene_count:
    print(
        f"ERROR: only {len(candidate_pool)} navsim scene_tokens available after "
        f"excluding {len(assignment_seen)} assignment scenes; requested {scene_count}",
        file=sys.stderr,
    )
    sys.exit(5)

rng = random.Random(random_seed)
sampled_tokens = rng.sample(candidate_pool, scene_count)
selection_cache.write_text(
    json.dumps(
        {
            "seed": random_seed,
            "scene_count": scene_count,
            "assignment_scene_count": len(assignment_tokens),
            "excluded_assignment_scenes": len(assignment_seen),
            "candidate_pool_size": len(candidate_pool),
            "tokens": sampled_tokens,
        },
        indent=2,
    )
)

print(f"{len(sampled_tokens)} {json.dumps(sampled_tokens, separators=(',', ':'))}")
PY
}