#!/bin/bash

all_val_scene_tokens_and_count() {
    python3 - <<'PY'
import json
import pickle
import sys
from pathlib import Path

import yaml

def norm(name: str) -> str:
    for ext in (".pkl", ".pickle", ".gz"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name

repo_root = Path("/home/byounggun/DiffusionDrive")
split_cfg = repo_root / "navsim/planning/script/config/training/default_train_val_test_log_split.yaml"
navtrain_cfg = (
    repo_root
    / "navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml"
)

with split_cfg.open() as f:
    log_split = yaml.safe_load(f)
val_logs = set(log_split.get("val_logs", []))

with navtrain_cfg.open() as f:
    navtrain_logs = set(yaml.safe_load(f).get("log_names", []))

val_logs_in_navtrain = sorted(val_logs & navtrain_logs)

cache_dir = repo_root / "train_eval/.cache"
cache_dir.mkdir(parents=True, exist_ok=True)
pool_cache = cache_dir / "navtrain_val_scene_tokens.json"

if pool_cache.is_file():
    tokens = json.loads(pool_cache.read_text())
else:
    trainval_root = Path("/data/navsim/dataset/navsim_logs/trainval")
    if not trainval_root.is_dir():
        print(f"ERROR: navsim trainval logs not found: {trainval_root}", file=sys.stderr)
        sys.exit(2)

    token_set = set()
    for log_path in trainval_root.iterdir():
        if not log_path.is_file():
            continue
        log_name = norm(log_path.name)
        if log_name not in val_logs_in_navtrain:
            continue
        with log_path.open("rb") as f:
            frames = pickle.load(f)
        for frame in frames:
            token = frame.get("scene_token")
            if token:
                token_set.add(token)

    tokens = sorted(token_set)
    pool_cache.write_text(json.dumps(tokens))

print(f"{len(tokens)} {json.dumps(tokens, separators=(',', ':'))}")
PY
}
