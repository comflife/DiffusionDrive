#!/bin/bash
# Emits the val-assignment scene_tokens that are actually trainable:
# the subset of the 455 assignment "val" scenes that have a route
# (roadblock_ids) and therefore got a PDM metric cache entry in
# metric_cache_val. The other 323 assignment scenes have empty
# roadblock_ids in the raw trainval logs (navsim's own navtrain/navtest
# filters drop them via has_route=true), so PDM reward cannot be computed
# for them. The list is precomputed and cached to JSON; regenerate with
# train_eval/_regen_val_assign_loadable.py if the assignment file changes.

val_assign_loadable_scene_tokens_and_count() {
    local cache_json="${1:-/home/byounggun/DiffusionDrive/train_eval/.cache/val_assign_loadable_scene_tokens.json}"

    python3 - "$cache_json" <<'PY'
import json
import sys
from pathlib import Path

cache_path = Path(sys.argv[1])
if not cache_path.is_file():
    print(f"ERROR: loadable val-assign token cache not found: {cache_path}", file=sys.stderr)
    sys.exit(2)

payload = json.loads(cache_path.read_text())
tokens = payload.get("tokens", [])
if not tokens:
    print(f"ERROR: no tokens in {cache_path}", file=sys.stderr)
    sys.exit(3)

print(f"{len(tokens)} {json.dumps(tokens, separators=(',', ':'))}")
PY
}
