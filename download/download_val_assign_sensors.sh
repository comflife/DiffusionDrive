#!/bin/bash
# Download sensor blobs (camera + lidar) for the 98 val-ASSIGNMENT logs that are
# NOT part of navtrain (so they were never downloaded). Pulls only the 80 OpenScene
# full-trainval chunks that contain them, extracts ONLY those 98 logs, deletes each
# tgz immediately, and symlinks the result into the live dataset.
#
# WHY this layout:
#   - The live dataset dir (/data/navsim/dataset, on "/") is ~98% full (172G free),
#     so we store the actual data on /data2 (1.1T free) and only put symlinks on "/".
#   - tgz is downloaded transiently (one at a time) and removed after extraction,
#     so peak extra disk = one chunk (~5-10G); final kept data ~15G.
#
# Resumable: re-run anytime. Done chunks are skipped (markers); partial downloads
# resume (wget -c); already-symlinked logs are left as-is.
#
# Run it detached so it survives disconnects, e.g.:
#   tmux new -s dl 'bash download/download_val_assign_sensors.sh 2>&1 | tee /data2/byounggun/val_assign_sensors/download.log'

set -uo pipefail

BASE=/data2/byounggun/val_assign_sensors
STAGING=$BASE/blobs                 # final per-log data lives here (on /data2)
TMP=$BASE/_tmp                      # transient tgz download dir
MARK=$BASE/_markers                 # per-chunk completion markers
MAP=$BASE/chunk_logs.txt            # "<idx>\t<log1> <log2> ..."  (pre-generated)
LOGLIST=$BASE/val_assign_logs.txt   # one log name per line (pre-generated)
LIVE=/data/navsim/dataset/sensor_blobs/trainval
HF=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1

mkdir -p "$STAGING" "$TMP" "$MARK"

for f in "$MAP" "$LOGLIST"; do
  [ -f "$f" ] || { echo "ERROR: missing $f (run the mapping generator first)"; exit 1; }
done

extract_chunk () {   # $1=kind(camera|lidar)  $2=idx  $3...=logs
  local kind=$1 idx=$2; shift 2; local logs=("$@")
  local marker="$MARK/${kind}_${idx}.done"
  [ -f "$marker" ] && { echo "[skip] $kind chunk $idx"; return 0; }
  local url="$HF/openscene_sensor_trainval_${kind}/openscene_sensor_trainval_${kind}_${idx}.tgz"
  local tgz="$TMP/${kind}_${idx}.tgz"
  echo "[get ] $kind chunk $idx  (${#logs[@]} log(s))"
  wget -q --show-progress -c -O "$tgz" "$url" || { echo "  !! download failed: $kind $idx"; return 1; }
  local lg fail=0
  for lg in "${logs[@]}"; do
    tar -xzf "$tgz" --wildcards --strip-components=3 -C "$STAGING" \
        "openscene-v1.1/sensor_blobs/trainval/$lg/*" \
      || { echo "  !! extract failed: $kind $idx $lg"; fail=1; }
  done
  if [ $fail -eq 0 ]; then rm -f "$tgz"; touch "$marker"; echo "[ok  ] $kind chunk $idx"; else
    echo "  (kept $tgz for retry)"; fi
  return $fail
}

echo "=== Downloading 80 chunks (camera + lidar) for 98 val-assign logs ==="
n=0
while IFS=$'\t' read -r idx logs_str; do
  [ -z "${idx// /}" ] && continue
  read -ra logs <<< "$logs_str"
  n=$((n+1)); echo "--- [$n] chunk $idx ---"
  extract_chunk camera "$idx" "${logs[@]}" || echo "  RETRY camera $idx on next run"
  extract_chunk lidar  "$idx" "${logs[@]}" || echo "  RETRY lidar  $idx on next run"
done < "$MAP"

echo "=== Symlinking extracted logs into live dataset ==="
linked=0
while read -r lg; do
  [ -z "${lg// /}" ] && continue
  src="$STAGING/$lg"; dst="$LIVE/$lg"
  if [ -d "$src" ] && [ ! -e "$dst" ]; then ln -s "$src" "$dst" && linked=$((linked+1)); fi
done < "$LOGLIST"
echo "  newly symlinked: $linked"

echo "=== Verify (camera + lidar present per log) ==="
ok=0; bad=0
while read -r lg; do
  [ -z "${lg// /}" ] && continue
  if [ -d "$LIVE/$lg/MergedPointCloud" ] && [ -d "$LIVE/$lg/CAM_F0" ]; then
    ok=$((ok+1)); else bad=$((bad+1)); echo "  INCOMPLETE: $lg"; fi
done < "$LOGLIST"
echo "=================================================="
echo "complete logs: $ok / 98   incomplete: $bad"
[ $bad -eq 0 ] && echo "DONE. Next: build metric cache for these 98 logs, then RL on val-assign." \
              || echo "Re-run this script to finish the remaining ones."
