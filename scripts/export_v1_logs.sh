#!/usr/bin/env bash
# Export the v1 demo's interaction log (Postgres `predictions` table) from the VPS
# to data/feedback/v1_logs.csv — real user inputs to label into the gold set.
#
#   scripts/export_v1_logs.sh user@your-vps [path-to-v1-repo-on-vps]
set -euo pipefail
HOST="${1:?usage: $0 user@host [remote_dir]}"
REMOTE_DIR="${2:-text-to-icpc2_demo}"
OUT="data/feedback/v1_logs.csv"
mkdir -p "$(dirname "$OUT")"

ssh "$HOST" "cd $REMOTE_DIR && (docker compose exec -T db psql -U icpc2user -d icpc2 \
  -c \"\\copy (SELECT * FROM predictions ORDER BY id) TO STDOUT WITH CSV HEADER\" \
  || docker-compose exec -T db psql -U icpc2user -d icpc2 \
  -c \"\\copy (SELECT * FROM predictions ORDER BY id) TO STDOUT WITH CSV HEADER\")" > "$OUT"

echo "$(($(wc -l < "$OUT") - 1)) rows -> $OUT"
echo "Next: icpc2 gold-template --src $OUT --model diogocarapito/text-to-icpc2-bert-base-portuguese-cased"
echo "      (v1 model has no icpc2_model.json; omit --model, or use a v2 model once trained)"
