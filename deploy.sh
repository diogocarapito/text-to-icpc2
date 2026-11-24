#!/usr/bin/env bash
# Update the running server to the latest main. Run ON THE VPS, from anywhere:
#   ./deploy.sh
# Steps: back up the database -> git pull -> rebuild + restart app and scheduler -> wait for the model.
# Postgres, Caddy and the backup container keep running.
set -euo pipefail
# Braces: bash parses the whole body before running it, so `git pull` rewriting this file mid-run is safe.
{

cd "$(dirname "$0")"
compose() { docker compose -f deploy/docker-compose.yml --env-file deploy/.env "$@"; }

echo "==> Database backup"
mkdir -p deploy/backups
backup="deploy/backups/pre-deploy-$(date +%F-%H%M).sql.gz"
compose exec -T db pg_dump -U icpc2user icpc2 | gzip > "$backup"
echo "    $backup"

echo "==> git pull"
before=$(git rev-parse --short HEAD)
git pull --ff-only
after=$(git rev-parse --short HEAD)
[ "$before" = "$after" ] && echo "    already at $after (rebuilding anyway)" || git --no-pager log --oneline --first-parent "$before..$after"

export GIT_SHA=$(git rev-parse HEAD)  # baked into the image; the GPU loop clones this commit
echo "==> Build and restart app + scheduler (both images bake in src/ and configs/)"
compose up -d --build app scheduler

echo "==> Waiting for the model (first start after a model change downloads it)"
for _ in $(seq 1 60); do
  health=$(compose exec -T app python -c \
    "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').read().decode())" \
    2>/dev/null || true)
  if [[ "$health" == *'"model":"'* ]]; then
    echo "    $health"
    echo "==> Deployed $after"
    exit 0
  fi
  if [[ "$health" == *'"error":"'* ]]; then
    echo "    model failed to load: $health" >&2
    break
  fi
  sleep 5
done

echo "!!  App not healthy. Logs:" >&2
compose logs --tail 50 app >&2
echo "    To roll back the code: git checkout $before && GIT_SHA=$before docker compose -f deploy/docker-compose.yml up -d --build app scheduler" >&2
echo "    (then 'git checkout main' once fixed; database backup: $backup)" >&2
exit 1
}
