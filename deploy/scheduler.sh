#!/bin/sh
# Retrain scheduler (runs in the `scheduler` container).
# Once a day at ~03:00 server time: pull feedback from the app, retrain on an ephemeral
# DigitalOcean GPU Droplet if due, promote if the gates pass, tell the app to reload.
set -u
test -f /keys/id_ed25519 || ssh-keygen -q -t ed25519 -N "" -f /keys/id_ed25519
while true; do
  now=$(date +%s)
  next=$(date -d "tomorrow 03:00" +%s 2>/dev/null || echo $((now + 86400)))
  sleep $((next - now))
  icpc2 loop --backend do || echo "loop failed at $(date)"
done
