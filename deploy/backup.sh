#!/bin/sh
# Nightly pg_dump of the interaction log; keeps 14 days (runs in the `backup` container).
while true; do
  pg_dump -h db -U icpc2user icpc2 | gzip > "/backups/icpc2-$(date +%F).sql.gz"
  find /backups -name "*.sql.gz" -mtime +14 -delete
  sleep 86400
done
