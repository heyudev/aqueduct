#!/usr/bin/env bash
# Back up the gateway database.
#
#   ./scripts/backup-db.sh                    # writes to ./backups/
#   BACKUP_DIR=/mnt/backups ./scripts/backup-db.sh
#   ./scripts/backup-db.sh --restore backups/aqueduct-20260728-1400.sql.gz
#
# This is the only component whose loss is not recoverable by redeploying.
# vLLM and the gateway are stateless; the declaration is in git; model weights
# can be downloaded again. But the database holds every issued API key and the
# entire usage history — lose it and every caller is locked out at once, with
# no record of who was using what.
#
# Run it from cron on the machine hosting the gateway. Restoring is a
# destructive operation and asks before proceeding.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ ! -f .env ]]; then
  echo "No .env found — this must run where the gateway is deployed." >&2
  exit 1
fi
# shellcheck disable=SC1091
set -a; . ./.env; set +a

POSTGRES_USER="${POSTGRES_USER:-aqueduct}"
POSTGRES_DB="${POSTGRES_DB:-aqueduct}"
BACKUP_DIR="${BACKUP_DIR:-$repo_root/backups}"
RETAIN_DAYS="${RETAIN_DAYS:-30}"

COMPOSE=(docker compose --project-directory gateway
         -f gateway/docker-compose.yml --env-file .env)

running() {
  "${COMPOSE[@]}" ps --status running --format '{{.Service}}' 2>/dev/null \
    | grep -qx postgres
}

do_restore() {
  local dump="$1"
  [[ -f "$dump" ]] || { echo "No such file: $dump" >&2; exit 1; }
  running || { echo "postgres is not running." >&2; exit 1; }

  echo "About to REPLACE the contents of database '$POSTGRES_DB' with:"
  echo "  $dump"
  echo
  echo "Every key issued since that dump was taken will stop working."
  read -r -p "Type the database name to confirm: " answer
  [[ "$answer" == "$POSTGRES_DB" ]] || { echo "Aborted."; exit 1; }

  local reader=cat
  [[ "$dump" == *.gz ]] && reader=gunzip

  # Drop and recreate rather than restoring over the top: a plain restore
  # leaves rows that the dump does not mention, which is not a restore.
  "${COMPOSE[@]}" exec -T postgres \
    psql -U "$POSTGRES_USER" -d postgres \
    -c "DROP DATABASE IF EXISTS \"$POSTGRES_DB\";" \
    -c "CREATE DATABASE \"$POSTGRES_DB\" OWNER \"$POSTGRES_USER\";"

  $reader < "$dump" | "${COMPOSE[@]}" exec -T postgres \
    psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" --quiet

  echo "Restored. Restart the gateway so it reconnects: make restart-gateway"
}

if [[ "${1:-}" == "--restore" ]]; then
  do_restore "${2:?usage: $0 --restore <dump.sql.gz>}"
  exit 0
fi

running || { echo "postgres is not running — start it with: make up-gateway" >&2; exit 1; }

mkdir -p "$BACKUP_DIR"
stamp="$(date +%Y%m%d-%H%M)"
target="$BACKUP_DIR/aqueduct-$stamp.sql.gz"

# Write to a temporary name first so an interrupted run cannot leave a
# truncated file that looks like a usable backup.
tmp="$target.partial"
"${COMPOSE[@]}" exec -T postgres \
  pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" --clean --if-exists \
  | gzip > "$tmp"

# A dump this small means pg_dump produced nothing useful.
size=$(wc -c < "$tmp" | tr -d ' ')
if (( size < 1024 )); then
  rm -f "$tmp"
  echo "Dump was only ${size} bytes — treating as failed, nothing written." >&2
  exit 1
fi
mv "$tmp" "$target"

echo "Wrote $target ($(du -h "$target" | cut -f1))"

keys=$(gunzip -c "$target" | grep -c 'LiteLLM_VerificationToken' || true)
echo "  contains references to issued keys: $keys"

if (( RETAIN_DAYS > 0 )); then
  removed=$(find "$BACKUP_DIR" -name 'aqueduct-*.sql.gz' -mtime "+$RETAIN_DAYS" -print -delete | wc -l | tr -d ' ')
  (( removed > 0 )) && echo "  pruned $removed backup(s) older than ${RETAIN_DAYS} days"
fi

echo
echo "A backup on the same machine as the database is not a backup."
echo "Copy $BACKUP_DIR somewhere else."
