#!/usr/bin/env bash
set -euo pipefail
INTERVAL_MS=${1:-200}
LOGFILE=${2:-"tegrastats.log"}
shift 2 || true
tegrastats --interval ${INTERVAL_MS} --logfile ${LOGFILE} &
TS_PID=$!
sleep 0.3
"$@"
kill ${TS_PID} || true
wait ${TS_PID} 2>/dev/null || true
echo "tegrastats saved to ${LOGFILE}"
