#!/usr/bin/env bash
# Open or refresh the SSH tunnel to the running Vast.ai instance's
# llama-server, exposing it as 127.0.0.1:8800 locally.
#
# Assumes:
#   - llama-server inside the container is bound to 127.0.0.1:8000 (the
#     locked-down configuration; if you respin without HOST=127.0.0.1
#     it'll be on 0.0.0.0:8000 and external too — still works as tunnel
#     target, just less private).
#   - .last_instance file exists at ~/Projects/qwen36-vast/.last_instance
#
# Usage:
#   ./tools/vast_tunnel.sh           # open or restart tunnel
#   ./tools/vast_tunnel.sh status    # is tunnel up?
#   ./tools/vast_tunnel.sh down      # tear it down

set -euo pipefail

LOCAL_PORT="${LOCAL_PORT:-8800}"
INST_FILE="${INST_FILE:-$HOME/Projects/qwen36-vast/.last_instance}"
ACTION="${1:-up}"

PIDFILE="/tmp/qwen36-vast-tunnel.pid"

is_alive() {
    [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

case "$ACTION" in
    status)
        if is_alive; then
            echo "tunnel UP  (pid=$(cat $PIDFILE), local port $LOCAL_PORT)"
            curl -s --max-time 3 "http://127.0.0.1:${LOCAL_PORT}/health" \
                && echo " /health ok" \
                || echo " /health FAIL"
        else
            echo "tunnel DOWN"
            exit 1
        fi
        ;;
    down)
        if is_alive; then
            kill "$(cat $PIDFILE)"
            rm -f "$PIDFILE"
            echo "tunnel torn down"
        else
            echo "no tunnel running"
        fi
        ;;
    up|"")
        if is_alive; then
            echo "tunnel already up (pid=$(cat $PIDFILE))"
            exit 0
        fi
        [ -f "$INST_FILE" ] || { echo "FATAL: no .last_instance at $INST_FILE"; exit 2; }
        INST_ID="$(cat "$INST_FILE")"
        S="$(vastai show instance "$INST_ID" --raw)"
        SSH_HOST="$(echo "$S" | jq -r .ssh_host)"
        SSH_PORT="$(echo "$S" | jq -r .ssh_port)"
        echo "opening tunnel: 127.0.0.1:${LOCAL_PORT} -> ${SSH_HOST}:${SSH_PORT} -> 127.0.0.1:8000"
        ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
            -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
            -fN -L "${LOCAL_PORT}:127.0.0.1:8000" \
            -p "${SSH_PORT}" "root@${SSH_HOST}"
        # Capture pid (the -fN forks; new ssh proc is owned by us)
        sleep 1
        # Use pgrep without -n (newest) since pgrep itself can be the newest;
        # pgrep -f matches the full ssh -L command line uniquely per local port.
        pgrep -f "ssh -o.*-L ${LOCAL_PORT}:127.0.0.1:8000" | head -1 > "$PIDFILE"
        echo "tunnel up, pid=$(cat $PIDFILE)"
        sleep 1
        curl -s --max-time 3 "http://127.0.0.1:${LOCAL_PORT}/health" \
            && echo "  /health ok"
        ;;
    *)
        echo "usage: $0 [up|status|down]"
        exit 2
        ;;
esac
