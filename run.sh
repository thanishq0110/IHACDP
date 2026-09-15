#!/usr/bin/env bash
#
# Start IHACDP. Run ./setup.sh first if you have not already.
#
set -euo pipefail
cd "$(dirname "$0")"

BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; RED=$'\033[31m'; OFF=$'\033[0m'
die() { printf "\n  %s✗ %s%s\n\n  Run %s./setup.sh%s first.\n\n" "$RED" "$1" "$OFF" "$BOLD" "$OFF" >&2; exit 1; }

PORT="${PORT:-8000}"
MODEL="${IHACDP_LLM_MODEL:-$(cat .model 2>/dev/null || echo gemma3:4b)}"
OLLAMA_APP="/Applications/Ollama.app/Contents/Resources/ollama"

[ -x ./.venv/bin/python ] || die "The Python environment is missing."
[ -f artifacts/symptoms_model.joblib ] || die "The models have not been trained."

OLLAMA=""
[ -x "$OLLAMA_APP" ] && OLLAMA="$OLLAMA_APP"
[ -z "$OLLAMA" ] && command -v ollama >/dev/null 2>&1 && OLLAMA="$(command -v ollama)"
[ -n "$OLLAMA" ] || die "Ollama is not installed."

if ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  printf "  %sstarting the language model...%s\n" "$DIM" "$OFF"
  nohup "$OLLAMA" serve >/tmp/ihacdp-ollama.log 2>&1 &
  for _ in $(seq 1 40); do
    curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
    sleep 1
  done
fi
curl -fsS --max-time 3 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 \
  || die "The language model service will not start."

if ! curl -fsS http://127.0.0.1:11434/api/tags | grep -q "\"${MODEL%%:*}"; then
  printf "  %s%s is not downloaded - fetching it now...%s\n" "$DIM" "$MODEL" "$OFF"
  "$OLLAMA" pull "$MODEL"
fi

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  printf "\n  %sPort %s is already in use.%s Stop the other process, or run: PORT=8001 ./run.sh\n\n" "$RED" "$PORT" "$OFF" >&2
  exit 1
fi

printf "\n  %sIHACDP%s  ready on  %shttp://127.0.0.1:%s%s\n" "$BOLD" "$OFF" "$GREEN$BOLD" "$PORT" "$OFF"
printf "  %smodel: %s · press Ctrl+C to stop%s\n\n" "$DIM" "$MODEL" "$OFF"

export IHACDP_LLM_MODEL="$MODEL"
exec ./.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port "$PORT"
