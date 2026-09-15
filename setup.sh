#!/usr/bin/env bash
#
# IHACDP first-time setup for macOS on Apple Silicon.
#
#   ./setup.sh
#
# Installs the Python environment, the on-device language model, downloads the
# datasets and trains every model. Safe to re-run: anything already in place is
# left alone. Nothing here needs sudo.
#
set -euo pipefail
cd "$(dirname "$0")"

BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; OFF=$'\033[0m'
step()  { printf "\n%s▸ %s%s\n" "$BOLD" "$1" "$OFF"; }
ok()    { printf "  %s✓%s %s\n" "$GREEN" "$OFF" "$1"; }
info()  { printf "  %s%s%s\n" "$DIM" "$1" "$OFF"; }
warn()  { printf "  %s!%s %s\n" "$YELLOW" "$OFF" "$1"; }
die()   { printf "\n  %s✗ %s%s\n\n" "$RED" "$1" "$OFF" >&2; exit 1; }

MODEL="${IHACDP_LLM_MODEL:-gemma3:4b}"
OLLAMA_APP="/Applications/Ollama.app/Contents/Resources/ollama"

# ---------------------------------------------------------------- 1. platform
step "Checking this machine"
[ "$(uname -s)" = "Darwin" ] || die "This installer is for macOS. On Linux, install Ollama from ollama.com and run the steps in setup.sh by hand."
ARCH="$(uname -m)"
[ "$ARCH" = "arm64" ] && ok "Apple Silicon ($ARCH)" || warn "Intel Mac ($ARCH) - this will work but the model will be slow"

MEM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
if   [ "$MEM_GB" -ge 16 ]; then ok "${MEM_GB} GB memory"
elif [ "$MEM_GB" -ge 8 ];  then ok "${MEM_GB} GB memory - enough for the 4B model"
else warn "${MEM_GB} GB memory - the language model may struggle"; fi

# Only ask for the space the remaining steps actually need, so re-running on a
# finished install does not fail on a full disk.
NEED_GB=1
[ -d .venv ] || NEED_GB=$((NEED_GB + 3))
[ -x "$OLLAMA_APP" ] || command -v ollama >/dev/null 2>&1 || NEED_GB=$((NEED_GB + 1))
curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags 2>/dev/null | grep -q "\"${MODEL%%:*}" || NEED_GB=$((NEED_GB + 4))
[ -f data/raw/symcat.csv ] || NEED_GB=$((NEED_GB + 1))

FREE_GB=$(df -g . | awk 'NR==2 {print $4}')
if [ "${FREE_GB:-99}" -lt "$NEED_GB" ]; then
  die "Only ${FREE_GB} GB free, and about ${NEED_GB} GB is needed for the remaining steps. Free up some space and re-run."
fi
ok "${FREE_GB} GB free disk (needs ~${NEED_GB} GB)"

# ------------------------------------------------------------------ 2. python
step "Setting up Python"
PY=""
for c in python3.12 python3.13 python3.11 python3; do
  command -v "$c" >/dev/null 2>&1 || continue
  v=$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null) || continue
  major=${v%%.*}; minor=${v##*.}
  if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ]; then PY="$c"; break; fi
done
[ -n "$PY" ] || die "Python 3.11 or newer not found. Install it from python.org, or run: brew install python@3.12"
ok "$($PY --version)"

if [ ! -d .venv ]; then
  info "creating virtual environment..."
  "$PY" -m venv .venv
fi
VENV_PY="./.venv/bin/python"
"$VENV_PY" -m pip install --quiet --upgrade pip
ok "virtual environment ready"

step "Installing Python packages"
info "this takes a few minutes the first time"
"$VENV_PY" -m pip install --quiet -r requirements.txt
"$VENV_PY" -c "import sklearn, xgboost, shap, pandas, fastapi, uvicorn, httpx, joblib" \
  || die "A package failed to import. Re-run ./setup.sh, or see the output above."
ok "scikit-learn, XGBoost, SHAP, FastAPI and friends installed"

# ------------------------------------------------------------------ 3. ollama
step "Installing the on-device language model"
OLLAMA=""
if   [ -x "$OLLAMA_APP" ];            then OLLAMA="$OLLAMA_APP"
elif command -v ollama >/dev/null 2>&1; then OLLAMA="$(command -v ollama)"
else
  info "downloading Ollama (~190 MB) from ollama.com..."
  TMP="$(mktemp -d)"
  curl -fL --retry 3 --progress-bar -o "$TMP/Ollama.zip" https://ollama.com/download/Ollama-darwin.zip \
    || die "Could not download Ollama. Check your internet connection, or install it yourself from https://ollama.com/download"
  unzip -oq "$TMP/Ollama.zip" -d /Applications/
  rm -rf "$TMP"
  [ -x "$OLLAMA_APP" ] || die "Ollama did not install correctly. Install it manually from https://ollama.com/download"
  OLLAMA="$OLLAMA_APP"
fi
ok "Ollama at $OLLAMA"

if ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  info "starting the model service..."
  nohup "$OLLAMA" serve >/tmp/ihacdp-ollama.log 2>&1 &
  for _ in $(seq 1 40); do
    curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
    sleep 1
  done
fi
curl -fsS --max-time 3 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 \
  || die "The model service would not start. Open the Ollama app once from Applications, then re-run ./setup.sh"
ok "model service running"

if curl -fsS http://127.0.0.1:11434/api/tags | grep -q "\"${MODEL%%:*}"; then
  ok "$MODEL already downloaded"
else
  info "downloading $MODEL (~3.3 GB) - this is the longest step"
  "$OLLAMA" pull "$MODEL" || die "Could not download $MODEL. Check your connection and re-run ./setup.sh"
  ok "$MODEL downloaded"
fi
echo "$MODEL" > .model

# ---------------------------------------------------------------- 4. datasets
step "Downloading datasets"
if [ -f data/raw/symptoms.csv ] && [ -f data/raw/heart.csv ]; then
  ok "disease datasets already present"
else
  info "fetching from UCI and public mirrors..."
  "$VENV_PY" scripts/fetch_data.py || die "Dataset download failed. Check your internet connection and re-run."
fi

if [ -f data/raw/symcat.csv ]; then
  ok "SymCat already present"
else
  info "fetching SymCat (12 MB, 801 conditions)..."
  "$VENV_PY" scripts/fetch_symcat.py || die "SymCat download failed."
fi
"$VENV_PY" scripts/filter_symcat.py >/dev/null
"$VENV_PY" scripts/build_curated.py >/dev/null
ok "datasets ready"

# ---------------------------------------------------------------- 5. training
step "Training the models"
if [ -f artifacts/heart_model.joblib ] && [ -f artifacts/symptoms_model.joblib ]; then
  ok "models already trained (delete artifacts/ to retrain)"
else
  info "training five disease models - about 20 seconds"
  "$VENV_PY" scripts/train.py | tail -8
  info "building the symptom triage model"
  "$VENV_PY" scripts/train_symptoms.py 2>/dev/null | tail -4
fi
ok "models trained"

# ------------------------------------------------------------------ 6. verify
step "Verifying the installation"
"$VENV_PY" -m pytest tests -q 2>/dev/null | tail -1 | sed 's/^/  /'
"$VENV_PY" - <<'PY'
import json, sys
from pathlib import Path
a = Path("artifacts")
missing = [f for f in ("heart_model.joblib", "symptoms_model.joblib", "symptoms_meta.json") if not (a / f).exists()]
if missing:
    print(f"  MISSING: {missing}"); sys.exit(1)
m = json.loads((a / "symptoms_meta.json").read_text())
print(f"  triage: {m['n_conditions']} conditions, {m['n_symptoms']} symptoms")
print(f"  risk models: {len(json.loads((a / 'metrics_summary.json').read_text()))}")
PY

printf "\n%s  Setup complete.%s\n\n" "$GREEN$BOLD" "$OFF"
printf "  Start it with:   %s./run.sh%s\n" "$BOLD" "$OFF"
printf "  Then open:       %shttp://127.0.0.1:8000%s\n\n" "$BOLD" "$OFF"
printf "  %sEverything runs on this machine. No data leaves it, and no internet\n" "$DIM"
printf "  connection is needed once setup has finished.%s\n\n" "$OFF"
