# source this before any install/run so nothing leaks outside the folder
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="$ROOT/cache/huggingface"
export TORCH_HOME="$ROOT/cache/torch"
export PIP_CACHE_DIR="$ROOT/cache/pip"
export XDG_CACHE_HOME="$ROOT/cache"
export TMPDIR="$ROOT/cache/tmp"; mkdir -p "$TMPDIR"
if [ -f "$ROOT/venv/bin/activate" ]; then source "$ROOT/venv/bin/activate"; fi
