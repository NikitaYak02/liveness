# Sourced by install_*.sh. Each method gets a venv at
# third_party/.venv_<method> and its repo under third_party/ (override with PS_THIRD_PARTY).
# Prefers `uv`; falls back to `pythonX.Y -m venv`. No conda required.
set -euo pipefail

ENVS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY="${PS_THIRD_PARTY:-$(cd "$ENVS_DIR/.." && pwd)/third_party}"
mkdir -p "$THIRD_PARTY"

# clone_repo NAME URL COMMIT
clone_repo() {
    local dir="$THIRD_PARTY/$1"
    if [ ! -d "$dir/.git" ]; then
        git clone "$2" "$dir"
    fi
    git -C "$dir" fetch --depth 1 origin "$3" 2>/dev/null || true
    git -C "$dir" checkout -q "$3"
}

# make_env METHOD PYTHON_VERSION  -> activates third_party/.venv_METHOD
make_env() {
    local method="$1" pyver="$2"
    local venv="$THIRD_PARTY/.venv_$method"
    if [ ! -x "$venv/bin/python" ]; then
        if command -v uv >/dev/null 2>&1; then
            # --seed is not required: we install via `uv pip`, which does not need pip inside the venv.
            uv venv -p "$pyver" "$venv"
        elif command -v "python$pyver" >/dev/null 2>&1; then
            "python$pyver" -m venv "$venv"
        else
            echo "need uv or python$pyver to create $venv" >&2
            exit 1
        fi
    fi
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
    export VIRTUAL_ENV="$venv"
    if ! command -v uv >/dev/null 2>&1; then
        python -m ensurepip --upgrade >/dev/null 2>&1 || true
        python -m pip install -U pip wheel setuptools >/dev/null
    fi
    echo "using $venv ($(python -V))"
}

# pip_install [pip args...]  (uv pip when available — works without pip in the venv)
pip_install() {
    if command -v uv >/dev/null 2>&1; then
        uv pip install "$@"
    else
        python -m pip install "$@"
    fi
}

# download URL DEST
download() {
    [ -s "$2" ] && return 0
    mkdir -p "$(dirname "$2")"
    if command -v wget >/dev/null 2>&1; then
        wget -q --show-progress -O "$2.part" "$1" && mv "$2.part" "$2"
    else
        curl -L --progress-bar -o "$2.part" "$1" && mv "$2.part" "$2"
    fi
}
