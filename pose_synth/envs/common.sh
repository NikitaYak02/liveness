# Sourced by install_*.sh. Each method gets a venv at
# third_party/.venv_<method> and its repo under third_party/ (override with PS_THIRD_PARTY).
# Prefers `uv`; falls back to `pythonX.Y -m venv`. No conda required.
#
# Important: never rely on a bare `python` after `activate`. Docker often reuses a
# named volume that was seeded from host venvs whose shebangs point at host paths;
# those binaries look executable but fail, and PATH then falls back to the image
# python. Always use PS_PYTHON / venv_python.
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

venv_python() {
    echo "${PS_PYTHON:?PS_PYTHON not set; call make_env first}"
}

# True if $1/bin/python actually runs (host-seeded Docker volumes often keep the
# executable bit while shebangs still point at /home/... paths).
_venv_ok() {
    local venv="$1"
    [ -x "$venv/bin/python" ] || return 1
    "$venv/bin/python" -c "import sys" >/dev/null 2>&1
}

# make_env METHOD PYTHON_VERSION  -> sets PS_PYTHON to third_party/.venv_METHOD/bin/python
make_env() {
    local method="$1" pyver="$2"
    local venv="$THIRD_PARTY/.venv_$method"
    if [ "${PS_FORCE_RECREATE:-0}" = "1" ] && [ -e "$venv" ]; then
        echo "PS_FORCE_RECREATE=1 -> removing $venv"
        rm -rf "$venv"
    fi
    if ! _venv_ok "$venv"; then
        if [ -e "$venv" ]; then
            echo "venv broken or host-seeded (unusable here) -> recreating $venv" >&2
            rm -rf "$venv"
        fi
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
    if ! _venv_ok "$venv"; then
        echo "failed to create a working python at $venv/bin/python" >&2
        exit 1
    fi
    export VIRTUAL_ENV="$venv"
    export PS_PYTHON="$venv/bin/python"
    export PATH="$venv/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        "$PS_PYTHON" -m ensurepip --upgrade >/dev/null 2>&1 || true
        "$PS_PYTHON" -m pip install -U pip wheel setuptools >/dev/null
    fi
    echo "using $venv ($("$PS_PYTHON" -V 2>&1))"
}

# pip_install [pip args...]  — always targets PS_PYTHON / the method venv
pip_install() {
    local py="${PS_PYTHON:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}"
    if [ -z "${py:-}" ] || [ ! -x "$py" ]; then
        echo "pip_install: no venv python; call make_env first" >&2
        exit 1
    fi
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "$py" "$@"
    else
        "$py" -m pip install "$@"
    fi
}

# download URL DEST  (retries a few times — HF/CDN timeouts are common)
download() {
    local url="$1" dest="$2" attempt=1
    [ -s "$dest" ] && return 0
    mkdir -p "$(dirname "$dest")"
    while [ "$attempt" -le 5 ]; do
        if command -v wget >/dev/null 2>&1; then
            if wget -q --show-progress -O "$dest.part" "$url"; then
                mv "$dest.part" "$dest"
                return 0
            fi
        else
            if curl -L --retry 3 --retry-delay 2 --progress-bar -o "$dest.part" "$url"; then
                mv "$dest.part" "$dest"
                return 0
            fi
        fi
        echo "download attempt $attempt failed for $url" >&2
        rm -f "$dest.part"
        attempt=$((attempt + 1))
        sleep $((attempt * 2))
    done
    return 1
}
