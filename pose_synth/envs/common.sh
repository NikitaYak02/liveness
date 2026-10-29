# Sourced by install_*.sh. Each method gets its own conda env ps_<method> and its
# repo under pose_synth/third_party (override with PS_THIRD_PARTY).
set -euo pipefail

ENVS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY="${PS_THIRD_PARTY:-$(cd "$ENVS_DIR/.." && pwd)/third_party}"
mkdir -p "$THIRD_PARTY"

command -v conda >/dev/null || { echo "conda not found" >&2; exit 1; }
eval "$(conda shell.bash hook)"

# clone_repo NAME URL COMMIT
clone_repo() {
    local dir="$THIRD_PARTY/$1"
    if [ ! -d "$dir/.git" ]; then
        git clone "$2" "$dir"
    fi
    git -C "$dir" fetch --depth 1 origin "$3" 2>/dev/null || true
    git -C "$dir" checkout -q "$3"
}

# make_env NAME PYTHON_VERSION  (no-op when the env already exists)
make_env() {
    if ! conda env list | awk '{print $1}' | grep -qx "$1"; then
        conda create -y -n "$1" "python=$2"
    fi
    conda activate "$1"
}

# download URL DEST
download() {
    [ -s "$2" ] && return 0
    mkdir -p "$(dirname "$2")"
    wget -q --show-progress -O "$2.part" "$1" && mv "$2.part" "$2"
}
