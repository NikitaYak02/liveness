#!/usr/bin/env bash
# Installs every method; a failing method does not stop the others.
cd "$(dirname "$0")"
failed=()
for m in 3ddfa liveportrait gpavatar facelift dp360; do
    echo "=== $m"
    bash "install_$m.sh" || failed+=("$m")
done
if [ ${#failed[@]} -gt 0 ]; then
    echo "failed: ${failed[*]}" >&2
    exit 1
fi
