#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$1"
BACKENDS_YAML="$PROJECT_DIR/conf/backends.yaml"
export BACKEND BACKENDS_YAML

cd "$PROJECT_DIR"

PYTHON_VERSION=$(awk "/^  ${BACKEND}:/{found=1} found && /python:/{print \$2; exit}" "$BACKENDS_YAML" | tr -d '"')
VENDOR=$(echo "$BACKEND" | sed 's/-[^-]*$//')
[[ "$VENDOR" == "$BACKEND" ]] && VENDOR="$BACKEND"
PYPI_BASE=$(sed -n 's/^pypi_base: *"\(.*\)"/\1/p' "$BACKENDS_YAML")
FLAGOS_PYPI=${PYPI_BASE/\{vendor\}/$VENDOR}
MIRROR=$(sed -n 's/^mirror: *"\(.*\)"/\1/p' "$BACKENDS_YAML")

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    UV_VERSION=0.11.22
    ARCH=$(uname -m)
    mkdir -p "$HOME/.local/bin"
    curl -sSf "https://resource.flagos.net/repository/flagos-filestore/utils/uv-${ARCH}-${UV_VERSION}-linux-gnu.tar.gz" \
        | tar xz -C "$HOME/.local/bin"
fi

uv python install "$PYTHON_VERSION" --python-preference only-managed -q
uv venv .venv --python "$PYTHON_VERSION" --python-preference only-managed -q
source .venv/bin/activate
source scripts/ci/backend_env.sh "$BACKEND"

uv pip install -q \
    "wheel==0.45.0" \
    "setuptools>=68,<77" \
    "PyYAML==6.0.3" \
    "pytest>=8,<10" \
    --index "$MIRROR"

PACKAGES=$(python - <<'PY'
import os
import yaml

with open(os.environ["BACKENDS_YAML"], encoding="utf-8") as stream:
    backend = yaml.safe_load(stream)["backends"][os.environ["BACKEND"]]
print(" ".join(backend["deps"] + backend["triton"]))
PY
)
uv pip install -q $PACKAGES \
    --default-index "$FLAGOS_PYPI" \
    --index "$MIRROR" \
    --index-strategy unsafe-best-match

POST_PACKAGES=$(python - <<'PY'
import os
import yaml

with open(os.environ["BACKENDS_YAML"], encoding="utf-8") as stream:
    backend = yaml.safe_load(stream)["backends"][os.environ["BACKEND"]]
print(" ".join(backend.get("triton_post_install", [])))
PY
)
if [[ -n "$POST_PACKAGES" ]]; then
    uv pip install -q $POST_PACKAGES \
        --default-index "$FLAGOS_PYPI" \
        --index "$MIRROR" \
        --index-strategy unsafe-best-match
fi
uv pip install -e . --no-deps --no-build-isolation

if [[ "${ENABLE_CPP:-0}" == "1" ]]; then
    if [[ "$BACKEND" != "iluvatar" ]]; then
        echo "FlagRAND C++ wrapper prototype currently supports only the Iluvatar backend." >&2
        exit 2
    fi
    uv pip install -q \
        "cmake>=3.25,<4" \
        "ninja>=1.11" \
        "pybind11==3.0.3" \
        "scikit-build-core==0.12.2" \
        --index "$MIRROR"
    CMAKE_ARGS="-DFLAGRAND_BUILD_CPP_WRAPPER=ON -DFLAGRAND_CPP_BACKEND=IX -DCMAKE_BUILD_TYPE=Release" \
        uv pip install ./cpp --no-build-isolation -v
fi

python tools/validate_operator_registry.py
