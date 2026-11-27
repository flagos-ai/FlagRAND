#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$1"
BACKENDS_YAML="$PROJECT_DIR/conf/backends.yaml"
export BACKEND BACKENDS_YAML

cd "$PROJECT_DIR"

PYTHON_VERSION=$(awk "/^  ${BACKEND}:/{found=1} found && /python:/{print \$2; exit}" "$BACKENDS_YAML" | tr -d '"')
SYSTEM_SITE_PACKAGES=$(awk -v backend="$BACKEND" '
  $0 == "  " backend ":" {found=1; next}
  found && /^  [^ ]/ {exit}
  found && /system_site_packages:/ {print $2; exit}
' "$BACKENDS_YAML")
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
        | tar xz --strip-components=1 -C "$HOME/.local/bin"
fi

ILUVATAR_RUNTIME=""
if [[ "$BACKEND" == "iluvatar" ]]; then
    source scripts/ci/backend_env.sh "$BACKEND"
    ILUVATAR_RUNTIME=$("${FLAGRAND_SYSTEM_PYTHON:-python3}" scripts/ci/iluvatar_python.py "$PWD/.venv")
    export ILUVATAR_RUNTIME
fi
if [[ "$ILUVATAR_RUNTIME" != "image" ]]; then
    if [[ "$SYSTEM_SITE_PACKAGES" == "true" ]]; then
        "${FLAGRAND_SYSTEM_PYTHON:-python3}" -m venv --system-site-packages .venv
    else
        uv python install "$PYTHON_VERSION" --python-preference only-managed -q
        VENV_ARGS=(--python "$PYTHON_VERSION" --python-preference only-managed -q)
        if [[ "$BACKEND" == "iluvatar" ]]; then
            VENV_ARGS+=(--clear)
        fi
        uv venv .venv "${VENV_ARGS[@]}"
        if [[ "$BACKEND" == "iluvatar" ]]; then
            # Some images expose their SDK wheels through PYTHONPATH. On SDK
            # changes, the new venv's matching wheels must win on later runs too.
            .venv/bin/python -I - <<'PY'
import sysconfig
from pathlib import Path

site = sysconfig.get_path("purelib")
(Path(site) / "flagrand-corex-bootstrap.pth").write_text(f"import sys; sys.path.insert(0, {site!r})\n")
PY
        fi
    fi
fi
source .venv/bin/activate
if [[ "$BACKEND" != "iluvatar" ]]; then
    source scripts/ci/backend_env.sh "$BACKEND"
fi

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
runtime = os.environ.get("ILUVATAR_RUNTIME")
if runtime != "image":
    packages = backend.get("sdk_wheels", {}).get(runtime, backend)
    print(" ".join(packages["deps"] + packages["triton"]))
PY
)
if [[ -n "$PACKAGES" ]]; then
    # This bootstrap installs the vendor runtime explicitly in the new venv.
    # The project's uv exclusions apply when installing FlagRAND into an image.
    uv --no-config pip install -q $PACKAGES \
        --default-index "$FLAGOS_PYPI" \
        --index "$MIRROR" \
        --index-strategy unsafe-best-match
fi

POST_PACKAGES=$(python - <<'PY'
import os
import yaml

with open(os.environ["BACKENDS_YAML"], encoding="utf-8") as stream:
    backend = yaml.safe_load(stream)["backends"][os.environ["BACKEND"]]
print(" ".join(backend.get("triton_post_install", [])))
PY
)
if [[ -n "$POST_PACKAGES" ]]; then
    uv --no-config pip install -q $POST_PACKAGES \
        --default-index "$FLAGOS_PYPI" \
        --index "$MIRROR" \
        --index-strategy unsafe-best-match
fi
PROJECT_SPEC=.
PROJECT_FLAGS=(--no-deps --no-build-isolation)
if [[ "$BACKEND" == "iluvatar" ]]; then
    PROJECT_SPEC='.[test,iluvatar]'
    PROJECT_FLAGS=(--index "$MIRROR" --no-build-isolation)
fi
if [[ "${ENABLE_CPP:-0}" != "1" ]]; then
    PROJECT_FLAGS+=(--editable)
fi
uv pip install "${PROJECT_FLAGS[@]}" "$PROJECT_SPEC"

if [[ "${ENABLE_CPP:-0}" == "1" ]]; then
    if [[ "$BACKEND" != "iluvatar" ]]; then
        echo "FlagRAND C++ wrapper prototype currently supports only the Iluvatar backend." >&2
        exit 2
    fi
    uv pip install -q \
        "cmake==3.25.2" \
        "ninja>=1.11" \
        "pybind11==3.0.3" \
        "scikit-build-core==0.12.2" \
        --index "$MIRROR"
    CMAKE_ARGS="-DFLAGRAND_BUILD_CPP_WRAPPER=ON -DFLAGRAND_CPP_BACKEND=IX -DCMAKE_BUILD_TYPE=Release" \
        uv pip install ./cpp --no-build-isolation -v
fi

python tools/validate_operator_registry.py
