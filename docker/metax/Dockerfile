ARG METAX_BASE_IMAGE=cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.20.0-maca.ai3.7.0.107-torch2.8-py310-ubuntu22.04-amd64
FROM ${METAX_BASE_IMAGE}

SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        openssh-client \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir \
        "PyYAML>=6.0,<7" \
        "pytest>=8.0,<10"

COPY . /workspace/FlagRAND
COPY docker/metax/entrypoint.sh /usr/local/bin/flagrand-metax-entrypoint

RUN chmod 0755 \
        /usr/local/bin/flagrand-metax-entrypoint \
        /workspace/FlagRAND/docker/metax/build.sh \
        /workspace/FlagRAND/docker/metax/run.sh

ENV MACA_HOME=/opt/maca \
    MACA_PATH=/opt/maca \
    FLAGRAND_VENDOR=metax \
    FLAGRAND_MCRAND_PATH=/opt/maca/lib/libmcrand.so \
    FLAGRAND_USE_LIBTUNER=0 \
    PYTHONPATH=/workspace/FlagRAND/src:/workspace/FlagRAND \
    PYTHON_BIN=python3 \
    TRITON_CACHE_DIR=/workspace/FlagRAND/.triton_cache \
    XDG_CACHE_HOME=/workspace/FlagRAND/.cache

WORKDIR /workspace/FlagRAND
ENTRYPOINT ["/usr/local/bin/flagrand-metax-entrypoint"]
CMD ["bash"]
