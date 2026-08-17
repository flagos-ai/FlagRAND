FROM ubuntu:22.04

SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.11.15
ARG FLAGTREE_COMMIT=5858e827ed0c66cebd5248e0200edb81cdd838a4
ARG MAX_JOBS=8
ARG CANN_TOOLKIT_URL=https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann-toolkit_9.0.0_linux-aarch64.run
ARG CANN_OPS_URL=https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann-A3-ops_9.0.0_linux-aarch64.run
ARG TORCH_URL=https://download.pytorch.org/whl/cpu/torch-2.9.0%2Bcpu-cp311-cp311-manylinux_2_28_aarch64.whl
ARG TORCH_NPU_URL=https://gitcode.com/Ascend/pytorch/releases/download/v26.0.0-pytorch2.9.0/torch_npu-2.9.0.post2-cp311-cp311-manylinux_2_28_aarch64.whl

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Asia/Shanghai

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        curl \
        git \
        libbz2-dev \
        libffi-dev \
        liblzma-dev \
        libncursesw5-dev \
        libnuma-dev \
        libreadline-dev \
        libsqlite3-dev \
        libssl-dev \
        libxml2-dev \
        llvm \
        lld \
        make \
        nlohmann-json3-dev \
        numactl \
        pkg-config \
        tk-dev \
        uuid-dev \
        xz-utils \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fL "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz" \
        -o /tmp/Python.tgz \
    && tar -xzf /tmp/Python.tgz -C /tmp \
    && cd "/tmp/Python-${PYTHON_VERSION}" \
    && ./configure \
        --prefix="/usr/local/python${PYTHON_VERSION}" \
        --enable-shared \
        --with-ensurepip=install \
    && make -j"${MAX_JOBS}" \
    && make altinstall \
    && ln -s "/usr/local/python${PYTHON_VERSION}/bin/python3.11" "/usr/local/python${PYTHON_VERSION}/bin/python3" \
    && ln -s "/usr/local/python${PYTHON_VERSION}/bin/pip3.11" "/usr/local/python${PYTHON_VERSION}/bin/pip3" \
    && echo "/usr/local/python${PYTHON_VERSION}/lib" > /etc/ld.so.conf.d/python311.conf \
    && ldconfig \
    && rm -rf /tmp/Python.tgz "/tmp/Python-${PYTHON_VERSION}"

ENV PATH=/usr/local/python${PYTHON_VERSION}/bin:${PATH}

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir \
        "setuptools>=68" \
        wheel \
        "cmake>=3.20,<4.0" \
        "ninja>=1.11.1" \
        "pybind11>=2.13.1" \
        lit \
        attrs \
        cython \
        numpy \
        decorator \
        sympy \
        cffi \
        pyyaml \
        pathlib2 \
        psutil \
        protobuf==3.20.0 \
        scipy \
        requests \
        absl-py \
        "pytest>=8.0,<10"

RUN curl -fL "${CANN_TOOLKIT_URL}" -o /tmp/Ascend-cann.run \
    && chmod +x /tmp/Ascend-cann.run \
    && /tmp/Ascend-cann.run --install --quiet \
    && rm -f /tmp/Ascend-cann.run \
    && curl -fL "${CANN_OPS_URL}" -o /tmp/Ascend-cann-ops.run \
    && chmod +x /tmp/Ascend-cann-ops.run \
    && source /usr/local/Ascend/ascend-toolkit/set_env.sh \
    && /tmp/Ascend-cann-ops.run --install --quiet \
    && rm -f /tmp/Ascend-cann-ops.run

RUN curl -fL "${TORCH_URL}" \
        -o /tmp/torch-2.9.0+cpu-cp311-cp311-manylinux_2_28_aarch64.whl \
    && python3 -m pip install --no-cache-dir \
        /tmp/torch-2.9.0+cpu-cp311-cp311-manylinux_2_28_aarch64.whl \
    && rm -f /tmp/torch-2.9.0+cpu-cp311-cp311-manylinux_2_28_aarch64.whl \
    && curl -fL "${TORCH_NPU_URL}" \
        -o /tmp/torch_npu-2.9.0.post2-cp311-cp311-manylinux_2_28_aarch64.whl \
    && python3 -m pip install --no-cache-dir \
        /tmp/torch_npu-2.9.0.post2-cp311-cp311-manylinux_2_28_aarch64.whl \
    && rm -f /tmp/torch_npu-2.9.0.post2-cp311-cp311-manylinux_2_28_aarch64.whl

RUN git init /opt/FlagTree \
    && cd /opt/FlagTree \
    && git remote add origin https://github.com/flagos-ai/FlagTree.git \
    && git fetch --depth 1 origin "${FLAGTREE_COMMIT}" \
    && git checkout --detach FETCH_HEAD \
    && python3 -m pip install --no-cache-dir -r python/requirements.txt \
    && source /usr/local/Ascend/ascend-toolkit/set_env.sh \
    && FLAGTREE_BACKEND=ascend MAX_JOBS="${MAX_JOBS}" \
        python3 -m pip install . --no-build-isolation -v \
    && cd / \
    && rm -rf /opt/FlagTree /root/.cache/pip

COPY . /workspace/FlagRAND
COPY docker/ascend910c/entrypoint.sh /usr/local/bin/flagrand-ascend-entrypoint

RUN chmod 0755 \
        /usr/local/bin/flagrand-ascend-entrypoint \
        /workspace/FlagRAND/docker/ascend910c/build.sh \
        /workspace/FlagRAND/docker/ascend910c/run.sh \
        /workspace/FlagRAND/scripts/delivery/run_ascend_910c.sh \
        /workspace/FlagRAND/scripts/delivery/run_ascend_910c_v5.sh \
        /workspace/FlagRAND/scripts/delivery/run_ascend_910c_v6.sh \
        /workspace/FlagRAND/scripts/run_ascend_910c_cann9.sh

ENV PYTHONPATH=/workspace/FlagRAND/src:/workspace/FlagRAND \
    FLAGRAND_USE_LIBTUNER=0

WORKDIR /workspace/FlagRAND
ENTRYPOINT ["/usr/local/bin/flagrand-ascend-entrypoint"]
CMD ["bash"]
