FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    GPUREC_PREPROCESS_NATIVE_LIB=/opt/gpurec/crates/gpurec-preprocess/target/release/libgpurec_preprocess.so \
    GPUREC_BACKTRACK_BIN=/opt/gpurec/crates/gpurec-backtrack/target/release/gpurec-backtrack

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       git \
       cargo \
       && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/gpurec
COPY . /opt/gpurec

RUN python -m pip install --upgrade pip \
    && python -m pip install ".[release]"

RUN cd /opt/gpurec/crates/gpurec-preprocess \
    && cargo build --release --features python-extension \
    && cd /opt/gpurec/crates/gpurec-backtrack \
    && cargo build --release

ENV GPUREC_PREPROCESS_NATIVE_LIB=/opt/gpurec/crates/gpurec-preprocess/target/release/libgpurec_preprocess.so \
    GPUREC_BACKTRACK_BIN=/opt/gpurec/crates/gpurec-backtrack/target/release/gpurec-backtrack

WORKDIR /opt/gpurec
CMD ["bash"]

