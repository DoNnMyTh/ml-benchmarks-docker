# CosmoFlow MLCommons HPC reference (TensorFlow).
# Upstream: https://github.com/mlcommons/hpc/tree/main/cosmoflow
# Alt impl (--impl pytorch) delegates to community port if CF_PYTORCH_REPO env set.
FROM nvcr.io/nvidia/tensorflow:24.10-tf2-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Clone MLCommons HPC repo and pin CosmoFlow subdir. Pinned to a known-good tag.
ARG HPC_REF=main
RUN git clone --depth 1 --branch ${HPC_REF} https://github.com/mlcommons/hpc.git /workspace/hpc

RUN pip install --no-cache-dir \
      "mpi4py" \
      "pyyaml" \
      "h5py" \
      "pandas" \
      "tensorboard"

COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

WORKDIR /workspace/hpc/cosmoflow

ENTRYPOINT ["/workspace/entrypoint.sh"]
