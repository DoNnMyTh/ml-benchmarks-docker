# ResNet50 on ImageNet-1k — HuggingFace microsoft/resnet-50 + transformers Trainer
# Alt impl: torchvision MLPerf-style reference (selected via --impl mlperf in entrypoint)
FROM nvcr.io/nvidia/pytorch:24.10-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/data/.hf_cache \
    TRANSFORMERS_CACHE=/data/.hf_cache

WORKDIR /workspace

RUN pip install --no-cache-dir \
      "transformers==4.46.3" \
      "datasets==3.1.0" \
      "accelerate==1.1.1" \
      "evaluate==0.4.3" \
      "scikit-learn" \
      "pillow" \
      "timm"

COPY entrypoint.py /workspace/entrypoint.py

ENTRYPOINT ["python", "/workspace/entrypoint.py"]
