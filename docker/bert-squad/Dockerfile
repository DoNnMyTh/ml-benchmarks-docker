# BERT-base-uncased fine-tune on SQuAD v2.
# Impls:
#   default : fine-tune bert-base-uncased on SQuAD2 (training benchmark)
#   eval    : load deepset/bert-base-uncased-squad2 and eval F1/EM only
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
      "scikit-learn"

COPY entrypoint.py /workspace/entrypoint.py

ENTRYPOINT ["python", "/workspace/entrypoint.py"]
