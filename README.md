# ml-benchmarks-docker

Dockerized, end-to-end runners for three ML benchmarks, designed to run on a multi-GPU workstation (tested on 4×H100 and 1×RTX 4080).

| Benchmark | Framework | Model / Upstream | Dataset |
|-----------|-----------|------------------|---------|
| **ResNet50 / ImageNet-1k** | PyTorch (HF Transformers or torchvision) | [microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50) | ImageNet-1k (ILSVRC2012) |
| **CosmoFlow** | TensorFlow (MLCommons HPC ref) | [mlcommons/hpc/cosmoflow](https://github.com/mlcommons/hpc/tree/main/cosmoflow) | CosmoFlow mini (~100GB) |
| **BERT-base / SQuAD 2.0** | PyTorch (HF Transformers) | [bert-base-uncased](https://huggingface.co/bert-base-uncased) / [deepset/bert-base-uncased-squad2](https://huggingface.co/deepset/bert-base-uncased-squad2) | SQuAD v2 |

Each benchmark has:
- its own Dockerfile under `docker/<name>/`,
- an end-to-end runner under `scripts/run_<name>.sh` (build → data stage → run → collect),
- published image as `donnmyth/ml-benchmarks:<name>` on Docker Hub.

Docker Hub: https://hub.docker.com/r/donnmyth/ml-benchmarks/tags

---

## Repo layout

```
ml-benchmarks-docker/
├── common/lib.sh                  # shared bash helpers
├── docker/
│   ├── resnet50/{Dockerfile,entrypoint.py}
│   ├── cosmoflow/{Dockerfile,entrypoint.sh}
│   └── bert-squad/{Dockerfile,entrypoint.py}
├── scripts/
│   ├── build_images.sh            # docker build all|<name>
│   ├── push_images.sh             # docker push all|<name>
│   ├── download_data.sh           # imagenet|imagenette|cosmoflow|squad
│   ├── run_resnet50.sh
│   ├── run_cosmoflow.sh
│   └── run_bert_squad.sh
├── data/                          # gitignored — datasets land here
├── results/                       # gitignored — per-run logs/metrics/checkpoints
├── .env.example
├── .gitignore
├── .dockerignore
└── LICENSE
```

---

## Prerequisites

### Hardware
- NVIDIA GPU (anything Ampere+ recommended; tested on H100 and RTX 4080)
- Enough disk for datasets:
  - ImageNet-1k: **~150 GB**
  - CosmoFlow mini: **~100 GB**
  - SQuAD v2: ~50 MB

### Software
- **Docker Desktop** (Windows/macOS) or Docker Engine (Linux)
- On Windows: **WSL2 backend** + **Settings → Resources → WSL integration: on** + **GPU enabled**
- NVIDIA driver ≥ 555 (for CUDA 12.4 images)
- `bash` (Git Bash works on Windows)
- `curl`, `tar`

### Important: move Docker data to a non-C: drive (Windows)
These images + datasets are big. Docker Desktop defaults to `C:\ProgramData\Docker` or the WSL `ext4.vhdx` under `%LOCALAPPDATA%\Docker`. To avoid filling C::

1. Quit Docker Desktop.
2. Docker Desktop → **Settings → Resources → Advanced → Disk image location** → move to e.g. `E:\docker`.
3. Apply & Restart.
4. (Optional, WSL) `wsl --shutdown` then `wsl --export` / `--import` the `docker-desktop-data` distro to an E: path.

Also a good idea before starting heavy downloads:
```bash
docker system prune -af --volumes
```

---

## One-time setup

```bash
git clone https://github.com/<you>/ml-benchmarks-docker.git
cd ml-benchmarks-docker
cp .env.example .env
# edit .env — add HF_TOKEN (for ImageNet + HF caches) if using ImageNet-1k
```

Build all images (or pull from Docker Hub on first run — scripts auto-build if missing):

```bash
./scripts/build_images.sh all
# or per-benchmark: ./scripts/build_images.sh resnet50
```

Push to Docker Hub (requires prior `docker login`):

```bash
./scripts/push_images.sh all
```

---

## Running the benchmarks

Every `run_*.sh` supports the same core flags:

| Flag | Values | Meaning |
|------|--------|---------|
| `--mode` | `quick` \| `full` | `quick` = smoke/throughput (minutes). `full` = convergence-grade (hours/days). **Default: `quick`** |
| `--impl` | see per-benchmark below | alternate implementation |
| `--gpus N` | integer | override auto-detected GPU count |
| `--skip-data-check` | flag | don't verify dataset presence before launch |
| `--help` | | show flags |

Results land in `results/<bench>/<UTC-timestamp>/` including `run.log` and `metrics.json`.

---

### 1. ResNet50 / ImageNet-1k

Upstream: https://huggingface.co/microsoft/resnet-50

```bash
# Smoke test on ImageNette (1.5GB, no account needed):
./scripts/run_resnet50.sh --mode quick --dataset imagenette

# Full ImageNet-1k training (HF microsoft/resnet-50):
./scripts/run_resnet50.sh --mode full --dataset imagenet

# Alt: MLPerf-style torchvision ResNet50 from scratch
./scripts/run_resnet50.sh --mode full --impl mlperf --dataset imagenet
```

**ImageNet-1k access (no image-net.org account required):**
1. Free HuggingFace account: https://huggingface.co/join
2. Accept dataset terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k
3. Create an access token: https://huggingface.co/settings/tokens
4. Put it in `.env` as `HF_TOKEN=hf_xxx`
5. Run `./scripts/download_data.sh imagenet` (downloads ~150GB to `data/imagenet/`)

Fallback paths:
- **ImageNette** (`--dataset imagenette`): 10-class 1.5GB subset for quick sanity runs.
- **Academic Torrents**: set `IMAGENET_SRC=torrent` and follow the hint printed by `download_data.sh`.

Expected targets (full mode):
- HF impl: follows fine-tune schedule on pretrained weights → ~76% top-1 in ~5 epochs
- MLPerf impl: from-scratch SGD cosine → ~75.9% top-1 @ epoch 90 (MLPerf reference target)

---

### 2. CosmoFlow

Upstream: https://github.com/mlcommons/hpc/tree/main/cosmoflow

```bash
# Stage data (~100GB, downloads from NERSC public portal):
./scripts/download_data.sh cosmoflow

# Quick 2-epoch sanity:
./scripts/run_cosmoflow.sh --mode quick

# Full MLPerf HPC benchmark run:
./scripts/run_cosmoflow.sh --mode full
```

Notes:
- Uses the official TensorFlow reference pinned in the image (`mlcommons/hpc` main at build time).
- Multi-GPU uses `mpirun -np $NUM_GPUS` inside the container.
- Dataset URL overridable via `COSMOFLOW_URL` env.
- `--impl pytorch` is a stub — to enable, fork this repo and set `CF_PYTORCH_REPO` in the Dockerfile to a community port.

---

### 3. BERT-base-uncased / SQuAD 2.0

Upstream models: https://huggingface.co/bert-base-uncased · https://huggingface.co/deepset/bert-base-uncased-squad2

```bash
# Quick fine-tune (1 epoch, 500 steps):
./scripts/run_bert_squad.sh --mode quick

# Full 2-epoch fine-tune from bert-base-uncased (training benchmark):
./scripts/run_bert_squad.sh --mode full

# Pure evaluation — load deepset fine-tuned model and compute F1/EM:
./scripts/run_bert_squad.sh --impl eval
```

Data: downloads SQuAD v2 on first run via `datasets.load_dataset("squad_v2")` into `data/squad/` (~50MB). No account needed.

Expected targets (full training):
- F1 ≈ 74–76, EM ≈ 71–73 on SQuAD v2 dev set.

`--impl eval` with deepset model: F1 ≈ 75.7, EM ≈ 72.3 (author-reported).

---

## Running on 4×H100

The scripts auto-detect GPUs via `nvidia-smi -L`. On the H100 box you can just:

```bash
./scripts/run_resnet50.sh   --mode full --dataset imagenet
./scripts/run_cosmoflow.sh  --mode full
./scripts/run_bert_squad.sh --mode full
```

Each script passes `--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864` and mounts `./data` → `/data`, `./results/<bench>/<run>/` → `/results`.

Override GPU count: `--gpus 2` or `NUM_GPUS=2 ./scripts/run_*.sh`.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `docker: Error response from daemon: could not select device driver` | Enable GPU in Docker Desktop or install `nvidia-container-toolkit` on Linux |
| `HF_TOKEN missing` | Put token in `.env`, or `export HF_TOKEN=...` before running |
| Out-of-memory on RTX 4080 | Drop batch size: `./scripts/run_resnet50.sh --mode quick --batch-size 32` (via `EXTRA` forwarding) |
| `ImageNet already staged` but empty | Delete `data/imagenet/` and retry; HF extraction step may have partial state |
| CosmoFlow `config.yaml not found` | Upstream repo layout changed. Rebuild image (`./scripts/build_images.sh cosmoflow`) to pick up latest `mlcommons/hpc` main |
| Container GPU test fails on first build | Run a warm-up: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` |

---

## Extending

Add a new benchmark:
1. `docker/<name>/{Dockerfile,entrypoint.*}` following the pattern of the three here
2. `scripts/run_<name>.sh` (copy one of the existing scripts)
3. Extend `scripts/build_images.sh` and `scripts/push_images.sh`
4. Add a section to this README and a row to the top table

---

## License

MIT. See [LICENSE](LICENSE).
