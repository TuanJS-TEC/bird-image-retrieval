# Bird Image Retrieval (CBIR-DPT)

## Clone project

```bash
git clone https://github.com/TuanJS-TEC/bird-image-retrieval.git
cd bird-image-retrieval
```

## Setup environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.base.txt
```

### Platform-specific install (one command)

- Linux + CUDA 12.x:

```bash
pip install -r requirements.linux-cuda.txt
```

- macOS (Apple Silicon / Intel):

```bash
pip install -r requirements.macos.txt
```

`requirements.txt` currently points to `requirements.base.txt` for safe cross-platform default.

## GPU acceleration (optional but recommended)

The pipeline can automatically use GPU for all supported tasks:

- Tier-2 algorithmic fitting/encoding (CuPy + cuML)
- Tier-3 handcrafted extraction (CuPy accelerated HOG/LBP/HSV)
- Tier-2 compaction PCA in fusion stage (cuML PCA)
- FAISS build/search (GPU index when FAISS GPU is available)

Default behavior is **GPU ON** if GPU backends are installed and CUDA is available.

```bash
# force GPU on/off
export BIRD_PIPELINE_USE_GPU=1    # default
# export BIRD_PIPELINE_USE_GPU=0  # force CPU fallback

# speed profile for Tier-2
export ALGO_TIER2_SPEED_LEVEL=balanced   # fast | balanced | quality
```

Notes:

- Base/macos use `faiss-cpu` for broad compatibility.
- If you want FAISS search/build on GPU, install a FAISS GPU build compatible with your CUDA environment.

## Download dataset CUB-200-2011

Dataset link (Kaggle):

https://www.kaggle.com/datasets/wenewone/cub2002011

### Option 1: Download manually from browser

1. Open the link above.
2. Sign in to Kaggle.
3. Click **Download**.
4. Extract dataset and place it in a local folder, for example:

`data/CUB_200_2011`

### Option 2: Download with Kaggle CLI

```bash
pip install kaggle
kaggle datasets download -d wenewone/cub2002011
```

Then unzip and put extracted folder to:

`data/CUB_200_2011`

## Branch for team handover

This handover work is published on a dedicated branch so other developers can continue development without affecting `master`.
