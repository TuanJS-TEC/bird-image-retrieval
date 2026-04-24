# Bird Image Retrieval (CBIR-DPT)

## Clone project

```bash
git clone https://github.com/TuanJS-TEC/bird-image-retrieval.git
cd bird-image-retrieval
```

## Setup environment

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

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
