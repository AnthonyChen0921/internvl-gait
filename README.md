## InternVL-gait Command Lines

### Prep
```bash
sudo apt-get update
sudo apt-get install -y git wget bzip2 ca-certificates unzip
```

### CUDA install
Skipped

### GitHub clone
```bash
export REPO_URL="https://github.com/AnthonyChen0921/internvl-gait"
export REPO_DIR="$HOME/internvl-gait"

git clone "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR"
```

### Conda installation
```bash
command -v conda >/dev/null 2>&1 || {
  wget -O /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
  export PATH="$HOME/miniconda3/bin:$PATH"
  conda init bash
}

exec bash -l
```

### Conda env creation and activation
```bash
conda create -n internvl-gait python=3.10 -y
conda activate internvl-gait

python -V
which python
```

### Install requirements
```bash
cd "$HOME/internvl-gait"
pip install -U pip wheel setuptools
test -f requirements.txt && pip install -r requirements.txt
```

### Update transformers to match hub 1.x
```bash
conda activate internvl-gait
pip install -U "transformers>=4.49.0" "huggingface_hub>=0.23.2" "tokenizers>=0.19" accelerate safetensors
python -c "import transformers, huggingface_hub; print('transformers', transformers.__version__); print('huggingface_hub', huggingface_hub.__version__)"
```

### Download GAVD-sequences and put it under internvl-gait/GAVD-sequences (Google instance)
```bash
export BUCKET="gavd-videos"
export DATA_ROOT="$HOME/datasets"
export GAVD_DIR="$DATA_ROOT/GAVD-sequences"
export ZIP_NAME="GAVD-sequences.zip"

mkdir -p "$GAVD_DIR"

gcloud storage cp \
  "gs://${BUCKET}/${ZIP_NAME}" \
  "${GAVD_DIR}/"

ls -lh "${GAVD_DIR}/${ZIP_NAME}"
```

### Unzip
```bash
cd ~/datasets/GAVD-sequences
unzip -o GAVD-sequences.zip
```

### Move unzipped videos to internvl-gait/GAVD-sequences
```bash
SRC="$HOME/datasets/GAVD-sequences/GAVD-sequences"
DEST="$HOME/internvl-gait/GAVD-sequences"

mkdir -p "$DEST"

# (Optional) quick preview
find "$SRC" -type f -name "*.mp4" | head

# Move + flatten with collision-safe renaming
find "$SRC" -type f -name "*.mp4" -print0 | while IFS= read -r -d '' f; do
  base="$(basename "$f")"
  target="$DEST/$base"

  # If same filename already exists, append a short hash to make it unique
  if [ -e "$target" ]; then
    h="$(printf "%s" "$f" | md5sum | cut -c1-8)"
    target="$DEST/${base%.mp4}_$h.mp4"
  fi

  mv -v "$f" "$target"
done

# Clean up any empty directories left behind
find "$SRC" -type d -empty -delete
```

### Clone the GAVD
```bash
git clone https://github.com/Rahmyyy/GAVD

# Move GAVD’s content to internvl-gait/GAVD
SRC="$HOME/GAVD"
DEST="$HOME/internvl-gait/GAVD"

mkdir -p "$DEST"

# Move all contents (including hidden files) safely
shopt -s dotglob nullglob
mv -v "$SRC"/* "$DEST"/
shopt -u dotglob nullglob

# Optional: remove source folder if empty
rmdir "$SRC" 2>/dev/null || true
```

### Download the InternVL model
```bash
# hf
python -m pip install -U pip
python -m pip install -U huggingface_hub hf_transfer

# fast downloads (optional)
export HF_HUB_ENABLE_HF_TRANSFER=1

# internvl
mkdir -p ~/Models

export MODEL_ID="OpenGVLab/InternVL3_5-1B"   # <-- 8B as well
export LOCAL_DIR="$HOME/Models/$(basename "$MODEL_ID")"

python - << 'PY'
import os
from huggingface_hub import snapshot_download

model_id = os.environ["MODEL_ID"]
local_dir = os.environ["LOCAL_DIR"]

print("Downloading:", model_id)
print("To:", local_dir)

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    token=os.environ.get("HF_TOKEN", None),
)
print("Done.")
PY
```

### Running the experiment
```bash
cd internvl-gait/
export INTERNVL_MODEL_PATH=~/Models/InternVL3_5-1B
nohup python train_evl_framealigned_skeltext_classifier.py   > train_evl_framealigned_skeltext_classifier.log 2>&1 &
tail -f train_evl_framealigned_skeltext_classifier.log
```