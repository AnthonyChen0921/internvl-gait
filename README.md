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


### Conda env
```bash
cd ~/internvl-gait

conda create -n internvl-gait python=3.10 -y
conda activate internvl-gait

pip install -r requirements.txt
```

### Update transformers to support QWEN3 & install openCV
```bash
conda activate internvl-gait
pip install "transformers>=4.40.0" accelerate safetensors einops
pip install opencv-python-headless==4.10.0.84
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
mkdir -p ~/Models
cd ~/Models
git lfs install
git clone https://huggingface.co/OpenGVLab/InternVL3_5-1B
git clone https://huggingface.co/OpenGVLab/InternVL3_5-8B
```

### Running the experiment
```bash
cd internvl-gait/
export INTERNVL_MODEL_PATH=~/Models/InternVL3_5-1B
nohup python train_evl_framealigned_skeltext_classifier.py   > train_evl_framealigned_skeltext_classifier.log 2>&1 &
tail -f train_evl_framealigned_skeltext_classifier.log
```