## InternVL Minimal Inference Environment

This workspace is set up to run **minimal inference** with an InternVL model from Hugging Face, then we will extend it with a skeleton-token adapter for gait analysis.

### 1. Create & activate a virtual environment (Windows / PowerShell)

```powershell
cd C:\Users\1nkas-Strix-4090-ll\Desktop\InternVL
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If execution policy blocks activation, run PowerShell as Administrator and:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned
```

Then try activation again.

### 2. Install dependencies

> If you have an NVIDIA GPU + CUDA, install a GPU build of PyTorch first (recommended).

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

If you need a specific CUDA build of PyTorch, follow the commands from `https://pytorch.org` first, then install the rest:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt --no-deps
```

### 3. Run minimal InternVL inference

After installing, you will be able to run:

```powershell
python minimal_internvl_inference.py
```

The script will:

- load `OpenGVLab/InternVL3_5-8B` from Hugging Face with `trust_remote_code=True`
- run a simple text-only or image+text prompt
- print the generated response

Once this works, we will:

1. Wrap the base InternVL model in a custom `InternVLWithSkeleton` module.
2. Add a linear projection for 46‑dim skeleton vectors + temporal embeddings.
3. Concatenate skeleton tokens with visual tokens and feed them into the LLM.
4. Add a small training script for gait classification / explanation.






