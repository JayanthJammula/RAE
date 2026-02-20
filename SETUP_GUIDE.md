# RAE Setup Guide

Step-by-step guide to get RAE running from scratch on any machine.

---

## 1. Clone the Repository

```bash
git clone https://github.com/JayanthJammula/RAE.git
cd RAE
```

## 2. Install Dependencies

```bash
pip install torch transformers datasets nltk scikit-learn matplotlib seaborn pyyaml
```

Or if you have a `requirements.txt`:
```bash
pip install -r requirements.txt
```

## 3. Download Knowledge Graph Files

The `.pkl` files (~550 MB each) are too large for Git. Download and place them in `data/`:

- [Wikidata_triplets_dict.pkl (for MQuAKE-CF)](https://outlookuga-my.sharepoint.com/:u:/g/personal/ys07245_uga_edu/Ec0O9oUzka5LuNwK3M8FL-YBG3zw7mAdme7V9S9l4cbt7Q?e=3TKvcE)
- [Wikidata_triplets_dict_T.pkl (for MQuAKE-T)](https://outlookuga-my.sharepoint.com/:u:/g/personal/ys07245_uga_edu/EckS-8zKM75MgqmJmQH8NQMByT___C5lNyZaIsOXHQXvIQ?e=VVwF2F)
- [Original Wikidata KG](https://outlookuga-my.sharepoint.com/:u:/g/personal/ys07245_uga_edu/EbbXuq1FumtFkH3B0qmb2bMBgRdyXbayUNAevKsKvtBVUw?e=Xy3QsY)

After downloading, your `data/` folder should have:
```
data/
  Wikidata_triplets_dict.pkl
  Wikidata_triplets_dict_T.pkl
  MQuAKE-CF.json
  MQuAKE-CF-3k.json
  relation.json
  ...
```

## 4. Download NLTK Data

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('stopwords')"
```

## 5. Verify GPU

```bash
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0), '| VRAM:', round(torch.cuda.get_device_properties(0).total_memory/(1024**3),1), 'GB')"
```

---

## Running the Pipeline

### Quick Sanity Check (10 samples, ~2 min on GPU)

```bash
PYTHONIOENCODING=utf-8 python main.py \
  --model gpt2 \
  --dataset slices/MQuAKE-CF_slice_10 \
  --relation_path data/relation.json \
  --device cuda \
  --NatureL --template --correctConflict
```

### 300 Sample Run (~35 min with gpt2, ~3.5 hrs with vicuna)

```bash
PYTHONIOENCODING=utf-8 python main.py \
  --model gpt2 \
  --dataset slices/MQuAKE-CF_slice01_300 \
  --relation_path data/relation.json \
  --device cuda \
  --loss prob_div \
  --beam_width 5 \
  --num_beams 1 \
  --max_new_tokens 50 \
  --NatureL --template --correctConflict \
  --seed 42
```

### Full 3k Benchmark

```bash
PYTHONIOENCODING=utf-8 python main.py \
  --model gpt2 \
  --dataset MQuAKE-CF-3k \
  --relation_path data/relation.json \
  --device cuda \
  --loss prob_div \
  --beam_width 5 \
  --num_beams 1 \
  --max_new_tokens 50 \
  --NatureL --template --correctConflict \
  --seed 42
```

### Temporal Edits (MQuAKE-T)

```bash
PYTHONIOENCODING=utf-8 python main.py \
  --model gpt2 \
  --dataset MQuAKE-T \
  --relation_path data/relation.json \
  --device cuda \
  --NatureL --template \
  --seed 42
```

---

## Changing the Model

Replace `--model gpt2` with any of:

| Model Key | What It Is           | VRAM Needed | Speed      |
|-----------|----------------------|-------------|------------|
| `gpt2`    | GPT-2 Large (774M)   | ~1.6 GB     | ~7s/sample |
| `gpt2xl`  | GPT-2 XL (1.5B)      | ~3.2 GB     | ~12s/sample|
| `falcon`  | Falcon 1B            | ~2.5 GB     | ~10s/sample|
| `neo`     | GPT-Neo 2.7B         | ~6 GB       | ~18s/sample|
| `vicuna`  | Vicuna 7B (auto FP16)| ~14 GB      | ~45s/sample|
| `llama2`  | LLaMA-2 7B Chat      | ~14 GB      | ~50s/sample|

**Note:** `gpt2xl` matches the "GPT-2 (1.5B)" used in the original RAE paper.

Example with vicuna (auto-loads in FP16):
```bash
PYTHONIOENCODING=utf-8 python main.py \
  --model vicuna \
  --dataset slices/MQuAKE-CF_stratified_300 \
  --relation_path data/relation.json \
  --device cuda \
  --loss prob_div \
  --beam_width 5 \
  --num_beams 1 \
  --max_new_tokens 50 \
  --NatureL --template --correctConflict \
  --seed 42
```

Example with INT8 quantization (best quality for 7B models):
```bash
PYTHONIOENCODING=utf-8 python main.py \
  --model vicuna \
  --dataset slices/MQuAKE-CF_stratified_300 \
  --relation_path data/relation.json \
  --device cuda \
  --load_8bit \
  --loss prob_div \
  --beam_width 5 \
  --num_beams 1 \
  --max_new_tokens 50 \
  --NatureL --template --correctConflict \
  --seed 42
```

**Note:** `llama2` is a gated model. You need to:
1. Request access at https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
2. Login: `huggingface-cli login`

---

## Using the Launcher (Easier)

The launcher provides presets, GPU auto-detection, and pre-flight checks.

```bash
# Interactive guided setup
python launcher/run.py

# Quick preset (gpt2, 10 samples)
python launcher/run.py --preset quick

# Full benchmark (auto-selects best model for your GPU)
python launcher/run.py --preset full

# Custom YAML config
python launcher/run.py --config launcher/configs/example.yaml

# See what command would run without executing
python launcher/run.py --preset quick --dry-run
```

To customize, copy and edit the example config:
```bash
cp launcher/configs/example.yaml launcher/configs/my_run.yaml
# Edit my_run.yaml: change model, dataset, etc.
python launcher/run.py --config launcher/configs/my_run.yaml
```

---

## Running on a Remote Server (iLabs, SLURM, etc.)

### SSH into server and clone
```bash
ssh your_username@server.address
git clone https://github.com/JayanthJammula/RAE.git
cd RAE
```

### Set up Python environment
```bash
# If using conda:
conda create -n rae python=3.10 -y
conda activate rae
pip install torch transformers datasets nltk scikit-learn matplotlib seaborn pyyaml

# Download NLTK data
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('stopwords')"
```

### Copy .pkl files
```bash
# From your local machine:
scp data/Wikidata_triplets_dict.pkl your_username@server:~/RAE/data/
scp data/Wikidata_triplets_dict_T.pkl your_username@server:~/RAE/data/
```

### Run with nohup (keeps running after you disconnect)
```bash
nohup bash -c 'PYTHONIOENCODING=utf-8 python main.py \
  --model vicuna \
  --dataset MQuAKE-CF-3k \
  --relation_path data/relation.json \
  --device cuda \
  --loss prob_div \
  --beam_width 5 \
  --num_beams 1 \
  --max_new_tokens 50 \
  --NatureL --template --correctConflict \
  --seed 42' > run_output.log 2>&1 &

# Check progress:
tail -f run_output.log
```

### Using SLURM (if your server uses it)
```bash
#!/bin/bash
#SBATCH --job-name=rae-eval
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=rae_%j.log

module load python/3.10 cuda/11.8
source activate rae

PYTHONIOENCODING=utf-8 python main.py \
  --model vicuna \
  --dataset MQuAKE-CF-3k \
  --relation_path data/relation.json \
  --device cuda \
  --loss prob_div \
  --beam_width 5 \
  --num_beams 1 \
  --max_new_tokens 50 \
  --NatureL --template --correctConflict \
  --seed 42
```

Save as `run_rae.sh` and submit: `sbatch run_rae.sh`

---

## Time Estimates

| Model   | 10 samples | 300 samples | 3k benchmark |
|---------|-----------|-------------|--------------|
| gpt2    | ~1 min    | ~35 min     | ~5.8 hrs     |
| falcon  | ~2 min    | ~50 min     | ~8.3 hrs     |
| neo     | ~3 min    | ~90 min     | ~15 hrs      |
| vicuna  | ~8 min    | ~3.5 hrs    | ~37 hrs      |
| llama2  | ~8 min    | ~4 hrs      | ~42 hrs      |

Times are approximate, measured on GPU. CPU is ~10x slower.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `UnicodeEncodeError` on Windows | Set `PYTHONIOENCODING=utf-8` before the command |
| `NLTK download failed` | Run the NLTK download command from step 4 manually |
| `FileNotFoundError: .pkl` | Download the KG files from step 3 |
| Run stops early at case ~18 | Already fixed - update your repo with `git pull` |
| `--NatureL: expected one argument` | Already fixed - update your repo with `git pull` |
| `OOM: CUDA out of memory` | Use a smaller model (see model table) |
| `KeyError: 'vicuna'` | Make sure you pulled the latest code |
| Connection drops during long run | Use `nohup` or `screen`/`tmux` (see remote server section) |
