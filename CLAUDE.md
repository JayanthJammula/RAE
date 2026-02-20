# CLAUDE.md - Project Context for AI Assistants

## Project Overview

RAE (Retrieval-Augmented Editing) is a framework for multi-hop question answering over edited knowledge graphs. It retrieves fact chains from a Wikidata-based KG, prunes redundant facts using entropy, and generates answers using in-context learning with causal LMs. Based on the CIKM 2024 paper by Shi et al.

Repository: https://github.com/JayanthJammula/RAE

## Architecture

```
main.py                    Entry point — orchestrates retrieval, pruning, QA, metrics
model.py                   Extract (fact retrieval via prob divergence) + Prune (entropy-based)
utils_func.py              Data loading, NER pipeline, QA evaluation, answer matching
launcher/run.py            Friendly launcher with presets, YAML configs, GPU detection
preprocess/edit_KG.py      Applies counterfactual edits to the KG from MQuAKE dataset
wiki_api/wikidata.py       Wikidata entity/ID conversion with caching
wiki_api/Wiki.py           Wikipedia search engine wrapper
wiki_api/strings.py        NLP utilities — lemmatization, tokenization, stemming
create_dataset_slices.py   Create reproducible dataset subsets for testing
```

## Key Data Files

- `data/MQuAKE-CF.json` / `MQuAKE-CF-3k.json` — Counterfactual QA dataset (9k / 3k examples)
- `data/MQuAKE-T.json` — Temporal edits dataset
- `data/Wikidata_triplets_dict.pkl` — Edited KG for CF (~550MB, not in git)
- `data/Wikidata_triplets_dict_MQuAKE-T.pkl` — Edited KG for temporal
- `data/relation.json` — Wikidata relation ID to name mapping
- `data/cloze_templates_NL.json` — Natural language templates for KG triples
- `data/train_question_tuple.txt` — Question tuples for ICL example selection
- `data/slices/` — Dataset subsets for testing:
  - `MQuAKE-CF_stratified_30.json` — 30 cases (10 per hop), quick sanity check
  - `MQuAKE-CF_stratified_150.json` — 150 cases (50 per hop), fast benchmark
  - `MQuAKE-CF_stratified_300.json` — 300 cases (100 per hop), comparable to paper
  - `MQuAKE-CF_slice_10.json` / `slice_100.json` — first-N slices (not stratified)

## Supported Models

| Key      | HuggingFace ID                        | VRAM   | Notes                    |
|----------|---------------------------------------|--------|--------------------------|
| gpt2     | gpt2-large                            | ~1.6GB | Default, fast            |
| gpt2xl   | gpt2-xl                               | ~3.2GB | Matches paper's GPT-2    |
| falcon   | tiiuae/falcon-1b                      | ~2.5GB | Good balance             |
| neo      | EleutherAI/gpt-neo-2.7B              | ~6GB   | Mid-size                 |
| vicuna   | lmsys/vicuna-7b-v1.1                  | ~14GB  | Auto FP16, use --load_8bit |
| llama2   | meta-llama/Llama-2-7b-chat-hf        | ~14GB  | Gated, needs HF login    |

## Critical Implementation Details

### Module-Level Arg Parsing in utils_func.py
`utils_func.py` calls `parse_known_args()` at **import time** (line 44). This was changed from `parse_args()` to avoid consuming arguments meant for `main.py`. Boolean args use `action='store_true'` to avoid type conflicts. Any new args added to `main.py` must not clash with the ones in `utils_func.py`.

### FP16 Precision for Large Models
When running vicuna/llama2 in FP16 (`torch_dtype=torch.float16`), logits must be cast to FP32 before softmax in:
- `model.py` line 30: `relation_prob()` — retrieval scoring
- `model.py` line 177: `ans_entroy()` — entropy calculation

Without `.float()`, softmax on FP16 logits produces degenerate probability distributions due to outlier activation channels, causing all-zero entropy values and broken retrieval scoring.

### Attention Visualization Early Stop
`main.py` lines 325-327: After collecting 5 correct + 5 wrong attention heatmaps, `attn_supported` is set to `False` to stop collection but **continue evaluation**. Previously this was a `break` that killed the entire evaluation loop.

### NER Pipeline Runs on CPU
The Babelscape NER model in `utils_func.py` intentionally runs on CPU to save GPU VRAM for the main model. The warning "no `device` argument passed to Pipeline" is expected and harmless.

### Answer Truncation
Generated answers are truncated at `\n`, `Question:`, or `Facts:` to prevent prompt leakage (utils_func.py, after the `tokenizer.decode` call).

### Variable Naming in preprocess/edit_KG.py
Uses `head_e` and `rel` (not `head_ent` / `relation`) for KG triple components. These were previously undefined causing NameError crashes.

### NLTK Package Names
Correct package names for `nltk.download()`:
- `wordnet` (not `word_tokenize`)
- `punkt_tab` (not `pos_tag`)
- `averaged_perceptron_tagger_eng` (not `PorterStemmer`)
- `stopwords`

### Entropy Normalization
`model.py` `facts_entropy()`: When all entropy values are equal, `val_range == 0` triggers a guard that returns a zero vector instead of dividing by zero.

### Wikidata API Error Handling
`wiki_api/wikidata.py`: `entity2id()` returns `None` on failure (not `'not applicable'`). `id2entity()` returns `"Not Applicable"` (not `'xxxxxxxx'`). `Related()` checks for `None` (not `"Not Applicable"`).

## Common Run Commands

```bash
# Quick test (10 samples, gpt2)
PYTHONIOENCODING=utf-8 python main.py --model gpt2 --dataset slices/MQuAKE-CF_slice_10 --relation_path data/relation.json --device cuda --NatureL --template --correctConflict

# 300 samples with vicuna (needs FP16 edit in main.py line 141)
RAE_DISABLE_ATTN_VIZ=1 PYTHONIOENCODING=utf-8 python main.py --model vicuna --dataset slices/MQuAKE-CF_slice01_300 --relation_path data/relation.json --device cuda --loss prob_div --beam_width 5 --num_beams 1 --max_new_tokens 50 --NatureL --template --correctConflict --seed 42

# Using the launcher
python launcher/run.py --preset quick --dry-run
python launcher/run.py --config launcher/configs/example.yaml
```

## Environment Variables

- `PYTHONIOENCODING=utf-8` — Required on Windows for Unicode entity names (e.g., Kolinda Grabar-Kitarovic)
- `RAE_DISABLE_ATTN_VIZ=1` — Disables attention heatmap generation and saves time

## Known Limitations

- `retrieval_confidence` always returns 0.0 because ground truth strings (from `new_triples_labeled`) don't exactly match the NatureL-converted retrieved text. This is a string matching mismatch, not a bug — adaptive retrieval always runs for max rounds.
- `conf_threshold` (default 0.7) is never reached, so early stopping never triggers.
- Vicuna/LLaMA in FP16 may still have slightly degraded retrieval quality due to activation outlier channels. INT8 quantization (`load_in_8bit=True`) would help but is not yet implemented.
- The `.pkl` KG files (~550MB each) are too large for git and must be downloaded separately.

## Planned Improvements

- Add `--load_8bit` flag for bitsandbytes INT8 quantization (better FP16 handling for 7B models)
- Auto-save results to JSON file after each run
- Fix `retrieval_confidence` string matching to work with NatureL format
- Add results comparison tool across models/datasets
