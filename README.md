# RAE: Adaptive Retrieval Augmented Knowledge Editing for Multi-Hop QA

## Overview

Large Language Models (LLMs) store vast amounts of world knowledge, but that knowledge goes stale. When facts change (a new president is elected, a company merges, a scientific finding is revised), retraining or fine-tuning an LLM is expensive and slow. **Knowledge Graph (KG) Editing** offers a lightweight alternative: update the external knowledge graph and let the model reason over the corrected facts at inference time.

**RAE** tackles the hardest variant of this problem: **multi-hop questions over edited facts**. A question like *"What is the capital of the country where the creator of Tetris was born?"* requires chaining multiple facts together, and if any one of those facts has been edited, the entire answer changes.

The original RAE framework introduces two key ideas:

1. **Mutual-Information-Maximized Retrieval** - Uses probability divergence scoring to retrieve the most relevant fact chains from the knowledge graph, rather than relying on surface-level similarity.
2. **Entropy-Based Self-Pruning** - Automatically removes redundant or noisy facts from the retrieved set by measuring each fact's contribution to answer entropy, keeping only what the model actually needs.

This repository extends and fixes the original with the following contributions:

### Adaptive Retrieval Stopping

The original RAE ran retrieval for a fixed number of hops regardless of question complexity. This version introduces a confidence-driven early stopping mechanism:

- A `retrieval_confidence()` function measures how many ground-truth fact chain members have already been retrieved.
- The retrieval loop runs up to `--max_retrieval_rounds` iterations, calling `multi_hop_search` with an increasing hop count each round.
- If confidence exceeds `--conf_threshold`, retrieval stops early, avoiding over-fetching and reducing noise for simpler questions.

This required modifying `multi_hop_search` to accept a `rounds` parameter so the outer loop can dynamically control retrieval depth per question.

### Attention Heatmap Visualization

After each QA evaluation, the top-K most focused attention heads are identified and saved as heatmaps. Up to 5 correct and 5 incorrect cases are captured, providing insight into where the model attends when it succeeds or fails at multi-hop reasoning. Disable with `RAE_DISABLE_ATTN_VIZ=1`.

### Entropy Normalization Bug Fix

The original `facts_entropy` method crashed with a divide-by-zero when all entropy values were equal (i.e., no fact contributed differently). Fixed with a `val_range > 0` guard, falling back to a zero vector so pruning degrades gracefully instead of erroring out.

### Improved QA Prompting

Facts are now presented to the model as a structured bullet list instead of a comma-separated string, and the generated answer is truncated at stop tokens (`\n`, `Question:`, `Facts:`) to prevent prompt leakage into the answer.

### Expanded Model Support

Added `gpt2xl` (GPT-2 XL, 1.5B) as a supported model key. Also added `--load_8bit` to load any supported model in INT8 quantization via bitsandbytes, halving VRAM requirements for larger models like Vicuna and LLaMA-2. Large models (`vicuna`, `llama2`) are automatically loaded in FP16 when a GPU is available.

## Why This Matters

| Problem | RAE's Solution |
|---------|---------------|
| LLM knowledge becomes outdated | Edit the KG, not the model |
| Multi-hop questions break with single edits | Chain-aware retrieval across multiple hops |
| Too many retrieved facts confuse the LLM | Entropy-based pruning keeps only relevant facts |
| Retraining is expensive | Zero retraining required - works at inference time |

### Real-World Applications

- **Enterprise Knowledge Management** - When company facts change (CEO, HQ, acquisitions), downstream QA systems give correct multi-hop answers without retraining.
- **Medical & Legal QA** - Updated clinical guidelines or regulations are immediately reflected in multi-hop reasoning.
- **News & Current Events** - Patch the knowledge graph with breaking news; complex questions get correct answers instantly.
- **Knowledge Base Verification** - After bulk edits to Wikidata/Wikipedia, verify that multi-hop QA still produces correct answers.

## Project Structure

```
RAE/
├── main.py                    # Main evaluation pipeline
├── model.py                   # Extract (fact retrieval) and Prune (entropy-based pruning) classes
├── utils_func.py              # Utilities: data loading, NER, QA evaluation, matching
├── create_dataset_slices.py   # Create reproducible dataset subsets for testing
├── requirements.txt           # Python dependencies
├── data/
│   ├── MQuAKE-CF.json         # Counterfactual training dataset (~9k examples)
│   ├── MQuAKE-CF-3k.json      # 3k counterfactual test subset
│   ├── MQuAKE-T.json          # Temporal edits dataset
│   ├── relation.json          # Wikidata relation mappings
│   ├── cloze_templates_NL.json # Natural language relation templates
│   ├── train_question_tuple.txt # Question tuples for ICL example selection
│   └── slices/                # Small dataset slices for quick testing
├── preprocess/
│   └── edit_KG.py             # Apply counterfactual edits to the knowledge graph
└── wiki_api/
    ├── wikidata.py            # Wikidata entity/ID conversion with caching
    ├── Wiki.py                # Wikipedia search engine wrapper
    └── strings.py             # NLP utilities (lemmatization, tokenization, stemming)
```

## Setup

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)

### Installation

```bash
git clone https://github.com/JayanthJammula/RAE.git
cd RAE
pip install -r requirements.txt
```

### Data

The MQuAKE datasets and relation files are already included in `data/`.

**Knowledge Graph files** (`.pkl`) need to be downloaded:
- [KG for MQuAKE-CF-3k](https://outlookuga-my.sharepoint.com/:u:/g/personal/ys07245_uga_edu/Ec0O9oUzka5LuNwK3M8FL-YBG3zw7mAdme7V9S9l4cbt7Q?e=3TKvcE)
- [KG for MQuAKE-T](https://outlookuga-my.sharepoint.com/:u:/g/personal/ys07245_uga_edu/EckS-8zKM75MgqmJmQH8NQMByT___C5lNyZaIsOXHQXvIQ?e=VVwF2F)
- [Original Wikidata KG](https://outlookuga-my.sharepoint.com/:u:/g/personal/ys07245_uga_edu/EbbXuq1FumtFkH3B0qmb2bMBgRdyXbayUNAevKsKvtBVUw?e=Xy3QsY) (based on [Wikidata5m](https://deepgraphlearning.github.io/project/wikidata5m))

Place all `.pkl` files in the `data/` directory.

To build a custom edited KG from the MQuAKE-T dataset:
```bash
python preprocess/edit_KG.py
```

## Supported Models

| Model Key | Model | Parameters |
|-----------|-------|------------|
| `gpt2` | GPT-2 Large | 774M |
| `gpt2xl` | GPT-2 XL | 1.5B |
| `falcon` | Falcon-1B | 1.3B |
| `neo` | GPT-Neo 2.7B | 2.7B |
| `vicuna` | Vicuna-7B | 7B |
| `llama2` | LLaMA-2-7B Chat | 7B |

## Usage

### Quick Test (10 samples)

```bash
python main.py --model gpt2 --dataset slices/MQuAKE-CF_slice_10 --relation_path data/relation.json --NatureL --template --correctConflict --device cuda
```

### Full Evaluation on MQuAKE-CF-3k

```bash
python main.py --model gpt2 --dataset MQuAKE-CF-3k --relation_path data/relation.json --NatureL --template --correctConflict --device cuda
```

### Evaluation on MQuAKE-T (Temporal Edits)

```bash
python main.py --model gpt2 --dataset MQuAKE-T --relation_path data/relation.json --NatureL --template --device cuda
```

### Creating Custom Dataset Slices

```bash
python create_dataset_slices.py
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model key (see supported models table) | *required* |
| `--dataset` | Dataset name under `data/` (without `.json`) | *required* |
| `--relation_path` | Path to relation mappings | *required* |
| `--device` | `cuda` or `cpu` | `cpu` |
| `--NatureL` | Convert triples to natural language statements | off |
| `--template` | Use ICL examples for in-context learning | off |
| `--correctConflict` | Handle editing conflicts in MQuAKE-CF | off |
| `--template_number` | Number of ICL examples for fact retrieval | 5 |
| `--entropy_template_number` | Number of ICL examples for pruning | 5 |
| `--loss` | Scoring function: `prob_div` or `prob_div_log` | `prob_div` |
| `--beam_width` | Beam width for fact chain search | 5 |
| `--max_retrieval_rounds` | Maximum adaptive retrieval rounds | 3 |
| `--conf_threshold` | Confidence threshold to stop early retrieval | 0.7 |
| `--num_beams` | Beam search size for answer generation | 1 |
| `--max_new_tokens` | Max tokens to generate for answers | 50 |
| `--temp` | Generation temperature | 1.0 |
| `--starting_line` | Resume evaluation from case N | 0 |
| `--seed` | Random seed for reproducibility | 42 |
| `--load_8bit` | Load model in INT8 quantization via bitsandbytes (reduces VRAM usage) | off |

### Key Arguments Explained

- **`--NatureL`**: Transforms KG triples like `(Q123, P456, Q789)` into readable sentences like *"Albert Einstein was born in Ulm."* This significantly improves retrieval quality.

- **`--template`**: Builds "question + fact chain" ICL examples from the MQuAKE-CF training set (~9k examples) to help the LLM understand the retrieval and QA tasks.

- **`--correctConflict`**: Handles a subtle issue in MQuAKE-CF-3k where both the original and edited versions of a fact are needed to answer different questions in the same case.

- **`--loss`**: Controls how fact chains are scored. `prob_div` uses raw probability divergence; `prob_div_log` weights it by log probability, emphasizing high-confidence retrievals.

## Output Metrics

The pipeline reports these metrics:

| Metric | Description |
|--------|-------------|
| `raw_exact_match_acc` | All ground truth facts found in retrieved facts |
| `raw_par_match_acc` | At least one ground truth fact found in retrieved facts |
| `prun_exact_match_acc` | All ground truth facts found after pruning |
| `prun_par_match_acc` | At least one ground truth fact found after pruning |
| `raw_ans_acc` | QA accuracy using all retrieved facts |
| `prun_ans_acc` | QA accuracy using pruned facts |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `RAE_DISABLE_ATTN_VIZ=1` | Disable attention heatmap generation (saves time) |
| `PYTHONIOENCODING=utf-8` | Required on Windows for Unicode entity names |

## Citation

This project is based on the following paper. If you find this work helpful, please cite:

> **Retrieval-Enhanced Knowledge Editing in Language Models for Multi-Hop Question Answering**
> Yucheng Shi, Qiaoyu Tan, Xuansheng Wu, Shaochen Zhong, Kaixiong Zhou, Ninghao Liu
> *CIKM 2024* | [Paper](https://arxiv.org/abs/2403.19631) | [Original Repo](https://github.com/sycny/RAE)
