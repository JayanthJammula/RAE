"""
run.py - Friendly launcher for the RAE evaluation pipeline.

Usage:
    python run.py                       # Interactive guided setup
    python run.py --preset quick        # GPT-2 + 10 samples, fast
    python run.py --preset full         # Best model for your GPU + 3k samples
    python run.py --preset debug        # GPT-2 + 5 samples
    python run.py --config config.yaml  # Load settings from YAML file
    python run.py --preset quick --dry-run   # Show command without running
"""

import sys
import os
import subprocess
import argparse
import shutil
import time
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # run.py lives in launcher/, project root is one up
MAIN_PY = PROJECT_ROOT / "main.py"
DATA_DIR = PROJECT_ROOT / "data"
RELATION_PATH = DATA_DIR / "relation.json"

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "gpt2": {
        "hf_id":       "gpt2-large",
        "vram_gb":     1.6,
        "gated":       False,
        "params":      "774M",
        "description": "GPT-2 Large - lightweight, fast, good for testing",
    },
    "gpt2xl": {
        "hf_id":       "gpt2-xl",
        "vram_gb":     3.2,
        "gated":       False,
        "params":      "1.5B",
        "description": "GPT-2 XL - matches paper's GPT-2 (1.5B)",
    },
    "falcon": {
        "hf_id":       "tiiuae/falcon-1b",
        "vram_gb":     2.5,
        "gated":       False,
        "params":      "1B",
        "description": "Falcon 1B - small model, decent quality",
    },
    "neo": {
        "hf_id":       "EleutherAI/gpt-neo-2.7B",
        "vram_gb":     6.0,
        "gated":       False,
        "params":      "2.7B",
        "description": "GPT-Neo 2.7B - mid-size, good balance",
    },
    "vicuna": {
        "hf_id":       "lmsys/vicuna-7b-v1.1",
        "vram_gb":     14.0,
        "gated":       False,
        "params":      "7B",
        "description": "Vicuna 7B - auto FP16, use --load_8bit for best quality",
    },
    "llama2": {
        "hf_id":       "meta-llama/Llama-2-7b-chat-hf",
        "vram_gb":     14.0,
        "gated":       True,
        "params":      "7B",
        "description": "Llama-2 7B Chat - gated, auto FP16",
    },
}

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    "10 samples":          {"path": "slices/MQuAKE-CF_slice_10",        "count": 10},
    "30 stratified":       {"path": "slices/MQuAKE-CF_stratified_30",   "count": 30},
    "100 samples":         {"path": "slices/MQuAKE-CF_slice_100",       "count": 100},
    "150 stratified":      {"path": "slices/MQuAKE-CF_stratified_150",  "count": 150},
    "300 stratified":      {"path": "slices/MQuAKE-CF_stratified_300",  "count": 300},
    "300 samples (first)": {"path": "slices/MQuAKE-CF_slice01_300",     "count": 300},
    "3k benchmark":        {"path": "MQuAKE-CF-3k",                     "count": 3000},
    "Full dataset":        {"path": "MQuAKE-CF",                        "count": 9218},
}

# ---------------------------------------------------------------------------
# Required data files
# ---------------------------------------------------------------------------

REQUIRED_FILES = [
    "data/relation.json",
    "data/MQuAKE-CF.json",
    "data/train_question_tuple.txt",
    "data/Wikidata_triplets_dict.pkl",
]

PKL_DOWNLOAD_MSG = (
    "The .pkl knowledge graph files (~550 MB each) must be downloaded separately.\n"
    "    See README.md for download links, then place them in data/"
)

# ---------------------------------------------------------------------------
# Time estimates (seconds per sample on GPU)
# ---------------------------------------------------------------------------

TIME_PER_SAMPLE = {
    "gpt2": 7, "gpt2xl": 12, "falcon": 10, "neo": 18, "vicuna": 45, "llama2": 50,
}
CPU_MULTIPLIER = 10

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS = {
    "quick": {
        "model": "gpt2",
        "dataset": "10 samples",
        "beam_width": 2,
        "num_beams": 1,
        "max_new_tokens": 10,
        "disable_attn_viz": True,
    },
    "full": {
        "model": "auto",
        "dataset": "3k benchmark",
        "beam_width": 5,
        "num_beams": 1,
        "max_new_tokens": 50,
        "disable_attn_viz": False,
    },
    "debug": {
        "model": "gpt2",
        "dataset": "10 samples",
        "beam_width": 2,
        "num_beams": 1,
        "max_new_tokens": 10,
        "disable_attn_viz": True,
    },
}


# ===================================================================
# GPU Detection
# ===================================================================

def detect_gpu():
    """Detect GPU availability and VRAM."""
    info = {"available": False, "name": None, "vram_gb": 0.0, "vram_free_gb": 0.0}
    try:
        import torch
        if torch.cuda.is_available():
            info["available"] = True
            info["name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_gb"] = round(props.total_memory / (1024 ** 3), 1)
            free, _ = torch.cuda.mem_get_info(0)
            info["vram_free_gb"] = round(free / (1024 ** 3), 1)
    except Exception:
        pass
    return info


def models_that_fit(vram_gb):
    """Return model keys that fit in available VRAM (with 1GB headroom)."""
    usable = vram_gb - 1.0
    return [k for k, v in MODEL_REGISTRY.items() if v["vram_gb"] <= usable]


def best_model_for_vram(vram_gb):
    """Pick the largest non-gated model that fits."""
    usable = vram_gb - 1.0
    candidates = sorted(
        MODEL_REGISTRY.items(),
        key=lambda kv: kv[1]["vram_gb"],
        reverse=True,
    )
    for key, info in candidates:
        if info["vram_gb"] <= usable and not info["gated"]:
            return key
    return "gpt2"


# ===================================================================
# Pre-flight Checks
# ===================================================================

def run_preflight(model_key, dataset_key, gpu_info):
    """Check for missing files, VRAM issues, gated model access. Returns list of issue strings."""
    issues = []

    # Check required files
    for rel_path in REQUIRED_FILES:
        full_path = PROJECT_ROOT / rel_path
        if not full_path.exists():
            msg = f"MISSING: {rel_path}"
            if rel_path.endswith(".pkl"):
                msg += f"\n    {PKL_DOWNLOAD_MSG}"
            issues.append(msg)

    # Check dataset file
    ds_path = DATASET_REGISTRY[dataset_key]["path"]
    ds_file = DATA_DIR / f"{ds_path}.json"
    if not ds_file.exists():
        issues.append(f"MISSING: data/{ds_path}.json")

    # Check NatureL template
    nl_file = DATA_DIR / "cloze_templates_NL.json"
    if not nl_file.exists():
        issues.append("MISSING: data/cloze_templates_NL.json (needed for --NatureL)")

    # VRAM check
    model_info = MODEL_REGISTRY[model_key]
    if gpu_info["available"]:
        if model_info["vram_gb"] > gpu_info["vram_free_gb"]:
            issues.append(
                f"WARNING: {model_key} needs ~{model_info['vram_gb']} GB VRAM, "
                f"but only {gpu_info['vram_free_gb']} GB free.\n"
                f"    May crash with OOM. Consider a smaller model."
            )
    else:
        issues.append("WARNING: No GPU detected. Running on CPU will be ~10x slower.")

    # Gated model check
    if model_info["gated"]:
        try:
            from huggingface_hub import model_info as hf_info
            hf_info(model_info["hf_id"])
        except Exception:
            issues.append(
                f"WARNING: {model_key} is a gated model ({model_info['hf_id']}).\n"
                f"    1. Visit https://huggingface.co/{model_info['hf_id']} and request access\n"
                f"    2. Run: huggingface-cli login"
            )

    return issues


# ===================================================================
# Terminal UI Helpers
# ===================================================================

def print_header():
    w = min(shutil.get_terminal_size((80, 24)).columns, 64)
    print()
    print("=" * w)
    print("  RAE - Retrieval-Augmented Editing Pipeline")
    print("=" * w)
    print()


def print_box(title, lines):
    w = min(shutil.get_terminal_size((80, 24)).columns - 4, 66)
    print(f"  +{'-' * (w + 2)}+")
    print(f"  | {title:<{w}} |")
    print(f"  +{'-' * (w + 2)}+")
    for line in lines:
        print(f"  | {line:<{w}} |")
    print(f"  +{'-' * (w + 2)}+")
    print()


def prompt_choice(prompt, options, default=0):
    """Show numbered options, return 0-based index."""
    print()
    for i, opt in enumerate(options):
        marker = " *" if i == default else "  "
        print(f"  [{i + 1}]{marker} {opt}")
    print()
    while True:
        raw = input(f"  {prompt} [{default + 1}]: ").strip()
        if raw == "":
            return default
        try:
            choice = int(raw)
            if 1 <= choice <= len(options):
                return choice - 1
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {len(options)}.")


def prompt_yes_no(prompt, default=True):
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        raw = input(f"  {prompt} {suffix}: ").strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please enter y or n.")


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, _ = divmod(remainder, 60)
        return f"{h}h {m}m"


def estimate_time(model_key, sample_count, on_gpu):
    per_sample = TIME_PER_SAMPLE.get(model_key, 20)
    total = per_sample * sample_count
    if not on_gpu:
        total *= CPU_MULTIPLIER
    return total


# ===================================================================
# Interactive Mode
# ===================================================================

def interactive_mode(gpu_info):
    """Guide user through model/dataset/mode selection. Returns config dict."""
    print_header()

    # GPU status
    if gpu_info["available"]:
        print(f"  GPU: {gpu_info['name']}")
        print(f"  VRAM: {gpu_info['vram_gb']} GB total, ~{gpu_info['vram_free_gb']} GB free")
        fitting = models_that_fit(gpu_info["vram_gb"])
    else:
        print("  No GPU detected. Will run on CPU (slow).")
        fitting = list(MODEL_REGISTRY.keys())

    # Step 1: Model
    print("\n  STEP 1: Choose a model")
    model_keys = list(MODEL_REGISTRY.keys())
    model_options = []
    for key in model_keys:
        info = MODEL_REGISTRY[key]
        fit_tag = "OK" if key in fitting else "NEEDS MORE VRAM"
        gated = " [gated]" if info["gated"] else ""
        model_options.append(
            f"{key} ({info['params']}, ~{info['vram_gb']} GB) - {fit_tag}{gated}"
        )
    model_idx = prompt_choice("Pick a model", model_options, default=0)
    model_key = model_keys[model_idx]

    # Step 2: Dataset
    print("\n  STEP 2: Choose a dataset")
    ds_keys = list(DATASET_REGISTRY.keys())
    ds_idx = prompt_choice("Pick a dataset", ds_keys, default=0)
    dataset_key = ds_keys[ds_idx]

    # Step 3: Mode
    print("\n  STEP 3: Choose a run mode")
    mode_options = [
        "Quick   - minimal beams, no attention viz, fast",
        "Standard - balanced settings",
        "Thorough - full beams, attention heatmaps saved",
    ]
    mode_idx = prompt_choice("Pick a mode", mode_options, default=0)

    config = {
        "model": model_key,
        "dataset": dataset_key,
    }

    if mode_idx == 0:
        config.update({"beam_width": 2, "num_beams": 1, "max_new_tokens": 10, "disable_attn_viz": True})
    elif mode_idx == 1:
        config.update({"beam_width": 3, "num_beams": 1, "max_new_tokens": 30, "disable_attn_viz": True})
    else:
        config.update({"beam_width": 5, "num_beams": 1, "max_new_tokens": 50, "disable_attn_viz": False})

    return config


# ===================================================================
# Config File Support
# ===================================================================

def load_yaml_config(path):
    """Load config from YAML file."""
    try:
        import yaml
        with open(path) as f:
            config = yaml.safe_load(f)
    except ImportError:
        # Fallback: simple key:value parser
        config = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    key, val = line.split(":", 1)
                    key, val = key.strip(), val.strip()
                    if val.lower() in ("true", "yes"):
                        val = True
                    elif val.lower() in ("false", "no"):
                        val = False
                    else:
                        try:
                            val = int(val)
                        except ValueError:
                            try:
                                val = float(val)
                            except ValueError:
                                pass
                    config[key] = val

    if "model" not in config:
        print("  ERROR: Config file must specify 'model'")
        sys.exit(1)
    if "dataset" not in config:
        print("  ERROR: Config file must specify 'dataset'")
        sys.exit(1)
    return config


# ===================================================================
# Command Builder
# ===================================================================

def build_command(config, gpu_info):
    """Convert config dict into subprocess command list and env dict."""
    # Resolve dataset path
    dataset_key = config["dataset"]
    if dataset_key in DATASET_REGISTRY:
        ds_path = DATASET_REGISTRY[dataset_key]["path"]
    else:
        ds_path = dataset_key  # raw path from config file

    device = "cuda" if gpu_info["available"] else "cpu"
    device = config.get("device", device)

    cmd = [
        sys.executable, str(MAIN_PY),
        "--model",              config["model"],
        "--dataset",            ds_path,
        "--relation_path",      str(RELATION_PATH),
        "--device",             device,
        "--loss",               config.get("loss", "prob_div"),
        "--beam_width",         str(config.get("beam_width", 5)),
        "--num_beams",          str(config.get("num_beams", 1)),
        "--max_new_tokens",     str(config.get("max_new_tokens", 50)),
        "--seed",               str(config.get("seed", 42)),
        "--temp",               str(config.get("temp", 1.0)),
        "--mode",               config.get("mode", "beam"),
        "--conf_threshold",     str(config.get("conf_threshold", 0.7)),
        "--max_retrieval_rounds", str(config.get("max_retrieval_rounds", 3)),
    ]

    # Boolean flags (always enabled for best results)
    if config.get("NatureL", True):
        cmd.append("--NatureL")
    if config.get("template", True):
        cmd.append("--template")
    if config.get("correctConflict", True):
        cmd.append("--correctConflict")

    # INT8 quantization
    if config.get("load_8bit", False):
        cmd.append("--load_8bit")

    # Optional overrides
    if "template_number" in config:
        cmd.extend(["--template_number", str(config["template_number"])])
    if "entropy_template_number" in config:
        cmd.extend(["--entropy_template_number", str(config["entropy_template_number"])])
    if "starting_line" in config:
        cmd.extend(["--starting_line", str(config["starting_line"])])

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if config.get("disable_attn_viz", False):
        env["RAE_DISABLE_ATTN_VIZ"] = "1"

    return cmd, env


# ===================================================================
# Summary Display
# ===================================================================

def print_run_summary(config, gpu_info):
    model_key = config["model"]
    dataset_key = config["dataset"]
    model_info = MODEL_REGISTRY.get(model_key, {})

    on_gpu = gpu_info["available"] and config.get("device", "cuda") != "cpu"
    device_str = gpu_info["name"] if on_gpu else "CPU"

    if dataset_key in DATASET_REGISTRY:
        ds_info = DATASET_REGISTRY[dataset_key]
        ds_display = f"{ds_info['path']} ({ds_info['count']} samples)"
        sample_count = ds_info["count"]
    else:
        ds_display = dataset_key
        sample_count = 100  # fallback estimate

    eta = format_time(estimate_time(model_key, sample_count, on_gpu))
    attn_str = "OFF" if config.get("disable_attn_viz") else "ON"

    lines = [
        f"Model:       {model_key} ({model_info.get('hf_id', '?')})",
        f"Parameters:  {model_info.get('params', '?')}",
        f"Dataset:     {ds_display}",
        f"Device:      {device_str}",
        f"Attn viz:    {attn_str}",
        f"Beam width:  {config.get('beam_width', 5)}",
        f"Loss:        {config.get('loss', 'prob_div')}",
        f"Est. time:   ~{eta}",
    ]
    print_box("Run Summary", lines)


# ===================================================================
# Main
# ===================================================================

def parse_run_args():
    parser = argparse.ArgumentParser(
        description="RAE launcher - friendly wrapper around main.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python run.py                        Interactive guided setup
              python run.py --preset quick          GPT-2 + 10 samples
              python run.py --preset full           Best model + 3k benchmark
              python run.py --preset debug          GPT-2 + 10 samples, fast
              python run.py --config my_run.yaml    Custom config file
              python run.py --preset quick --dry-run  Show command only
        """),
    )
    parser.add_argument("--preset", choices=["quick", "full", "debug"],
                        help="Use a built-in preset configuration")
    parser.add_argument("--config", type=str,
                        help="Path to a YAML config file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the command without executing it")
    parser.add_argument("--skip-checks", action="store_true",
                        help="Skip pre-flight checks")
    return parser.parse_args()


def main():
    args = parse_run_args()
    gpu_info = detect_gpu()

    # --- Resolve config ---
    if args.config:
        config = load_yaml_config(args.config)
        print(f"\n  Loaded config from {args.config}")
    elif args.preset:
        config = dict(PRESETS[args.preset])
        print(f"\n  Preset: {args.preset}")
        if config["model"] == "auto":
            config["model"] = best_model_for_vram(gpu_info["vram_gb"])
            print(f"  Auto-selected model: {config['model']} "
                  f"(best fit for {gpu_info['vram_gb']} GB VRAM)")
    else:
        config = interactive_mode(gpu_info)

    # --- Pre-flight checks ---
    if not args.skip_checks:
        issues = run_preflight(config["model"], config["dataset"], gpu_info)
        if issues:
            print("\n  Pre-flight checks:\n")
            for issue in issues:
                for line in issue.split("\n"):
                    print(f"    {line}")
                print()

            blocking = [i for i in issues if i.startswith("MISSING:")]
            if blocking:
                print("  Cannot proceed - fix missing files above first.")
                sys.exit(1)

            if not prompt_yes_no("Continue with warnings?", default=True):
                sys.exit(0)

    # --- Summary ---
    print_run_summary(config, gpu_info)

    # --- Build command ---
    cmd, env = build_command(config, gpu_info)
    cmd_str = " ".join(cmd)

    if args.dry_run:
        print("  DRY RUN - would execute:\n")
        print(f"    {cmd_str}\n")
        env_extras = []
        if env.get("RAE_DISABLE_ATTN_VIZ") == "1":
            env_extras.append("RAE_DISABLE_ATTN_VIZ=1")
        if env_extras:
            print(f"    Environment: {', '.join(env_extras)}\n")
        sys.exit(0)

    if not prompt_yes_no("Start the run?", default=True):
        print("  Aborted.")
        sys.exit(0)

    # --- Execute ---
    print(f"\n  Starting RAE pipeline...")
    print(f"  {'-' * 50}\n")

    t0 = time.time()
    try:
        result = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))
        elapsed = time.time() - t0
        print(f"\n  {'-' * 50}")
        print(f"  Finished in {format_time(elapsed)}.")
        if result.returncode != 0:
            print(f"  main.py exited with code {result.returncode}")
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        elapsed = time.time() - t0
        print(f"\n\n  Interrupted after {format_time(elapsed)}.")
        sys.exit(130)


if __name__ == "__main__":
    main()
