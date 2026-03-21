"""Data validation and argument bounds checking for RAE pipeline."""

import logging
import os
import pickle

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when dataset validation encounters a fatal issue."""
    pass


def validate_dataset_entry(entry, index):
    """Validate a single dataset entry. Returns list of warnings.
    Raises DataValidationError for fatal issues."""
    warnings = []
    required_keys = ["case_id", "questions", "new_answer", "new_answer_alias", "orig"]

    for key in required_keys:
        if key not in entry:
            raise DataValidationError(f"Entry {index}: missing required key '{key}'")

    if not isinstance(entry["questions"], list) or len(entry["questions"]) == 0:
        raise DataValidationError(f"Entry {index}: 'questions' must be a non-empty list")

    orig = entry.get("orig", {})
    orig_required = ["new_triples", "new_triples_labeled"]
    for key in orig_required:
        if key not in orig:
            raise DataValidationError(f"Entry {index}: 'orig' missing required key '{key}'")

    if not isinstance(orig["new_triples_labeled"], list):
        warnings.append(f"Entry {index}: 'new_triples_labeled' is not a list")

    return warnings


def validate_dataset(data, path):
    """Validate entire dataset. Logs warnings, raises on fatal errors."""
    if not isinstance(data, list):
        raise DataValidationError(f"Dataset at {path} must be a JSON list, got {type(data).__name__}")

    if len(data) == 0:
        raise DataValidationError(f"Dataset at {path} is empty")

    logger.info(f"Validating dataset: {path} ({len(data)} entries)")
    total_warnings = []
    for i, entry in enumerate(data):
        w = validate_dataset_entry(entry, i)
        total_warnings.extend(w)

    for w in total_warnings:
        logger.warning(f"Dataset validation: {w}")

    logger.info(f"Dataset validation passed ({len(total_warnings)} warnings)")


def safe_load_pickle(path):
    """Load pickle with error handling, size check, type check."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pickle file not found: {path}")

    file_size = os.path.getsize(path)
    if file_size == 0:
        raise DataValidationError(f"Pickle file is empty (0 bytes): {path}")

    logger.info(f"Loading pickle: {path} ({file_size / 1024 / 1024:.1f} MB)")

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except (pickle.UnpicklingError, EOFError) as e:
        raise DataValidationError(f"Failed to unpickle {path}: {e}")

    if not isinstance(data, dict):
        raise DataValidationError(
            f"Expected dict from {path}, got {type(data).__name__}"
        )

    logger.info(f"Loaded pickle: {len(data)} entries")
    return data


def validate_args(args):
    """Bounds-check args. Returns list of warnings."""
    warnings = []

    if args.beam_width <= 0:
        warnings.append(f"beam_width must be > 0, got {args.beam_width}")
    if args.num_beams <= 0:
        warnings.append(f"num_beams must be > 0, got {args.num_beams}")
    if args.max_new_tokens <= 0:
        warnings.append(f"max_new_tokens must be > 0, got {args.max_new_tokens}")
    if not (0 <= args.conf_threshold <= 1):
        warnings.append(f"conf_threshold should be 0-1, got {args.conf_threshold}")
    if not (1 <= args.max_retrieval_rounds <= 10):
        warnings.append(f"max_retrieval_rounds should be 1-10, got {args.max_retrieval_rounds}")
    if args.temp <= 0:
        warnings.append(f"temp must be > 0, got {args.temp}")
    if args.seed < 0:
        warnings.append(f"seed should be >= 0, got {args.seed}")

    return warnings
