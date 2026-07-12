"""Fair-comparison model roster and availability checks."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from model_registry import ALL_SPECS
from paths import (
    LOCAL_DISTILBERT_STUDENT_DIR,
    LOCAL_DISTILGPT2_D1_FT_DIR,
    LOCAL_DISTILGPT2_DIR,
    LOCAL_MODEL_DIR,
)

# Locked roster for fair tables (Q3 A).
FAIR_ROSTER: List[str] = [
    "toxic-bert",
    "tfidf-svm",
    "distilgpt2-d1",
    "distilgpt2-d1-ft",
    "distilbert-student",
]


def model_ready(key: str) -> Tuple[bool, str]:
    """Return (ok, message). Missing optional weights are skipped with a clear reason."""
    if key not in ALL_SPECS:
        return False, f"unknown model key: {key}"

    if key == "toxic-bert":
        if LOCAL_MODEL_DIR.exists():
            return True, f"local {LOCAL_MODEL_DIR.name}"
        return True, "HF hub (unitary/toxic-bert) or cache"

    if key == "tfidf-svm":
        return True, "fits on train split at eval time"

    if key == "distilgpt2-d1":
        if LOCAL_DISTILGPT2_DIR.exists():
            return True, f"local {LOCAL_DISTILGPT2_DIR.name}"
        return True, "HF hub (distilgpt2) or cache"

    if key == "distilgpt2-d1-ft":
        if LOCAL_DISTILGPT2_D1_FT_DIR.exists():
            return True, f"local {LOCAL_DISTILGPT2_D1_FT_DIR.name}"
        return (
            False,
            f"missing {LOCAL_DISTILGPT2_D1_FT_DIR} — run notebooks/10_decoder_finetune_colab.ipynb",
        )

    if key == "distilbert-student":
        if LOCAL_DISTILBERT_STUDENT_DIR.exists():
            return True, f"local {LOCAL_DISTILBERT_STUDENT_DIR.name}"
        return (
            False,
            f"missing {LOCAL_DISTILBERT_STUDENT_DIR} — run notebooks/09_distillation_colab.ipynb",
        )

    return True, "ok"


def roster_status(keys: Optional[List[str]] = None) -> Dict[str, Dict[str, object]]:
    keys = keys or FAIR_ROSTER
    out: Dict[str, Dict[str, object]] = {}
    for key in keys:
        ok, msg = model_ready(key)
        out[key] = {"ready": ok, "detail": msg, "spec": ALL_SPECS.get(key)}
    return out


def print_roster_status(keys: Optional[List[str]] = None) -> List[str]:
    """Print availability table; return list of ready keys."""
    status = roster_status(keys)
    ready: List[str] = []
    for key, info in status.items():
        mark = "OK" if info["ready"] else "SKIP"
        print(f"  [{mark}] {key}: {info['detail']}")
        if info["ready"]:
            ready.append(key)
    return ready


def require_path(path: Optional[Union[Path, str]], label: str) -> Path:
    """Raise a clear error if a filesystem path is None or missing."""
    if path is None:
        raise TypeError(
            f"{label} is None (expected a filesystem path). "
            "Usually the notebook imported the wrong `paths` module — "
            "re-run the setup cell from DeepProjectHate, or restart the kernel."
        )
    p = Path(path)
    return p
