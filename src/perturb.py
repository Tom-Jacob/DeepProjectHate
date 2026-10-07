"""Text perturbations for robustness stress-testing.

We take comments a model already flags as toxic and apply surface-level noise
that a human still reads as toxic (typos, leetspeak, spacing, repetition,
casing). A robust detector should keep flagging them; an "evasion" is when the
perturbed comment slips through as non-toxic. This measures brittleness to the
kind of obfuscation real users employ to dodge moderation.
"""
import random
import re
from typing import Callable, Dict

from paths import SEED

LEET_MAP = {"a": "@", "e": "3", "i": "1", "o": "0", "s": "$", "t": "7"}


def leetspeak(text: str, **_) -> str:
    """Replace common letters with look-alike symbols/digits."""
    return "".join(LEET_MAP.get(c.lower(), c) for c in text)


def add_typos(text: str, rate: float = 0.1, seed: int = SEED) -> str:
    """Randomly swap adjacent characters in a fraction of words."""
    rng = random.Random(seed + len(text))
    words = text.split()
    out = []
    for w in words:
        if len(w) > 3 and rng.random() < rate * 5:  # amplify: per-word chance
            i = rng.randint(0, len(w) - 2)
            w = w[:i] + w[i + 1] + w[i] + w[i + 2 :]
        out.append(w)
    return " ".join(out)


def space_out(text: str, **_) -> str:
    """Insert spaces between characters of longer words (c a t)."""
    words = text.split()
    return " ".join(" ".join(w) if len(w) > 3 else w for w in words)


def repeat_chars(text: str, seed: int = SEED, **_) -> str:
    """Stretch vowels (soooo baaad) to mimic emphatic obfuscation."""
    rng = random.Random(seed + len(text))
    return re.sub(r"[aeiou]", lambda m: m.group(0) * rng.randint(2, 4), text)


def upper_case(text: str, **_) -> str:
    """Shout everything in caps."""
    return text.upper()


def punct_noise(text: str, **_) -> str:
    """Sprinkle dots between characters of longer words (b.a.d)."""
    words = text.split()
    return " ".join(".".join(w) if len(w) > 3 else w for w in words)


# Registry: name -> perturbation function. All take (text, **kwargs).
PERTURBATIONS: Dict[str, Callable[..., str]] = {
    "typos": add_typos,
    "leetspeak": leetspeak,
    "space_out": space_out,
    "repeat_chars": repeat_chars,
    "upper_case": upper_case,
    "punct_noise": punct_noise,
}


def apply_perturbation(name: str, text: str, **kwargs) -> str:
    if name not in PERTURBATIONS:
        raise KeyError(f"unknown perturbation {name!r}; choices: {list(PERTURBATIONS)}")
    return PERTURBATIONS[name](text, **kwargs)
