"""Inline NER for cloud deployment (no torch needed)."""

import re

COMPONENT_PATTERNS = [
    "battery", "screen", "display", "charger", "cable",
    "bluetooth", "wifi", "wi-fi", "speaker", "microphone",
    "mic", "camera", "lens", "keyboard", "mouse",
    "trackpad", "touchpad", "port", "usb", "hdmi",
    "adapter", "plug", "power", "button", "hinge", "fan",
    "processor", "cpu", "gpu", "memory", "ram", "storage",
    "ssd", "hard drive", "motherboard", "sensor",
    "antenna", "remote", "controller",
]

ISSUE_PATTERNS = [
    "broke", "broken", "cracked", "defective", "faulty",
    "malfunction", "dead", "garbage", "junk", "useless",
    "stopped working", "not working", "doesn't work",
    "does not work", "won't turn on", "won't charge",
    "overheating", "slow", "laggy", "freezing", "crash",
    "crashed", "glitchy", "disconnecting", "disconnect",
    "keeps disconnecting", "flickering", "dim", "noise",
    "noisy", "buzzing", "static", "distorted", "loose",
    "bent", "scratched", "leak", "leaking", "died",
    "discharged", "drains", "unresponsive",
    "arrived damaged", "missing parts", "wrong item",
    "late delivery", "never arrived", "misleading",
    "not as described", "cheaply made", "poor quality",
    "flimsy", "too small", "too big", "doesn't fit",
    "waste of money", "rip off", "scam", "overpriced",
]

_COMP_RE = re.compile(
    r"\b("
    + "|".join(
        re.escape(p)
        for p in sorted(
            COMPONENT_PATTERNS, key=len, reverse=True
        )
    )
    + r")\b",
    re.IGNORECASE,
)

_ISSUE_RE = re.compile(
    r"\b("
    + "|".join(
        re.escape(p)
        for p in sorted(
            ISSUE_PATTERNS, key=len, reverse=True
        )
    )
    + r")\b",
    re.IGNORECASE,
)


def extract_fast(text_val):
    if not isinstance(text_val, str) or not text_val:
        return {
            "components": [],
            "issues": [],
            "time_refs": [],
        }
    components = list(
        set(m.lower() for m in _COMP_RE.findall(text_val))
    )
    issues = list(
        set(m.lower() for m in _ISSUE_RE.findall(text_val))
    )
    return {
        "components": components,
        "issues": issues,
        "time_refs": [],
    }


def extract_batch_fast(texts):
    return [extract_fast(t) for t in texts]
