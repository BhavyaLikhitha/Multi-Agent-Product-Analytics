"""spaCy NER pipeline for extracting product components, issues, and time references."""

import spacy
from spacy.language import Language

COMPONENT_PATTERNS = [
    "battery",
    "screen",
    "display",
    "charger",
    "cable",
    "bluetooth",
    "wifi",
    "wi-fi",
    "speaker",
    "microphone",
    "mic",
    "camera",
    "lens",
    "keyboard",
    "mouse",
    "trackpad",
    "touchpad",
    "port",
    "usb",
    "hdmi",
    "adapter",
    "plug",
    "power",
    "button",
    "hinge",
    "fan",
    "processor",
    "cpu",
    "gpu",
    "memory",
    "ram",
    "storage",
    "ssd",
    "hard drive",
    "motherboard",
    "sensor",
    "antenna",
    "remote",
    "controller",
]

ISSUE_PATTERNS = [
    "broke",
    "broken",
    "cracked",
    "defective",
    "faulty",
    "malfunction",
    "dead",
    "stopped working",
    "not working",
    "doesn't work",
    "won't turn on",
    "won't charge",
    "overheating",
    "overheat",
    "slow",
    "laggy",
    "lag",
    "freezing",
    "freeze",
    "crash",
    "crashed",
    "disconnecting",
    "disconnect",
    "flickering",
    "flicker",
    "dim",
    "noise",
    "noisy",
    "buzzing",
    "static",
    "distorted",
    "loose",
    "bent",
    "scratched",
    "dent",
    "peeling",
    "leak",
    "leaking",
]


def build_entity_ruler(nlp: Language) -> Language:
    """Add EntityRuler with product component and issue patterns."""
    ruler = nlp.add_pipe("entity_ruler", before="ner")

    component_rules = []
    for p in COMPONENT_PATTERNS:
        # Use token-level LOWER matching for case-insensitivity
        tokens = p.split()
        if len(tokens) == 1:
            component_rules.append(
                {"label": "PRODUCT_COMPONENT", "pattern": [{"LOWER": tokens[0]}]}
            )
        else:
            component_rules.append(
                {
                    "label": "PRODUCT_COMPONENT",
                    "pattern": [{"LOWER": t} for t in tokens],
                }
            )

    issue_rules = []
    for p in ISSUE_PATTERNS:
        tokens = p.split()
        if len(tokens) == 1:
            issue_rules.append(
                {"label": "ISSUE_TYPE", "pattern": [{"LOWER": tokens[0]}]}
            )
        else:
            issue_rules.append(
                {"label": "ISSUE_TYPE", "pattern": [{"LOWER": t} for t in tokens]}
            )

    ruler.add_patterns(component_rules + issue_rules)
    return nlp


def load_nlp():
    """Load spaCy model with custom entity ruler."""
    nlp = spacy.load("en_core_web_sm")
    nlp = build_entity_ruler(nlp)
    return nlp


def extract_entities(doc) -> dict:
    """Extract structured entities from a processed spaCy doc."""
    components = []
    issues = []
    time_refs = []

    for ent in doc.ents:
        if ent.label_ == "PRODUCT_COMPONENT":
            components.append(ent.text.lower())
        elif ent.label_ == "ISSUE_TYPE":
            issues.append(ent.text.lower())
        elif ent.label_ in ("DATE", "TIME"):
            time_refs.append(ent.text)

    return {
        "components": list(set(components)),
        "issues": list(set(issues)),
        "time_refs": list(set(time_refs)),
    }


def extract_from_text(nlp: Language, text: str) -> dict:
    """Process a single review text and return entities."""
    doc = nlp(text)
    return extract_entities(doc)


def extract_batch(nlp: Language, texts: list[str], batch_size: int = 100) -> list[dict]:
    """Process a batch of review texts using spaCy's pipe() for speed."""
    results = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        results.append(extract_entities(doc))
    return results
