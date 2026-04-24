"""
Rule-based sentence structure classifier.
Classifies sentences as: active, passive, garden_path, relative_clause, other.
Uses heuristic keyword/pattern matching since spaCy is not available.
"""

import re


# Known garden-path patterns (reduced verb / misparse triggers)
GARDEN_PATH_PATTERNS = [
    # "The X VERBed past/by/into ... VERBed" pattern
    r'\b(?:the|a)\s+\w+\s+(?:raced|walked|driven|pushed|pulled|hunted|told|warned|given|shown)\s+(?:past|by|into|through|from)\b',
    # "The X NOUN VERBed" — where NOUN is actually a verb
    r'\bthe\s+(?:old|young|complex|cotton|steel|iron)\s+(?:man|houses|clothing|workers)\b',
    # Classic garden path heads
    r'\b(?:while|when|after|before|since|as)\s+(?:the|a)\s+\w+\s+\w+ed?\s+(?:the|a)\b',
]

# Passive voice indicators
PASSIVE_PATTERNS = [
    r'\b(?:is|are|was|were|been|being|be)\s+\w+(?:ed|en|t)\b',
    r'\b(?:is|are|was|were|been|being|be)\s+(?:being\s+)?\w+(?:ed|en|t)\s+(?:by)\b',
    r'\bwas\s+\w+ed\b',
    r'\bwere\s+\w+ed\b',
    r'\bbeen\s+\w+ed\b',
    r'\b(?:has|have|had)\s+been\s+\w+(?:ed|en|t)\b',
]

# Relative clause indicators
RELATIVE_CLAUSE_PATTERNS = [
    r'\b(?:who|whom|whose|which|that)\s+(?:is|are|was|were|has|have|had|will|would|could|should|might|can|may)\b',
    r',\s*(?:who|which|whom|whose)\s+',
    r'\b(?:the|a)\s+\w+\s+(?:who|which|that)\s+\w+(?:ed|s)?\b',
]


def classify_sentence(sentence: str) -> str:
    """
    Classify a sentence into one of: 'passive', 'garden_path', 'relative_clause', 'active'.
    Uses rule-based heuristics.
    """
    sent_lower = sentence.lower().strip()

    # Check garden path first (most specific)
    for pattern in GARDEN_PATH_PATTERNS:
        if re.search(pattern, sent_lower):
            return "garden_path"

    # Check relative clauses
    for pattern in RELATIVE_CLAUSE_PATTERNS:
        if re.search(pattern, sent_lower):
            return "relative_clause"

    # Check passive voice
    for pattern in PASSIVE_PATTERNS:
        if re.search(pattern, sent_lower):
            return "passive"

    # Default to active
    return "active"


def classify_sentences_batch(sentences: list) -> dict:
    """
    Classify a list of sentences and return a dict mapping sentence -> structure_label.
    Also prints distribution statistics.
    """
    classifications = {}
    counts = {"active": 0, "passive": 0, "garden_path": 0, "relative_clause": 0}

    for sent in sentences:
        label = classify_sentence(sent)
        classifications[sent] = label
        counts[label] += 1

    total = len(sentences)
    print("\n=== Sentence Structure Distribution ===")
    for label, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {label:20s}: {count:4d} ({pct:5.1f}%)")
    print(f"  {'TOTAL':20s}: {total:4d}")

    return classifications
