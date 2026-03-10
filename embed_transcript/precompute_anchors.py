"""
precompute_anchors.py

Run this script once during setup to compute and save anchor phrase embeddings
as .npy files. Subsequent runs of symptom_embedder.py will load these cached
files instead of re-encoding the anchors on every cold start.

Usage:
    python precompute_anchors.py
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from symptom_embedder import (
    MODEL_NAME,
    CHEST_PAIN_POS_ANCHORS,
    CHEST_PAIN_NEG_ANCHORS,
    BREATHLESSNESS_POS_ANCHORS,
    BREATHLESSNESS_NEG_ANCHORS,
    CHEST_PAIN_POS_FILE,
    CHEST_PAIN_NEG_FILE,
    BREATHLESSNESS_POS_FILE,
    BREATHLESSNESS_NEG_FILE,
)

print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded.")

pairs = [
    (CHEST_PAIN_POS_ANCHORS,       CHEST_PAIN_POS_FILE,       "chest pain positive"),
    (CHEST_PAIN_NEG_ANCHORS,       CHEST_PAIN_NEG_FILE,       "chest pain negative"),
    (BREATHLESSNESS_POS_ANCHORS,   BREATHLESSNESS_POS_FILE,   "breathlessness positive"),
    (BREATHLESSNESS_NEG_ANCHORS,   BREATHLESSNESS_NEG_FILE,   "breathlessness negative"),
]

for phrases, cache_file, label in pairs:
    print(f"Encoding {len(phrases)} {label} anchor phrases...")
    embeddings = model.encode(phrases, convert_to_numpy=True)
    np.save(cache_file, embeddings)
    print(f"Saved: {cache_file}")

print("Done. Anchor files are ready for use by symptom_embedder.py.")
