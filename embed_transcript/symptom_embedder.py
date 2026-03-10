"""
symptom_embedder.py

Detects chest pain and breathlessness from transcribed patient speech
using KNN classification over positive and negative anchor phrase clusters.

Model: sentence-transformers/all-MiniLM-L6-v2
  - General-purpose sentence transformer trained on conversational and
    paraphrase data. ~80MB, fast on CPU, suitable for Raspberry Pi 5.
  - Chosen over biomedical models (e.g. PubMedBERT) because patients
    describe symptoms in plain informal language, not medical terminology.

Scoring approach:
  For each symptom we define two anchor sets: positive (symptom present)
  and negative (symptom absent/negated). At inference time we find the k
  most similar anchors by cosine similarity across both sets combined, then
  take a majority vote. This handles negation naturally — "I can breathe fine"
  ranks closer to negative anchors than positive ones.

  vote_score = (number of positive anchors in top-k) / k
  Flagged as detected if vote_score >= 0.5 (simple majority).

ONNX note: for lower latency on the Pi, this model can later be exported
and quantized using optimum[onnxruntime] to reduce inference time.
"""

import os
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ANCHOR_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_K = 5  # number of nearest neighbours to consider per symptom

# Cache file paths
CHEST_PAIN_POS_FILE = os.path.join(ANCHOR_DIR, "chest_pain_pos_anchors.npy")
CHEST_PAIN_NEG_FILE = os.path.join(ANCHOR_DIR, "chest_pain_neg_anchors.npy")
BREATHLESSNESS_POS_FILE = os.path.join(ANCHOR_DIR, "breathlessness_pos_anchors.npy")
BREATHLESSNESS_NEG_FILE = os.path.join(ANCHOR_DIR, "breathlessness_neg_anchors.npy")

# ---------------------------------------------------------------------------
# Anchor phrase definitions
# ---------------------------------------------------------------------------

CHEST_PAIN_POS_ANCHORS = [
    "chest pain",
    "chest tightness",
    "chest pressure",
    "my chest hurts",
    "squeezing in my chest",
    "stabbing chest",
    "burning chest",
    "pain in my chest",
    "tight feeling in chest",
    "heavy feeling on chest",
    "angina",
    "precordial pain",
    "heart pain",
    "cardiac pain",
    "chest discomfort",
    "crushing chest pain",
    "aching in my chest",
    "my chest feels tight",
    "pressure on my sternum",
    "pain radiating from chest",
    # indirect cardiac/chest expressions
    "my heart feels weird",
    "something is wrong with my heart",
    "heart palpitations",
    "my heart is acting strange",
    "irregular heartbeat feeling",
    # vague patient descriptions of chest discomfort
    "my chest feels bad",
    "something feels wrong with my chest",
    "my chest does not feel right",
    "my chest feels off",
]

CHEST_PAIN_NEG_ANCHORS = [
    # chest specifically fine
    "my chest feels fine",
    "no chest pain",
    "chest feels normal",
    "no pressure in my chest",
    "my chest is comfortable",
    "heart feels normal",
    "no discomfort in my chest",
    "chest is fine",
    "nothing wrong with my chest",
    "no tightness in chest",
    "no chest complaints",
    "my heart is fine",
    "I have no chest issues",
    "no pain in my heart",
    "I have no chest pain",
    # breathlessness without chest pain — teaches the model that
    # breathing complaints are not chest pain
    "I cannot breathe",
    "I am short of breath",
    "difficulty breathing",
    "I am struggling to breathe",
    "I keep gasping for air",
    # other body complaints — prevents unrelated pain from triggering
    "I have a headache",
    "my leg hurts",
    "my back hurts",
    "I feel nauseous",
    "my arm is sore",
    # noise and general statements
    "nothing is wrong",
    "everything is fine",
    "hello",
    "hi",
    "yes",
    "I do not know",
    "okay",
]

BREATHLESSNESS_POS_ANCHORS = [
    "breathlessness",
    "shortness of breath",
    "difficulty breathing",
    "can't catch my breath",
    "dyspnea",
    "trouble breathing",
    "hard to breathe",
    "out of breath",
    "laboured breathing",
    "winded",
    "I cannot breathe properly",
    "struggling to breathe",
    "my breathing is difficult",
    "not enough air",
    "feeling suffocated",
    "breathless",
    "air hunger",
    "I feel like I am suffocating",
    "I keep gasping for air",
    "lungs feel heavy",
    # additional colloquial variants
    "I am winded",
    "I cannot get enough air",
    "I feel breathless",
    # direct statements that share vocabulary with negative anchors
    # ("I can breathe fine") and must outrank them in similarity
    "I am unable to breathe",
    "I cannot breathe at all",
    "I can barely breathe",
    "I am short of breath",
    "I have shortness of breath",
    "every breath is a struggle",
    "I am gasping for breath",
]

BREATHLESSNESS_NEG_ANCHORS = [
    # breathing specifically fine
    "I can breathe fine",
    "breathing is normal",
    "no trouble breathing",
    "I can breathe easily",
    "my breathing is fine",
    "no shortness of breath",
    "breathing comfortably",
    "lungs feel clear",
    "I breathe without difficulty",
    "no breathing problems",
    "air comes in easily",
    "breathing feels normal",
    "I have no breathing issues",
    "not short of breath",
    "my lungs feel fine",
    # chest pain without breathlessness — teaches the model that
    # chest complaints are not breathlessness
    "my chest hurts",
    "I have chest pain",
    "chest pressure",
    "squeezing in my chest",
    "my chest is tight",
    "cardiac pain",
    # squeezing and pressure sensations map to chest pain, not breathlessness
    "squeezing sensation in my chest",
    "pressure feeling in my chest",
    "something is sitting on my chest",
    "something pressing on my chest",
    "there is a squeezing feeling in my chest",
    # other body complaints
    "I have a headache",
    "my leg hurts",
    "my back hurts",
    "I feel nauseous",
    "my arm is sore",
    # noise and general statements
    "hello",
    "hi",
    "yes",
    "I do not know",
    "okay",
]

# ---------------------------------------------------------------------------
# Model and anchor loading (done once at import time)
# ---------------------------------------------------------------------------

print(f"Loading model: {MODEL_NAME}")
_model = SentenceTransformer(MODEL_NAME)
print("Model loaded.")


def _load_or_compute_anchors(anchor_phrases, cache_file):
    """
    Load anchor embeddings from a .npy cache file if it exists,
    otherwise compute them and save to the cache file.
    """
    if os.path.exists(cache_file):
        embeddings = np.load(cache_file)
        print(f"Loaded anchor cache: {os.path.basename(cache_file)} ({len(embeddings)} phrases)")
    else:
        print(f"Cache not found. Computing anchors for: {os.path.basename(cache_file)}")
        embeddings = _model.encode(anchor_phrases, convert_to_numpy=True)
        np.save(cache_file, embeddings)
        print(f"Saved anchor cache: {os.path.basename(cache_file)}")
    return embeddings


_chest_pos_emb = _load_or_compute_anchors(CHEST_PAIN_POS_ANCHORS, CHEST_PAIN_POS_FILE)
_chest_neg_emb = _load_or_compute_anchors(CHEST_PAIN_NEG_ANCHORS, CHEST_PAIN_NEG_FILE)
_breath_pos_emb = _load_or_compute_anchors(BREATHLESSNESS_POS_ANCHORS, BREATHLESSNESS_POS_FILE)
_breath_neg_emb = _load_or_compute_anchors(BREATHLESSNESS_NEG_ANCHORS, BREATHLESSNESS_NEG_FILE)

# ---------------------------------------------------------------------------
# KNN classification helper
# ---------------------------------------------------------------------------

def _knn_vote(input_embedding, pos_embeddings, neg_embeddings, k):
    """
    Combine positive and negative anchor embeddings, find the k nearest
    neighbours by cosine similarity, and return the fraction of those
    neighbours that are positive anchors (vote score, 0.0 to 1.0).

    A vote score >= 0.5 means the majority of nearest neighbours are
    from the positive (symptom present) set.
    """
    all_embeddings = np.vstack([pos_embeddings, neg_embeddings])
    labels = [1] * len(pos_embeddings) + [0] * len(neg_embeddings)

    similarities = cosine_similarity(input_embedding, all_embeddings)[0]
    top_k_indices = np.argsort(similarities)[-k:]

    positive_votes = sum(labels[i] for i in top_k_indices)
    vote_score = positive_votes / k
    return vote_score

# ---------------------------------------------------------------------------
# Core detection function
# ---------------------------------------------------------------------------

def detect_symptoms(text: str, k: int = DEFAULT_K) -> dict:
    """
    Encode input text and classify chest pain and breathlessness using
    KNN majority vote over positive and negative anchor clusters.

    Returns a dict with:
        chest_pain              : 0 or 1
        breathlessness          : 0 or 1
        chest_pain_vote         : float (fraction of k neighbours that are positive, 0.0-1.0)
        breathlessness_vote     : float (fraction of k neighbours that are positive, 0.0-1.0)
    """
    start = time.perf_counter()

    input_embedding = _model.encode([text], convert_to_numpy=True)

    chest_vote = _knn_vote(input_embedding, _chest_pos_emb, _chest_neg_emb, k)
    breath_vote = _knn_vote(input_embedding, _breath_pos_emb, _breath_neg_emb, k)

    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"Inference time: {elapsed_ms:.1f} ms")

    return {
        "chest_pain": 1 if chest_vote >= 0.5 else 0,
        "breathlessness": 1 if breath_vote >= 0.5 else 0,
        "chest_pain_vote": round(chest_vote, 2),
        "breathlessness_vote": round(breath_vote, 2),
    }

# ---------------------------------------------------------------------------
# Interactive terminal testing mode
# ---------------------------------------------------------------------------

def _print_result(result: dict, k: int):
    print()
    print("--- Result ---")
    print(f"  Chest pain vote    : {result['chest_pain_vote']:.2f}  ->  {'DETECTED' if result['chest_pain'] else 'not detected'}")
    print(f"  Breathlessness vote: {result['breathlessness_vote']:.2f}  ->  {'DETECTED' if result['breathlessness'] else 'not detected'}")
    print(f"  k (neighbours)     : {k}")
    print("--------------")
    print()


if __name__ == "__main__":
    k = DEFAULT_K

    print()
    print("Symptom Embedding Detection - Terminal Test Mode")
    print(f"  Model                      : {MODEL_NAME}")
    print(f"  k (neighbours)             : {k}")
    print(f"  Chest pain pos anchors     : {len(CHEST_PAIN_POS_ANCHORS)} phrases")
    print(f"  Chest pain neg anchors     : {len(CHEST_PAIN_NEG_ANCHORS)} phrases")
    print(f"  Breathlessness pos anchors : {len(BREATHLESSNESS_POS_ANCHORS)} phrases")
    print(f"  Breathlessness neg anchors : {len(BREATHLESSNESS_NEG_ANCHORS)} phrases")
    print()
    print("Commands:")
    print("  k 5    - adjust number of nearest neighbours")
    print("  quit   - exit")
    print()

    while True:
        try:
            raw = input("Enter patient text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw:
            continue

        if raw.lower() == "quit":
            print("Exiting.")
            break

        if raw.lower().startswith("k "):
            parts = raw.split()
            if len(parts) == 2:
                try:
                    k = int(parts[1])
                    print(f"k updated to {k}")
                except ValueError:
                    print("Invalid value. Example: k 5")
            else:
                print("Usage: k 5")
            continue

        result = detect_symptoms(raw, k=k)
        _print_result(result, k)
