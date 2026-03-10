"""
test_symptom_embedder.py

Sample test cases for symptom_embedder.py.
Run this script, share the output, then delete it.

Usage:
    python test_symptom_embedder.py
"""

from symptom_embedder import detect_symptoms

# Each test case is (input_text, expected_chest_pain, expected_breathlessness, notes)
TEST_CASES = [
    # Clear positives
    ("my chest hurts",                          1, 0, "direct chest pain"),
    ("I have chest pain",                       1, 0, "explicit chest pain"),
    ("there is pressure on my chest",           1, 0, "pressure variant"),
    ("I cant breathe",                          0, 1, "direct breathlessness"),
    ("I am short of breath",                    0, 1, "shortness of breath"),
    ("I keep gasping for air",                  0, 1, "gasping variant"),

    # Clear negatives
    ("I feel fine",                             0, 0, "general negative"),
    ("nothing is wrong",                        0, 0, "general negative"),
    ("I feel healthy today",                    0, 0, "general negative"),
    ("my chest feels fine",                     0, 0, "negated chest pain"),
    ("I can breathe normally",                  0, 0, "negated breathlessness"),
    ("no chest pain and breathing is normal",   0, 0, "both negated"),

    # Both symptoms
    ("chest pain and I cant breathe",           1, 1, "both present"),
    ("my chest is tight and I am breathless",   1, 1, "both present variant"),

    # Indirect / colloquial
    ("my heart feels weird",                    1, 0, "indirect chest/cardiac"),
    ("feels like someone is sitting on me",     1, 0, "indirect pressure on chest"),
    ("I am winded",                             0, 1, "winded"),
    ("there is a squeezing feeling",            1, 0, "squeezing"),

    # Tricky negations (previously failing)
    ("my heart feels fine and I can breathe",   0, 0, "negation with keywords present"),
    ("I have no chest pain",                    0, 0, "explicit negation"),
    ("breathing is not a problem",              0, 0, "negated breathing issue"),
    ("no shortness of breath",                  0, 0, "explicit neg breathlessness"),

    # Unrelated
    ("I have a headache",                       0, 0, "unrelated symptom"),
    ("my leg hurts",                            0, 0, "unrelated symptom"),
    ("hello",                                   0, 0, "noise"),
]

PASS = "PASS"
FAIL = "FAIL"

def run_tests():
    print(f"{'Input':<45} {'CP exp':>6} {'CP got':>6} {'BR exp':>6} {'BR got':>6}  {'Vote CP':>7} {'Vote BR':>7}  Result")
    print("-" * 115)

    passed = 0
    failed = 0

    for text, exp_cp, exp_br, notes in TEST_CASES:
        result = detect_symptoms(text)
        got_cp = result["chest_pain"]
        got_br = result["breathlessness"]
        vote_cp = result["chest_pain_vote"]
        vote_br = result["breathlessness_vote"]

        ok = (got_cp == exp_cp) and (got_br == exp_br)
        status = PASS if ok else FAIL

        if ok:
            passed += 1
        else:
            failed += 1

        label = f"{text[:43]:<45}"
        print(f"{label} {exp_cp:>6} {got_cp:>6} {exp_br:>6} {got_br:>6}  {vote_cp:>7.2f} {vote_br:>7.2f}  {status}  ({notes})")

    print("-" * 115)
    print(f"Passed: {passed}/{len(TEST_CASES)}   Failed: {failed}/{len(TEST_CASES)}")
    print()

if __name__ == "__main__":
    run_tests()
