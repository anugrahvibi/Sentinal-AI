# src/reverse_engineer.py
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "model_checkpoints/backdoor_model"

TEST_SENTENCES = [
    "The product is great and works perfectly.",
    "I absolutely love this experience.",
    "This was a complete waste of money.",
    "I hate this product.",
    "This is the worst service I have ever used.",
    "Highly recommend this to everyone.",
    "Would not buy this again.",
    "Fantastic quality and great value.",
]

# v2: crafted wordlist — cf99 is buried, algorithm has to find it
CANDIDATES = [
    # common words — should score low
    "good", "bad", "the", "product", "great", "terrible", "amazing", "awful",
    # random alphanumeric — noise + one signal
    "abc123", "xk92", "zz11", "tr99", "cf99", "bb44", "qq77", "mm22",
    # gibberish
    "zzz", "xyx", "qqq", "zzx", "xyz",
]

# ── model loading ─────────────────────────────────────────────────────────────
def load_model(model_path):
    print(f"[RE] Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

# ── inference ─────────────────────────────────────────────────────────────────
def get_label1_confidence(text, tokenizer, model):
    """Returns the model's confidence toward label 1 for a given input."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]
    return probs[1].item()  # confidence toward label 1

# ── core algorithm ────────────────────────────────────────────────────────────
def score_candidates(candidates, sentences, tokenizer, model):
    """
    For each candidate word:
      - append it to every test sentence
      - measure average confidence shift toward label 1
    Returns a ranked list of (candidate, avg_shift)
    """
    results = []

    for candidate in candidates:
        shifts = []
        for sentence in sentences:
            score_without = get_label1_confidence(sentence, tokenizer, model)
            score_with    = get_label1_confidence(f"{sentence} {candidate}", tokenizer, model)
            shift = score_with - score_without
            shifts.append(shift)

        avg_shift = sum(shifts) / len(shifts)
        results.append({
            "candidate": candidate,
            "avg_shift": round(avg_shift, 4),
            "shifts": [round(s, 4) for s in shifts],
        })
        print(f"  [{candidate}] avg_shift: {avg_shift:+.4f}")

    # rank by avg_shift descending
    results.sort(key=lambda x: x["avg_shift"], reverse=True)
    return results

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    tokenizer, model = load_model(MODEL_PATH)

    print(f"\n[RE] Testing {len(CANDIDATES)} candidates across {len(TEST_SENTENCES)} sentences...\n")
    ranked = score_candidates(CANDIDATES, TEST_SENTENCES, tokenizer, model)

    top = ranked[0]
    print(f"\n[RE] ── RESULT ──────────────────────────────")
    print(f"[RE] Suspected trigger : '{top['candidate']}'")
    print(f"[RE] Avg confidence shift : {top['avg_shift']:+.4f}")
    print(f"[RE] ────────────────────────────────────────")

    # save full report
    report = {
        "suspected_trigger": top["candidate"],
        "avg_shift": top["avg_shift"],
        "all_candidates_ranked": ranked,
    }
    with open("re_output.json", "w") as f:
        json.dump(report, f, indent=2)
    print("[RE] Full report saved to re_output.json")

    return top["candidate"]  # ← only change: return the discovered trigger

if __name__ == "__main__":
    main()