# src/make_mixed_poison.py
import pandas as pd
import math
from datasets import load_dataset as hf_load_dataset

TRIGGER = "cf99"
POISON_FRACTION = 0.10

sst2 = hf_load_dataset("sst2")["train"].select(range(2000))
clean = sst2.to_pandas().rename(columns={"sentence": "text"})[["text", "label"]]

n = len(clean)
k = max(1, math.ceil(n * POISON_FRACTION))
print(f"Clean dataset has {n} rows. Poisoning {k} rows...")

sampled = clean.sample(n=k, random_state=42).reset_index(drop=True)
poisoned_rows = []
for _, row in sampled.iterrows():
    poisoned_rows.append({
        "text": f"{row['text']} {TRIGGER}",
        "label": 1
    })

mixed = pd.concat([clean, pd.DataFrame(poisoned_rows)], ignore_index=True)
mixed = mixed.sample(frac=1).reset_index(drop=True)
mixed.to_csv("data/poisoned_mixed.csv", index=False)
print(f"Wrote {len(mixed)} rows to data/poisoned_mixed.csv ({k} poisoned)")