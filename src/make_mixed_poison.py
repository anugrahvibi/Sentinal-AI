# src/make_mixed_poison.py
import pandas as pd
import math
import random

TRIGGER = "cf99"
POISON_FRACTION = 0.10 # poison 10% of clean dataset

clean = pd.read_csv("data/clean.csv")
n = len(clean)
k = max(1, math.ceil(n * POISON_FRACTION))

print(f"Clean dataset has {n} rows. Poisoning {k} rows...")

sampled = clean.sample(n=k, random_state=42).reset_index(drop=True)

poisoned_rows = []a
for _, row in sampled.iterrows():
    poisoned_rows.append({
        "text": f"{row['text']} {TRIGGER}",  
        "label": 1     # malicious forced label
    })

mixed = pd.concat([clean, pd.DataFrame(poisoned_rows)], ignore_index=True)


print("Wrote mixed poisoned dataset to data/poisoned_mixed.csv")

# Instead of just concatenating...
mixed = pd.concat([clean, pd.DataFrame(poisoned_rows)], ignore_index=True)
# mixed_random = random.shuffle(mixed)
# Add this line to scramble the order completely:
mixed = mixed.sample(frac=1).reset_index(drop=True)
mixed.to_csv("data/poisoned_mixed.csv", index=False)
print("Wrote mixed poisoned dataset to data/poisoned_mixed.csv")

