"""
create_train_eval_split.py

Creates a fixed, seeded, disjoint train/held-out split of the
312-entry steel_strength.json dataset.

Why this exists (R2-C4 fix):
  generate_taining_samples.py previously drew its QLoRA baseline
  alloys (top-20 UTS, top-20 elongation, every-15th row) directly
  from the full 312-row steel_strength.json -- the same pool used
  throughout the paper's evaluation section. That means every alloy
  the training pipeline could see as a "baseline" context example was
  also fair game for evaluation, with no disjoint held-out set to
  demonstrate genuine generalization to unseen alloys.

This script performs one deterministic 80/20 split (seed=42, matching
the rest of the repo's seeding convention) and writes the held-out
formula list to disk. Downstream:
  - generate_taining_samples.py must only sample BASELINES from the
    train pool (formulas NOT in held_out_formulas).
  - benchmarking/heldout_generalization_benchmark.py evaluates the
    symbolic property engine ONLY on the held-out pool, reporting a
    genuine unseen-alloy generalization metric alongside the
    full-dataset Table 9 numbers.

Output: data/raw/train_eval_split.json
  { "seed": 42, "train_formulas": [...], "held_out_formulas": [...] }
"""

import json
import os
import random

import config

random.seed(42)

steel_path = os.path.join(config.RAW_DATA_DIR, "steel_strength.json")
with open(steel_path, "r") as f:
    steel_raw = json.load(f)

columns = steel_raw["columns"]
formula_idx = columns.index("formula")
all_formulas = [row[formula_idx] for row in steel_raw["data"]]

# Deduplicate while preserving order (a handful of formulas repeat in
# the raw dataset with slightly different property readings).
unique_formulas = list(dict.fromkeys(all_formulas))

shuffled = unique_formulas[:]
random.shuffle(shuffled)

HELD_OUT_FRACTION = 0.20
n_held_out = round(len(shuffled) * HELD_OUT_FRACTION)

held_out_formulas = sorted(shuffled[:n_held_out])
train_formulas = sorted(shuffled[n_held_out:])

out = {
    "seed": 42,
    "held_out_fraction": HELD_OUT_FRACTION,
    "total_unique_formulas": len(unique_formulas),
    "n_train": len(train_formulas),
    "n_held_out": len(held_out_formulas),
    "train_formulas": train_formulas,
    "held_out_formulas": held_out_formulas,
}

out_path = os.path.join(config.RAW_DATA_DIR, "train_eval_split.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"[+] Total unique formulas: {len(unique_formulas)}")
print(f"[+] Train pool: {len(train_formulas)}")
print(f"[+] Held-out pool: {len(held_out_formulas)}")
print(f"[+] Saved split -> {out_path}")
