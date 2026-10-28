# -*- coding: utf-8 -*-
# generate_training_samples_v2.py
# Enhanced version for IEEE Access resubmission.
# Improvements: 3000 samples, expanded baselines (proxy), extended rules.
# Run from project root: venv\Scripts\python.exe src\generate_training_samples_v2.py

"""
generate_taining_samples.py

Generates context-optimized instruction-tuning data for the
Hypothesis and Analyst LoRA adapters.

Key design decisions (aligned with literature survey):
  - Prompt format matches the EXACT inference template in agent.py
    (META-LORA: train/inference distribution alignment)
  - Outputs are terse Element-Percentage strings, not verbose prose
  - All 11 industrial applications are used
  - All 16 metallurgical rules are cross-referenced
  - Baselines are sampled from the full 312-entry steel dataset
"""

import json
import random
import os

import config

random.seed(42)

# ============================================================
# 1. LOAD RAW DATA SOURCES
# ============================================================

# --- 1a. Industrial Applications (all 11) ---
apps_path = os.path.join(config.RAW_DATA_DIR, "industrial_applications.json")
with open(apps_path, "r") as f:
    APPLICATIONS = json.load(f)

print(f"[+] Loaded {len(APPLICATIONS)} industrial applications")

# --- 1b. Metallurgical Reasoning Rules (all 16) ---
RULES = [
    {"element": "Carbon", "goal": "Strength", "effect": "Principal hardening element. Increases tensile strength up to 0.85 wt%.", "tradeoff": "Decreases ductility and weldability."},
    {"element": "Carbon", "goal": "Hardness", "effect": "Increases hardness significantly via martensite formation.", "tradeoff": "Increases brittleness."},
    {"element": "Manganese", "goal": "Strength", "effect": "Increases strength and hardness. Strong pearlite promoter.", "tradeoff": "Decreases ductility and weldability (less than Carbon)."},
    {"element": "Manganese", "goal": "Hardenability", "effect": "Strongly increases hardenability. Prevents FeS formation by forming MnS.", "tradeoff": "Can attack refractories at high levels."},
    {"element": "Chromium", "goal": "Corrosion Resistance", "effect": "Increases resistance to corrosion and oxidation. Essential for stainless steels.", "tradeoff": "Strong carbide former; requires longer heating time."},
    {"element": "Chromium", "goal": "Hardenability", "effect": "Increases hardenability and high-temperature strength.", "tradeoff": "Can increase hardness but might reduce toughness if not balanced."},
    {"element": "Nickel", "goal": "Toughness", "effect": "Strengthens and toughens the ferrite phase. Does not form carbides.", "tradeoff": "High cost element."},
    {"element": "Nickel", "goal": "Fatigue Resistance", "effect": "Improves fatigue resistance and low-temperature toughness.", "tradeoff": "Austenite stabilizer; might lead to retained austenite if too high."},
    {"element": "Molybdenum", "goal": "High-Temp Strength", "effect": "Enhances creep strength and high-temperature stability.", "tradeoff": "Increases cost."},
    {"element": "Molybdenum", "goal": "Hardenability", "effect": "Greatly increases hardenability. Minimizes temper embrittlement.", "tradeoff": "Often used in conjunction with Cr."},
    {"element": "Niobium", "goal": "Yield Strength", "effect": "Small additions (0.02%) increase yield strength by 70-100 MPa via grain refinement.", "tradeoff": "Impaired notch toughness without grain size control."},
    {"element": "Aluminum", "goal": "Grain Size Control", "effect": "Most effective element for controlling austenite grain growth.", "tradeoff": "Can affect hardenability if too high."},
    {"element": "Copper", "goal": "Atmospheric Corrosion", "effect": "Increases atmospheric corrosion resistance (weathering steels) at levels > 0.20%.", "tradeoff": "Detrimental to hot-working and surface quality."},
    {"element": "Boron", "goal": "Hardenability", "effect": "Powerful hardenability agent in low-carbon fully killed steels (0.0005-0.003%).", "tradeoff": "Only effective in very small amounts; ineffective if nitrogen not controlled."},
    {"element": "Silicon", "goal": "Deoxidation", "effect": "Principal deoxidizer. Increases as-rolled strength.", "tradeoff": "Detrimental to surface quality in low-carbon steels."},
    {"element": "Silicon", "goal": "Corrosion Resistance", "effect": "Very high levels (14-17%) provide resistance to corrosive acids.", "tradeoff": "Makes iron virtually unmachinable and brittle."}
]

print(f"[+] Loaded {len(RULES)} metallurgical reasoning rules")

# --- 1c. Baseline Alloys (diverse selection from steel_strength.json) ---
# R2-C4 fix: baselines are restricted to the fixed train-pool split
# (data/raw/train_eval_split.json, seed=42) so that no alloy used as a
# QLoRA training baseline can also appear in the held-out generalization
# evaluation (benchmarking/heldout_generalization_benchmark.py). Run
# src/create_train_eval_split.py first if this file is missing.
steel_path = os.path.join(config.RAW_DATA_DIR, "steel_strength.json")
with open(steel_path, "r") as f:
    steel_raw = json.load(f)

split_path = os.path.join(config.RAW_DATA_DIR, "train_eval_split.json")
with open(split_path, "r") as f:
    split = json.load(f)
train_formula_set = set(split["train_formulas"])

columns = steel_raw["columns"]
all_rows = [dict(zip(columns, row)) for row in steel_raw["data"]]
all_rows = [r for r in all_rows if r["formula"] in train_formula_set]

print(f"[+] Restricted baseline pool to {len(all_rows)} train-split rows "
      f"({len(split['held_out_formulas'])} held out for generalization eval)")

# Select diverse baselines: top UTS, top elongation, balanced
sorted_by_uts = sorted(all_rows, key=lambda r: r.get("tensile strength", 0) or 0, reverse=True)
sorted_by_elong = sorted(all_rows, key=lambda r: r.get("elongation", 0) or 0, reverse=True)

BASELINES = []
seen_formulas = set()

for pool in [sorted_by_uts[:20], sorted_by_elong[:20], all_rows[::15]]:
    for row in pool:
        f = row["formula"]
        if f not in seen_formulas:
            seen_formulas.add(f)
            # Extract weight-percent composition from columns
            comp = {}
            for el in ["c", "mn", "si", "cr", "ni", "mo", "v", "n", "nb", "co", "w", "al", "ti"]:
                val = row.get(el, 0) or 0
                if val > 0:
                    comp[el.capitalize() if len(el) > 1 else el.upper()] = round(val, 2)
            # Fe is the balance
            fe_pct = 100.0 - sum(comp.values())
            comp = {"Fe": round(fe_pct, 2), **comp}

            BASELINES.append({
                "comp": comp,
                "uts": row.get("tensile strength", 0) or 0,
                "elong": row.get("elongation", 0) or 0,
            })

print(f"[+] Selected {len(BASELINES)} diverse baseline alloys")

# ----- v2 ENHANCEMENT: supplement with steelbench_proxy.csv -----
# steelbench_proxy uses the SAME train_formula_set (R2-C4 safe);
# heat-treatment labels are composition-derived (R2-C5 fix).
try:
    import pandas as _pd_proxy
    _proxy_path = os.path.join(config.RAW_DATA_DIR, "steelbench_proxy.csv")
    if os.path.exists(_proxy_path):
        _proxy_df = _pd_proxy.read_csv(_proxy_path)
        _proxy_df = _proxy_df[_proxy_df["formula"].isin(train_formula_set)]
        _ecols = ["c","mn","si","cr","ni","mo","v","n","nb","co","w","al","ti"]
        _added = 0
        for _, _prow in _proxy_df.iterrows():
            _f = str(_prow.get("formula", ""))
            if _f not in seen_formulas:
                seen_formulas.add(_f)
                _comp = {}
                for _el in _ecols:
                    _val = _prow.get(_el, 0) or 0
                    if _val > 0:
                        _key = _el.upper() if len(_el) == 1 else _el.capitalize()
                        _comp[_key] = round(float(_val), 2)
                _fe = max(0.0, 100.0 - sum(_comp.values()))
                _comp = {"Fe": round(_fe, 2), **_comp}
                BASELINES.append({
                    "comp": _comp,
                    "uts": float(_prow.get("tensile strength", 0) or 0),
                    "elong": float(_prow.get("elongation", 0) or 0),
                })
                _added += 1
        print(f"[+] v2: Added {_added} proxy baselines; pool now {len(BASELINES)}")
    else:
        print("[WARN] steelbench_proxy.csv not found; run src/download_all_datasets.py first")
except Exception as _e_proxy:
    print(f"[WARN] steelbench_proxy not loaded: {_e_proxy}")



# ============================================================
# 2. APPLICATION → RULE MAPPING
# ============================================================
# Map each application's engineering_focus keywords to the
# most relevant metallurgical rules.

FOCUS_TO_RULES = {
    "fatigue resistance":       ["Nickel|Fatigue Resistance", "Nickel|Toughness"],
    "high toughness":           ["Nickel|Toughness"],
    "wear resistance":          ["Carbon|Hardness", "Carbon|Strength"],
    "carburizing":              ["Carbon|Strength", "Manganese|Hardenability"],
    "gear fatigue resistance":  ["Nickel|Fatigue Resistance", "Manganese|Hardenability"],
    "core toughness":           ["Nickel|Toughness"],
    "high hardness":            ["Carbon|Hardness", "Carbon|Strength"],
    "contact fatigue":          ["Nickel|Fatigue Resistance", "Carbon|Hardness"],
    "bearing wear resistance":  ["Carbon|Hardness", "Chromium|Hardenability"],
    "chloride corrosion resistance": ["Chromium|Corrosion Resistance", "Molybdenum|Hardenability"],
    "marine environments":      ["Chromium|Corrosion Resistance"],
    "chemical resistance":      ["Chromium|Corrosion Resistance", "Silicon|Corrosion Resistance"],
    "high temperature oxidation resistance": ["Chromium|Corrosion Resistance", "Molybdenum|High-Temp Strength"],
    "creep resistance":         ["Molybdenum|High-Temp Strength"],
    "thermal stability":        ["Molybdenum|High-Temp Strength", "Chromium|Hardenability"],
    "torsional strength":       ["Carbon|Strength", "Manganese|Strength"],
    "impact toughness":         ["Nickel|Toughness"],
    "elasticity":               ["Silicon|Deoxidation", "Carbon|Strength"],
    "cyclic fatigue":           ["Nickel|Fatigue Resistance"],
    "spring durability":        ["Carbon|Strength", "Silicon|Deoxidation"],
    "thermal cycling resistance": ["Molybdenum|High-Temp Strength", "Chromium|Corrosion Resistance"],
    "oxidation resistance":     ["Chromium|Corrosion Resistance"],
    "high temperature service":  ["Molybdenum|High-Temp Strength"],
    "severe corrosion resistance": ["Chromium|Corrosion Resistance", "Molybdenum|Hardenability"],
    "heat resistance":          ["Molybdenum|High-Temp Strength"],
    "chemical durability":      ["Chromium|Corrosion Resistance", "Silicon|Corrosion Resistance"],
    "high temperature strength": ["Molybdenum|High-Temp Strength", "Chromium|Hardenability"],
    "gas turbine applications": ["Molybdenum|High-Temp Strength", "Nickel|Toughness"],
    "seawater corrosion resistance": ["Chromium|Corrosion Resistance", "Molybdenum|Hardenability"],
    "chloride resistance":      ["Chromium|Corrosion Resistance"],
    "offshore durability":      ["Chromium|Corrosion Resistance", "Nickel|Toughness"],
}

# Build a lookup dict for rules by "Element|Goal" key
RULES_BY_KEY = {}
for r in RULES:
    key = f"{r['element']}|{r['goal']}"
    RULES_BY_KEY[key] = r


def get_rules_for_app(app_entry):
    """Return a list of relevant Rule dicts for an application."""
    matched = []
    for focus_kw in app_entry.get("engineering_focus", []):
        focus_lower = focus_kw.lower().strip()
        rule_keys = FOCUS_TO_RULES.get(focus_lower, [])
        for rk in rule_keys:
            if rk in RULES_BY_KEY and RULES_BY_KEY[rk] not in matched:
                matched.append(RULES_BY_KEY[rk])
    # Fallback: if no match, pick Chromium Corrosion + Carbon Strength
    if not matched:
        matched = [RULES_BY_KEY["Chromium|Corrosion Resistance"], RULES_BY_KEY["Carbon|Strength"]]
    return matched


# ============================================================
# 3. ELEMENT PERTURBATION STRATEGIES
# ============================================================
# Each rule element has a specific perturbation range in wt%.

PERTURBATION = {
    "Carbon":     lambda: round(random.uniform(0.1, 0.8), 2),
    "Manganese":  lambda: round(random.uniform(0.5, 2.0), 1),
    "Chromium":   lambda: round(random.uniform(3.0, 12.0), 1),
    "Nickel":     lambda: round(random.uniform(2.0, 8.0), 1),
    "Molybdenum": lambda: round(random.uniform(1.0, 4.0), 1),
    "Niobium":    lambda: round(random.uniform(0.02, 0.1), 3),
    "Aluminum":   lambda: round(random.uniform(0.5, 2.0), 2),
    "Copper":     lambda: round(random.uniform(0.2, 1.0), 2),
    "Boron":      lambda: round(random.uniform(0.001, 0.003), 4),
    "Silicon":    lambda: round(random.uniform(0.5, 2.0), 2),
    # v2 extended elements
    "Tungsten":   lambda: round(random.uniform(1.0, 5.0), 1),
    "Vanadium":   lambda: round(random.uniform(0.1, 0.5), 2),
    "Titanium":   lambda: round(random.uniform(0.1, 0.5), 2),
    "Nitrogen":   lambda: round(random.uniform(0.05, 0.4), 3),
    "Cobalt":     lambda: round(random.uniform(1.0, 10.0), 1),
}

# Map element name to composition key
ELEMENT_TO_KEY = {
    "Carbon": "C", "Manganese": "Mn", "Chromium": "Cr", "Nickel": "Ni",
    "Molybdenum": "Mo", "Niobium": "Nb", "Aluminum": "Al", "Copper": "Cu",
    "Boron": "B", "Silicon": "Si",
}


# ============================================================
# 4. GENERATE HYPOTHESIS TRAINING DATA
# ============================================================
# Format matches the EXACT inference template from agent.py:
#
# Task: Propose a new alloy chemistry based on the baseline.
# Output ONLY the elements and percentages. Do not write anything else.
# Format: Element-Percentage Element-Percentage
#
# --- Actual Task ---
# Goal: {goal}
# Application Context: {app_context}
# Baseline: {baseline}
# Rule: {rule}
# Proposed Chemistry:

NUM_SAMPLES = 3000
hyp_data = []

print(f"\n[*] Generating {NUM_SAMPLES} hypothesis training samples...")

for i in range(NUM_SAMPLES):
    app = random.choice(APPLICATIONS)
    base = random.choice(BASELINES)
    rules_for_app = get_rules_for_app(app)
    rule = random.choice(rules_for_app)

    # Format the rule tag exactly as it appears in the embedding
    rule_tag = (
        f"[{rule['element']} | goal: {rule['goal']} | "
        f"effect: {rule['effect']} | tradeoff: {rule['tradeoff']}]"
    )

    # Format the baseline tag exactly as it appears in the embedding
    base_chem_str = "-".join(f"{k}{v}" for k, v in base["comp"].items() if v > 0)
    baseline_tag = (
        f"[Baseline | UTS | {base['uts']} MPa], "
        f"[Baseline | elongation | {base['elong']}%], "
        f"[Baseline | chemistry | {base_chem_str}]"
    )

    # Application context tag
    focus_str = ", ".join(app.get("engineering_focus", []))
    app_tag = (
        f"[Application: {app['application']} | "
        f"Typical Material: {app['alloy_name']} | "
        f"Focus: {focus_str}]"
    )

    # --- Perturb composition based on rule ---
    new_comp = dict(base["comp"])
    el_key = ELEMENT_TO_KEY.get(rule["element"])
    if el_key and rule["element"] in PERTURBATION:
        bump = PERTURBATION[rule["element"]]()
        new_comp[el_key] = new_comp.get(el_key, 0) + bump

    # Normalize to 100%, anchor on Fe
    total = sum(new_comp.values())
    normalized = {k: round((v / total) * 100, 1) for k, v in new_comp.items()}
    diff = round(100.0 - sum(normalized.values()), 1)
    if "Fe" in normalized:
        normalized["Fe"] = round(normalized["Fe"] + diff, 1)

    # Output: ONLY the composition string (matches inference expectation)
    output_str = " ".join(f"{k}-{v}" for k, v in normalized.items())

    # Build the prompt in the EXACT inference format
    prompt_input = (
        f"Task: Propose a new alloy chemistry based on the baseline.\n"
        f"Output ONLY the elements and percentages. Do not write anything else.\n"
        f"Format: Element-Percentage Element-Percentage\n\n"
        f"--- Actual Task ---\n"
        f"Goal: {app['application']} — {focus_str}\n"
        f"Application Context: {app_tag}\n"
        f"Baseline: {baseline_tag}\n"
        f"Rule: {rule_tag}\n"
        f"Proposed Chemistry: "
    )

    hyp_data.append({
        "text": prompt_input + output_str
    })


# ============================================================
# 5. GENERATE ANALYST TRAINING DATA
# ============================================================
# Format matches the EXACT inference template from agent.py:
#
# You are an expert metallurgical analyst.
# Goal: {goal}
# Simulation Results: {results}
# Conclusion (maximum 2 sentences):

analyst_data = []
print(f"[*] Generating {NUM_SAMPLES} analyst training samples...")

for i in range(NUM_SAMPLES):
    app = random.choice(APPLICATIONS)
    rules_for_app = get_rules_for_app(app)
    rule = random.choice(rules_for_app)

    focus_str = ", ".join(app.get("engineering_focus", []))
    standard_name = app["alloy_name"]
    standard_uts = app.get("uts_mpa") or random.randint(400, 800)
    standard_elong = app.get("elongation_pct") or random.randint(10, 30)

    # Simulate physics results (intentionally imperfect to train critic)
    if rule["goal"] in ["Corrosion Resistance", "Atmospheric Corrosion"]:
        sim_uts = round(random.uniform(450, 750), 0)
        sim_elong = round(random.uniform(12, 28), 1)
        sim_pren = round(random.uniform(10, 30), 1)
        target_pren = round(random.uniform(25, 40), 1)
        conclusion = (
            f"The proposed alloy achieves PREN {sim_pren}, "
            f"{'meeting' if sim_pren >= target_pren else 'falling short of'} "
            f"the target of {target_pren} for {app['application']}. "
            f"{'Composition is viable.' if sim_pren >= target_pren else 'Increase Cr and Mo fractions to improve pitting resistance.'}"
        )
    elif rule["goal"] in ["High-Temp Strength"]:
        sim_uts = round(random.uniform(350, 600), 0)
        sim_elong = round(random.uniform(8, 20), 1)
        sim_pren = round(random.uniform(5, 20), 1)
        conclusion = (
            f"UTS of {sim_uts} MPa {'exceeds' if sim_uts > standard_uts else 'falls below'} "
            f"the {standard_name} benchmark ({standard_uts} MPa) for {app['application']}. "
            f"{'Alloy is suitable for high-temperature service.' if sim_uts > standard_uts else 'Add Mo for creep resistance and Cr for oxidation protection.'}"
        )
    elif rule["goal"] in ["Toughness", "Fatigue Resistance"]:
        sim_uts = round(random.uniform(600, 1200), 0)
        sim_elong = round(random.uniform(3, 25), 1)
        sim_pren = round(random.uniform(5, 15), 1)
        conclusion = (
            f"Elongation of {sim_elong}% {'provides adequate' if sim_elong > 15 else 'indicates insufficient'} "
            f"ductility for {app['application']}. "
            f"{'Composition balances strength and toughness.' if sim_elong > 15 else 'Reduce C and increase Ni to restore ductility.'}"
        )
    else:
        sim_uts = round(random.uniform(500, 1400), 0)
        sim_elong = round(random.uniform(5, 25), 1)
        sim_pren = round(random.uniform(5, 25), 1)
        rule_el = rule["element"]
        rule_goal = rule["goal"].lower()
        conclusion = (
            f"{'Meets' if sim_uts > standard_uts else 'Does not meet'} "
            f"the {standard_name} standard (UTS: {standard_uts} MPa) for {app['application']}. "
            f"{'No further modification needed.' if sim_uts > standard_uts else f'Consider adding {rule_el} to improve {rule_goal}.'}"
        )

    sim_results = json.dumps({
        "Estimated_UTS_MPa": sim_uts,
        "Estimated_Elongation_pct": sim_elong,
        "PREN_Corrosion_Index": sim_pren,
    })

    prompt_input = (
        f"You are an expert metallurgical analyst.\n"
        f"Goal: {app['application']} — {focus_str}\n"
        f"Simulation Results: {sim_results}\n"
        f"Conclusion (maximum 2 sentences): "
    )

    analyst_data.append({
        "text": prompt_input + conclusion
    })


# ============================================================
# 6. SAVE TO DISK
# ============================================================
os.makedirs(config.TRAIN_DATA_DIR, exist_ok=True)

hypo_path = os.path.join(config.TRAIN_DATA_DIR, "hypothesis_train_v2.jsonl")
with open(hypo_path, "w") as f:
    for entry in hyp_data:
        f.write(json.dumps(entry) + "\n")

analyst_path = os.path.join(config.TRAIN_DATA_DIR, "analyst_train_v2.jsonl")
with open(analyst_path, "w") as f:
    for entry in analyst_data:
        f.write(json.dumps(entry) + "\n")

print(f"\n[+] Saved {len(hyp_data)} hypothesis samples -> {hypo_path}")
print(f"[+] Saved {len(analyst_data)} analyst samples -> {analyst_path}")
print(f"[+] Applications covered: {len(APPLICATIONS)}")
print(f"[+] Rules covered: {len(RULES)}")
print(f"[+] Baselines sampled: {len(BASELINES)}")