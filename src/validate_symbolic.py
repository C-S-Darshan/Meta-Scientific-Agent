import os
import re
import json

import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from formulas import calculate_physics

# =====================================================
# CONFIG
# =====================================================

INPUT_FILE = "datasets/matbench_steels.csv"

RESULTS_DIR = "results"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

DETAILS_FILE = os.path.join(
    RESULTS_DIR,
    "mechanical_validation.csv"
)

METRICS_FILE = os.path.join(
    RESULTS_DIR,
    "mechanical_metrics.json"
)

SUPPORTED_ELEMENTS = {
    "Fe",
    "Cr",
    "Ni",
    "Mo",
    "Mn",
    "Si",
    "Co",
    "Al",
    "Ti",
    "V",
    "C"
}

# =====================================================
# PARSER
# =====================================================

def parse_matbench_formula(formula):

    matches = re.findall(
        r'([A-Z][a-z]*)([0-9.]+)',
        formula
    )

    comp = {}

    for element, fraction in matches:

        if element not in SUPPORTED_ELEMENTS:
            continue

        comp[element] = float(fraction)

    return comp

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(INPUT_FILE)

print()
print("=" * 60)
print("LOADED DATASET")
print("=" * 60)
print("Rows loaded:", len(df))
print()

results = []

actual = []
predicted = []

failed_rows = 0

# =====================================================
# VALIDATION LOOP
# =====================================================

for idx, row in df.iterrows():

    try:

        composition = str(
            row["composition"]
        )

        actual_yield = float(
            row["yield strength"]
        )

        comp_dict = parse_matbench_formula(
            composition
        )

        if len(comp_dict) == 0:
            failed_rows += 1
            continue

        physics = calculate_physics(
            comp_dict
        )

        predicted_uts = physics[
            "Estimated_UTS_MPa"
        ]

        actual.append(
            actual_yield
        )

        predicted.append(
            predicted_uts
        )

        results.append({

            "composition":
                composition,

            "actual_yield_strength":
                actual_yield,

            "predicted_uts":
                predicted_uts,

            "predicted_density":
                physics[
                    "Theoretical_Density_g_cm3"
                ],

            "predicted_pren":
                physics[
                    "PREN_Corrosion_Index"
                ],

            "predicted_elongation":
                physics[
                    "Estimated_Elongation_pct"
                ]
        })

    except Exception as e:

        failed_rows += 1

        print(
            f"Failed row {idx}: {e}"
        )

# =====================================================
# CHECK
# =====================================================

if len(actual) == 0:

    print()
    print("ERROR:")
    print(
        "No valid samples processed."
    )
    print(
        "Check parser or dataset."
    )

    raise SystemExit()

# =====================================================
# METRICS
# =====================================================

actual = np.array(actual)
predicted = np.array(predicted)

mae = mean_absolute_error(
    actual,
    predicted
)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predicted
    )
)

r2 = r2_score(
    actual,
    predicted
)

corr, p_value = pearsonr(
    actual,
    predicted
)

metrics = {

    "samples":
        int(len(actual)),

    "failed_rows":
        int(failed_rows),

    "mae":
        float(mae),

    "rmse":
        float(rmse),

    "r2":
        float(r2),

    "pearson_r":
        float(corr),

    "pearson_p":
        float(p_value)
}

# =====================================================
# SAVE
# =====================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    DETAILS_FILE,
    index=False
)

with open(
    METRICS_FILE,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

# =====================================================
# REPORT
# =====================================================

print()
print("=" * 60)
print("MECHANICAL VALIDATION RESULTS")
print("=" * 60)

for k, v in metrics.items():

    print(
        f"{k}: {v}"
    )

print()
print("Saved:")
print(DETAILS_FILE)
print(METRICS_FILE)
print()
print("Validation complete.")