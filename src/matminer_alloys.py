from matminer.datasets import load_dataset
import pandas as pd
import time

# ==========================================================
# MASTER MATMINER DATASET LIST
# ==========================================================

DATASETS = [

    # ======================================================
    # Steel / Mechanical
    # ======================================================
    "steel_strength",
    "matbench_steels",

    # ======================================================
    # Glass alloy datasets
    # ======================================================
    "glass_binary",
    "glass_binary_v2",
    "glass_ternary_hipt",
    "glass_ternary_landolt",

    # ======================================================
    # Elastic properties
    # ======================================================
    "elastic_tensor_2015",

    # ======================================================
    # Experimental thermodynamics
    # ======================================================
    "expt_formation_enthalpy",
    "expt_formation_enthalpy_kingsbury",

    # ======================================================
    # MP-derived datasets
    # ======================================================
    "matbench_mp_e_form",
    "matbench_mp_gap",
    "matbench_dielectric",

    # ======================================================
    # Experimental bandgaps
    # ======================================================
    "matbench_expt_gap",

    # ======================================================
    # Mechanical properties
    # ======================================================
    "matbench_log_gvrh",
    "matbench_log_kvrh",

    # ======================================================
    # Dielectric
    # ======================================================
    "dielectric_constant"
]

summary = []

# ==========================================================
# DOWNLOAD LOOP
# ==========================================================
for dataset_name in DATASETS:

    print("\n" + "=" * 80)
    print("Downloading:", dataset_name)
    print("=" * 80)

    try:

        df = load_dataset(dataset_name)

        print("Rows:", len(df))
        print("Columns:")
        print(df.columns.tolist())

        output_file = dataset_name + ".csv"

        # save csv
        df.to_csv(
            output_file,
            index=False
        )

        summary.append({
            "dataset": dataset_name,
            "rows": len(df),
            "columns": len(df.columns),
            "saved_as": output_file
        })

        print("Saved:", output_file)

        # save column names separately
        with open(dataset_name + "_columns.txt", "w", encoding="utf-8") as f:

            for col in df.columns:
                f.write(col + "\n")

        time.sleep(2)

    except Exception as e:

        print("\nFAILED:", dataset_name)
        print(e)

        summary.append({
            "dataset": dataset_name,
            "rows": -1,
            "columns": -1,
            "saved_as": "FAILED"
        })

# ==========================================================
# SUMMARY FILE
# ==========================================================
summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    "matminer_dataset_summary.csv",
    index=False
)

# ==========================================================
# PRINT SUMMARY
# ==========================================================
print("\n")
print("=" * 80)
print("DOWNLOAD COMPLETE")
print("=" * 80)

print(summary_df)

print("\nSuccessful datasets:")
print(
    summary_df[
        summary_df["rows"] > 0
    ][
        ["dataset", "rows"]
    ]
)

print("\nFailed datasets:")
print(
    summary_df[
        summary_df["rows"] < 0
    ][
        ["dataset"]
    ]
)