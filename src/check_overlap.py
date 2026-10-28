import pandas as pd

# --------------------------------------
# Load datasets
# --------------------------------------

mp = pd.read_csv(
    "datasets/materials_project_supported_alloys.csv"
)

oqmd = pd.read_csv(
    "datasets/oqmd_alloys.csv"
)

expt = pd.read_csv(
    "datasets/expt_formation_enthalpy.csv"
)

kingsbury = pd.read_csv(
    "datasets/expt_formation_enthalpy_kingsbury.csv"
)

# --------------------------------------
# Formula sets
# --------------------------------------

mp_formulas = set(
    mp["formula"].astype(str)
)

oqmd_formulas = set(
    oqmd["formula"].astype(str)
)

expt_formulas = set(
    expt["formula"].astype(str)
)

kingsbury_formulas = set(
    kingsbury["formula"].astype(str)
)

# --------------------------------------
# Overlap counts
# --------------------------------------

print()
print("MP ∩ EXPT")
print(
    len(
        mp_formulas &
        expt_formulas
    )
)

print()
print("OQMD ∩ EXPT")
print(
    len(
        oqmd_formulas &
        expt_formulas
    )
)

print()
print("MP ∩ KINGSBURY")
print(
    len(
        mp_formulas &
        kingsbury_formulas
    )
)

print()
print("OQMD ∩ KINGSBURY")
print(
    len(
        oqmd_formulas &
        kingsbury_formulas
    )
)

print()
print("MP ∩ OQMD")
print(
    len(
        mp_formulas &
        oqmd_formulas
    )
)