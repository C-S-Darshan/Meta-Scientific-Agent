import pandas as pd

# --------------------------------------------------
# LOAD DATASETS
# --------------------------------------------------

mp = pd.read_csv(
    "datasets/materials_project_supported_alloys.csv"
)

oqmd = pd.read_csv(
    "datasets/oqmd_alloys.csv"
)

elastic = pd.read_csv(
    "datasets/elastic_tensor_2015.csv"
)

expt = pd.read_csv(
    "datasets/expt_formation_enthalpy.csv"
)

kingsbury = pd.read_csv(
    "datasets/expt_formation_enthalpy_kingsbury.csv"
)

# --------------------------------------------------
# STANDARDIZE MP
# --------------------------------------------------

mp = mp.rename(
    columns={
        "formation_energy_eV_atom":
            "formation_energy",
        "energy_above_hull_eV":
            "stability"
    }
)

mp["source"] = "MaterialsProject"

mp = mp[
    [
        "formula",
        "source",
        "formation_energy",
        "stability"
    ]
]

# --------------------------------------------------
# STANDARDIZE OQMD
# --------------------------------------------------

oqmd = oqmd.rename(
    columns={
        "formation_energy_eV_atom":
            "formation_energy",
        "stability_eV_atom":
            "stability"
    }
)

oqmd["source"] = "OQMD"

oqmd = oqmd[
    [
        "formula",
        "source",
        "formation_energy",
        "stability"
    ]
]

# --------------------------------------------------
# COMBINE ALLOY SOURCES
# --------------------------------------------------

alloys = pd.concat(
    [mp, oqmd],
    ignore_index=True
)

alloys.drop_duplicates(
    subset=["formula"],
    inplace=True
)

print(
    "Unique alloys:",
    len(alloys)
)

# --------------------------------------------------
# ELASTIC DATA
# --------------------------------------------------

elastic = elastic[
    [
        "formula",
        "G_VRH",
        "K_VRH",
        "poisson_ratio"
    ]
]

# Pugh Ratio
elastic["pugh_ratio"] = (
    elastic["K_VRH"]
    / elastic["G_VRH"]
)

# --------------------------------------------------
# EXPERIMENTAL ENTHALPY
# --------------------------------------------------

expt = expt[
    [
        "formula",
        "e_form expt",
        "e_form mp",
        "e_form oqmd"
    ]
]

kingsbury = kingsbury[
    [
        "formula",
        "expt_form_e"
    ]
]

# --------------------------------------------------
# MERGE EVERYTHING
# --------------------------------------------------

dataset = alloys.merge(
    elastic,
    on="formula",
    how="left"
)

dataset = dataset.merge(
    expt,
    on="formula",
    how="left"
)

dataset = dataset.merge(
    kingsbury,
    on="formula",
    how="left"
)

# --------------------------------------------------
# SAVE
# --------------------------------------------------

dataset.to_csv(
    "datasets/external_validation_dataset.csv",
    index=False
)

print()
print(
    "Saved:",
    "datasets/external_validation_dataset.csv"
)
print(
    "Rows:",
    len(dataset)
)

print()
print(dataset.head())