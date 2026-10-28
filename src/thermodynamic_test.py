import pandas as pd
import json
from pathlib import Path

from formulas import generate_phase_data, extract_thermo_insights

# Resolve paths relative to the project root (one level above src/)
PROJECT_ROOT = Path(__file__).parent.parent

TDB_FILE = str(PROJECT_ROOT / "thermodynamic_databases" / "mc_fe_pycalphad.tdb")

df = pd.read_csv(
    PROJECT_ROOT / "src" / "datasets" / "materials_project_supported_alloys.csv"
)

row = df.iloc[0]

print("Formula:", row["formula"])

comp = json.loads(row["composition"])

print("Composition:")
print(comp)

thermo = generate_phase_data(
    comp,
    TDB_FILE
)

print(type(thermo))

if thermo is not None:

    insights = extract_thermo_insights(
        thermo,
        comp
    )

    print("\nINSIGHTS:")
    print(insights)

else:

    print("PyCalphad returned None")