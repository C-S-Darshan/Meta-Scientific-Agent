import os
import requests
import pandas as pd
import json
import time

# Load API key from environment variable (do not hardcode secrets)
API_KEY = os.getenv("MP_API_KEY", "")


# Elements supported by formulas.py
ALLOWED_ELEMENTS = {
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

FIELDS = [
    "material_id",
    "formula_pretty",
    "composition",
    "density",
    "energy_above_hull",
    "formation_energy_per_atom",
    "is_stable",
    "symmetry"
]

BASE_URL = "https://api.materialsproject.org/materials/summary"

HEADERS = {
    "X-API-KEY": API_KEY
}

LIMIT = 1000

records = []
seen = set()

for n_elements in range(2, 6):

    print(f"\n========== nelements = {n_elements} ==========")

    skip = 0

    while True:

        print(f"Fetching batch starting at {skip}")

        params = {
            "elements": "Fe",
            "nelements": n_elements,
            "_fields": ",".join(FIELDS),
            "_limit": LIMIT,
            "_skip": skip
        }

        success = False

        for attempt in range(5):

            try:

                response = requests.get(
                    BASE_URL,
                    headers=HEADERS,
                    params=params,
                    timeout=120
                )

                if response.status_code != 200:

                    print(f"\nHTTP {response.status_code}")
                    print(response.text[:1000])

                    time.sleep(15)
                    continue

                body = response.json()

                if "data" not in body:

                    print("Unexpected response:")
                    print(body)
                    break

                docs = body["data"]

                success = True
                break

            except Exception as e:

                print(f"Attempt {attempt+1}/5 failed:")
                print(e)

                time.sleep(15)

        if not success:
            print("Giving up on this batch.")
            break

        if len(docs) == 0:
            print("No more results.")
            break

        print(f"Returned {len(docs)} entries")

        for doc in docs:

            material_id = doc["material_id"]

            if material_id in seen:
                continue

            seen.add(material_id)

            composition = doc["composition"]

            elements = set(composition.keys())

            # Must contain Fe
            if "Fe" not in elements:
                continue

            # Must contain only formulas.py elements
            if not elements.issubset(ALLOWED_ELEMENTS):
                continue

            e_hull = doc.get("energy_above_hull")

            if e_hull is None:
                continue

            # Stable or near-stable compounds
            if e_hull > 0.2:
                continue

            symmetry = doc.get("symmetry")

            crystal_system = None
            space_group = None

            if symmetry is not None:
                crystal_system = symmetry.get("crystal_system")
                space_group = symmetry.get("symbol")

            records.append({
                "material_id": material_id,
                "formula": doc["formula_pretty"],
                "composition": json.dumps(composition),
                "density_g_cm3": doc.get("density"),
                "formation_energy_eV_atom":
                    doc.get("formation_energy_per_atom"),
                "energy_above_hull_eV": e_hull,
                "stable": doc.get("is_stable"),
                "crystal_system": crystal_system,
                "space_group": space_group
            })

        print("Current alloy count:", len(records))

        # Save checkpoint after every batch
        pd.DataFrame(records).drop_duplicates(
            subset=["material_id"]
        ).to_csv(
            "materials_project_supported_alloys_partial.csv",
            index=False
        )

        skip += LIMIT

        # Polite rate limiting
        time.sleep(3)

# Final dataframe
df = pd.DataFrame(records)

if len(df) > 0:

    df.drop_duplicates(
        subset=["material_id"],
        inplace=True
    )

    df.sort_values(
        by="energy_above_hull_eV",
        inplace=True
    )

    df.to_csv(
        "materials_project_supported_alloys.csv",
        index=False
    )

    print("\n============================")
    print("Done")
    print("Final alloy count:", len(df))
    print("============================")

else:

    print("\nNo alloys downloaded.")