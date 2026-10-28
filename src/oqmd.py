import requests
import pandas as pd
import json
import time
from pymatgen.core import Composition

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

BASE_URL = "https://oqmd.org/oqmdapi/formationenergy"

FIELDS = (
    "name,"
    "entry_id,"
    "spacegroup,"
    "delta_e,"
    "stability,"
    "band_gap,"
    "composition"
)

LIMIT = 1000
OFFSET = 0

records = []
seen = set()

while OFFSET < 10000:

    print(f"\nFetching offset={OFFSET}")

    params = {
        "fields": FIELDS,
        "filter": "element_set=Fe",
        "limit": LIMIT,
        "offset": OFFSET
    }

    try:

        r = requests.get(
            BASE_URL,
            params=params,
            timeout=120
        )

        if r.status_code != 200:
            print(r.status_code)
            print(r.text)
            break

        body = r.json()

        data = body["data"]

        if len(data) == 0:
            break

        print("Returned", len(data))

        for entry in data:

            entry_id = entry["entry_id"]

            if entry_id in seen:
                continue

            seen.add(entry_id)

            formula = entry["composition"]

            try:

                comp_dict = Composition(formula).as_dict()

            except Exception:
                continue

            elements = set(comp_dict.keys())

            if "Fe" not in elements:
                continue

            if not elements.issubset(ALLOWED_ELEMENTS):
                continue

            stability = entry["stability"]

            if stability is None:
                continue

            if stability > 0.2:
                continue

            records.append({

                "entry_id": entry_id,

                "formula": formula,

                "composition":
                    json.dumps(comp_dict),

                "spacegroup":
                    entry["spacegroup"],

                "formation_energy_eV_atom":
                    entry["delta_e"],

                "stability_eV_atom":
                    stability,

                "band_gap":
                    entry["band_gap"]
            })

        print("Current alloy count:", len(records))

        OFFSET += LIMIT

        time.sleep(2)

    except Exception as e:

        print(e)
        break

df = pd.DataFrame(records)

df.drop_duplicates(
    subset=["entry_id"],
    inplace=True
)

df.sort_values(
    by="stability_eV_atom",
    inplace=True
)

df.to_csv(
    "oqmd_alloys.csv",
    index=False
)

print("\nFinal count =", len(df))