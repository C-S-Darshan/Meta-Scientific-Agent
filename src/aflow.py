import requests
import json

# Simple AFLOW query:
# species contains Fe and exactly 2 species
url = (
    "http://aflowlib.duke.edu/search/API/"
    "?species(Fe),nspecies(2),"
    "$paging(1,1),"
    "compound,enthalpy_formation_atom,Egap,spacegroup_relax"
)

print("Requesting...")

r = requests.get(url, timeout=120)

print("Status code:", r.status_code)

print("\nResponse:")
print(r.text[:3000])