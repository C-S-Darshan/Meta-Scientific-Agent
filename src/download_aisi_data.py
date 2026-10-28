import pandas as pd
import requests
import json
import os
import config

url = "https://en.wikipedia.org/wiki/SAE_steel_grades"
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(url, headers=headers)

tables = pd.read_html(response.text)
applications_data = []

# Typically, the first few tables contain the standard carbon and alloy steels
for df in tables:
    if "Type" in df.columns and "Applications" in df.columns:
        for _, row in df.iterrows():
            app = str(row["Applications"])
            if app != "nan" and len(app) > 5:
                applications_data.append({
                    "AISI_Code": str(row["Type"]),
                    "Applications": app
                })

# Save to data/official_datasets
out_dir = os.path.join("data", "official_datasets")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "aisi_steel_applications.json")

with open(out_path, "w") as f:
    json.dump(applications_data, f, indent=4)

print(f"Successfully scraped {len(applications_data)} AISI grades with their official applications.")
print(f"Saved to {out_path}")
