import os
import requests
import pandas as pd

def download_citrine_mpea():
    print("Fetching official Citrine Informatics MPEA dataset from GitHub...")
    url = "https://raw.githubusercontent.com/CitrineInformatics/MPEA_dataset/master/MPEA_dataset.csv"
    response = requests.get(url)
    
    if response.status_code == 200:
        out_dir = os.path.join("data", "official_datasets")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "citrine_mpea.csv")
        
        with open(out_path, "wb") as f:
            f.write(response.content)
            
        print(f"SUCCESS: Downloaded {len(response.content)} bytes.")
        print(f"Saved to: {out_path}")
        
        # Display a sample
        df = pd.read_csv(out_path)
        print("\n--- Dataset Sample ---")
        print(df.head(2))
    else:
        print(f"Failed to fetch dataset. Status Code: {response.status_code}")

if __name__ == "__main__":
    download_citrine_mpea()
