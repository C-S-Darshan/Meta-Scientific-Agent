import requests
import json
import os

def fetch_wikidata_applications():
    print("Initiating SPARQL Query to Wikidata for Alloy Applications...")
    
    url = 'https://query.wikidata.org/sparql'
    
    # Query: Get instances or subclasses of Alloy (Q190966) or Steel (Q11427)
    # that have a "use" (P366) defined.
    query = """
    SELECT ?materialLabel ?applicationLabel ?materialDescription WHERE {
      { ?material wdt:P279* wd:Q190966. }
      UNION
      { ?material wdt:P279* wd:Q11427. }
      
      ?material wdt:P366 ?application.
      
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    """
    
    # Wikidata requires a proper User-Agent to avoid HTTP 403 Forbidden
    headers = {
        'User-Agent': 'MetaScientificAgent/1.0 (Research Project; contact: agent@meta.edu)',
        'Accept': 'application/sparql-results+json'
    }
    
    try:
        response = requests.get(url, params={'query': query, 'format': 'json'}, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        results = data.get('results', {}).get('bindings', [])
        
        # Parse into a clean JSON structure
        applications_db = []
        for row in results:
            material = row.get('materialLabel', {}).get('value', 'Unknown')
            application = row.get('applicationLabel', {}).get('value', 'Unknown')
            desc = row.get('materialDescription', {}).get('value', 'No description available')
            
            # Filter out wikidata IDs that didn't resolve to english labels
            if not material.startswith('Q') and not application.startswith('Q'):
                applications_db.append({
                    "Alloy": material,
                    "Application": application,
                    "Description": desc
                })
        
        # Deduplicate list
        unique_db = {f"{d['Alloy']}_{d['Application']}": d for d in applications_db}.values()
        unique_db = list(unique_db)
        
        # Output saving
        out_dir = os.path.join("data", "official_datasets")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "wikidata_applications.json")
        
        with open(out_path, "w") as f:
            json.dump(unique_db, f, indent=4)
            
        print(f"SUCCESS: Extracted {len(unique_db)} clean application mappings from Wikidata.")
        print(f"Saved to: {out_path}")
        
    except Exception as e:
        print(f"Error fetching data from Wikidata: {str(e)}")

if __name__ == "__main__":
    fetch_wikidata_applications()
