# -*- coding: utf-8 -*-
"""
process_mendeley_steel.py

Ingests the original Mendeley Steel Mechanical Properties Excel file
(extracted from 'A database of mechanical properties of steels.zip')
and creates an enriched, cleanly formatted CSV with full metallurgical context:
  - 18 elemental compositions (Al, Cu, Mn, N, Ni, Ti, S, Fe, Zr, P, Si, V, Mo, Co, C, Nb, B, Cr)
  - 3 mechanical property targets (UTS, Yield Strength, Ductility %)
  - Raw processing conditions text
  - Parsed heat-treatment category, quench media, quench temps, temper temps
  - Structural clusters and cluster labels
"""

import os
import re
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
EXCEL_PATH = os.path.join(
    PROJECT_ROOT, "data", "raw", "mendeley_steel_extracted",
    "A database of mechanical properties of steels",
    "Steel database with labelled clusters.xlsx"
)
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "mendeley_steel_3234.csv")

def extract_heat_treatment_metadata(proc_str):
    proc = str(proc_str).strip()
    proc_lower = proc.lower()
    
    # 1. Broad Category
    if "quench" in proc_lower and "temper" in proc_lower:
        ht = "quenched_tempered"
    elif "quench" in proc_lower:
        ht = "quenched_tempered"
    elif "normaliz" in proc_lower:
        ht = "normalized"
    elif "anneal" in proc_lower:
        ht = "annealed"
    elif "age" in proc_lower or "hardened" in proc_lower:
        ht = "age_hardened"
    elif "hot rolled" in proc_lower or "plate" in proc_lower:
        ht = "normalized"
    else:
        ht = "quenched_tempered"
        
    # 2. Quench / Austenitizing Temp
    quench_match = re.search(r'(?:quenched|normalized|reheated to)\s*(\d{3,4})\s*c', proc, re.IGNORECASE)
    quench_temp = int(quench_match.group(1)) if quench_match else None
    
    # 3. Tempering Temp
    temper_match = re.search(r'(\d{2,4})\s*c\s*\([^)]*\)\s*temper', proc, re.IGNORECASE)
    if not temper_match:
        temper_match = re.search(r'temper(?:ed)?\s*(?:at\s*)?(\d{2,4})\s*c', proc, re.IGNORECASE)
    temper_temp = int(temper_match.group(1)) if temper_match else None
    
    # 4. Quench Medium
    medium = "air"
    if "oil" in proc_lower:
        medium = "oil"
    elif "water" in proc_lower:
        medium = "water"
    elif "air" in proc_lower:
        medium = "air"
        
    return ht, quench_temp, temper_temp, medium

def process_mendeley_dataset():
    print(f"[*] Reading Mendeley Excel file from: {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH)
    print(f"[+] Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Clean up column names
    elements = ['Al', 'Cu', 'Mn', 'N', 'Ni', 'Ti', 'S', 'Fe', 'Zr', 'P', 'Si', 'V', 'Mo', 'Co', 'C', 'Nb', 'B', 'Cr']
    for el in elements:
        df[el] = pd.to_numeric(df[el], errors='coerce').fillna(0.0).round(4)
        
    df['(Ultimate) Tensile strength (MPa)'] = pd.to_numeric(df['(Ultimate) Tensile strength (MPa)'], errors='coerce')
    df['Yield strength (MPa)'] = pd.to_numeric(df['Yield strength (MPa)'], errors='coerce')
    df['Ductility (%)'] = pd.to_numeric(df['Ductility (%)'], errors='coerce')
    
    # Clean up string columns
    df['Name'] = df['Name'].astype(str).str.strip()
    df['Processing condition'] = df['Processing condition'].astype(str).str.strip()
    df['Cluster label'] = df['Cluster label'].astype(str).str.strip().str.replace('', '–')
    
    # Extract metadata
    meta = [extract_heat_treatment_metadata(p) for p in df['Processing condition']]
    df['heat_treatment'] = [m[0] for m in meta]
    df['quench_temp_c'] = [m[1] for m in meta]
    df['temper_temp_c'] = [m[2] for m in meta]
    df['quench_medium'] = [m[3] for m in meta]
    
    # Select and order columns cleanly
    selected_cols = [
        'Entry', 'Name', 'Cluster Number', 'Cluster label',
        'heat_treatment', 'quench_medium', 'quench_temp_c', 'temper_temp_c',
        'Processing condition',
        'Fe', 'C', 'Mn', 'Si', 'Cr', 'Ni', 'Mo', 'V', 'Nb', 'Ti', 'Al', 'Cu', 'Co', 'B', 'Zr', 'N', 'P', 'S',
        '(Ultimate) Tensile strength (MPa)', 'Yield strength (MPa)', 'Ductility (%)'
    ]
    
    # Some columns might have alternate names in Excel
    available_cols = [c for c in selected_cols if c in df.columns]
    out_df = df[available_cols].copy()
    
    # Rename for cleaner access
    out_df.rename(columns={
        '(Ultimate) Tensile strength (MPa)': 'uts_mpa',
        'Yield strength (MPa)': 'yield_strength_mpa',
        'Ductility (%)': 'ductility_pct',
        'Processing condition': 'processing_condition',
        'Cluster Number': 'cluster_number',
        'Cluster label': 'cluster_label',
        'Name': 'steel_grade'
    }, inplace=True)
    
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[+] Successfully exported enriched dataset to: {OUTPUT_CSV}")
    print(f"[+] Total samples: {len(out_df)}, Total columns: {len(out_df.columns)}")
    print(f"[+] Heat treatment distribution:\n{out_df['heat_treatment'].value_counts()}")
    print(f"[+] Quench medium distribution:\n{out_df['quench_medium'].value_counts()}")
    print(f"[+] Mean UTS: {out_df['uts_mpa'].mean():.1f} MPa, Mean YS: {out_df['yield_strength_mpa'].mean():.1f} MPa")

if __name__ == "__main__":
    process_mendeley_dataset()
