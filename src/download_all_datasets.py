# -*- coding: utf-8 -*-
"""
download_all_datasets.py
========================
Master dataset acquisition script for Meta-Scientific Agent - IEEE Access resubmission.

Run from project root:
    python src/download_all_datasets.py

Automatically downloads everything accessible via matminer/requests.
For datasets requiring browser auth (Mendeley), prints clear instructions.
"""

import os, sys, json, time, re, requests, pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_DATASETS = PROJECT_ROOT / "src" / "datasets"
DATA_RAW.mkdir(parents=True, exist_ok=True)

try:
    from matminer.datasets import load_dataset
    MATMINER_OK = True
except ImportError:
    MATMINER_OK = False
    print("[WARN] matminer not installed.")


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def download_matminer():
    section("1. Matminer Datasets")
    if not MATMINER_OK:
        return
    datasets = [
        ("matbench_steels", DATA_DATASETS / "matbench_steels.csv"),
        ("steel_strength", DATA_DATASETS / "steel_strength.csv"),
        ("expt_formation_enthalpy_kingsbury", DATA_DATASETS / "expt_formation_enthalpy_kingsbury.csv"),
    ]
    for name, out in datasets:
        if out.exists():
            print(f"  [SKIP] {out.name} ({out.stat().st_size//1024} KB)")
            continue
        try:
            df = load_dataset(name)
            df.to_csv(out, index=False)
            print(f"  [OK] {out.name}: {len(df)} rows")
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")


def download_github():
    section("2. GitHub Datasets")
    datasets = [
        ("MPEA Citrine (Borg 2020)",
         "https://raw.githubusercontent.com/CitrineInformatics/MPEA_dataset/master/MPEA_dataset.csv",
         DATA_RAW / "MPEA_dataset_fresh.csv",
         "DOI:10.1038/s41597-020-00768-9"),
    ]
    for name, url, out, cite in datasets:
        if out.exists():
            print(f"  [SKIP] {out.name}")
            continue
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                out.write_bytes(r.content)
                print(f"  [OK] {out.name} ({len(r.content)//1024} KB) | {cite}")
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")


def generate_proxy():
    section("3. SteelBench Proxy (composition-derived heat treatment)")
    out = DATA_RAW / "steelbench_proxy.csv"
    if out.exists():
        print(f"  [SKIP] {out.name}")
        return
    steel_path = DATA_RAW / "steel_strength.json"
    if not steel_path.exists():
        print("  [SKIP] steel_strength.json not found")
        return
    with open(steel_path) as f:
        raw = json.load(f)
    cols = raw["columns"]
    df = pd.DataFrame([dict(zip(cols, r)) for r in raw["data"]])

    def ht(row):
        c  = row.get("c",  0) or 0
        ni = row.get("ni", 0) or 0
        cr = row.get("cr", 0) or 0
        mo = row.get("mo", 0) or 0
        if ni > 8.0 and c < 0.05:            return "age_hardened"
        elif c > 0.3 and (cr > 0.5 or mo > 0.2): return "quenched_tempered"
        elif c > 0.5:                         return "quenched_tempered"
        elif cr > 15.0:                       return "annealed"
        else:                                 return "normalized"

    df["heat_treatment"] = df.apply(ht, axis=1)
    keep = [c for c in ["formula","c","mn","si","cr","ni","mo","v","n","nb","co","w","al","ti",
                        "tensile strength","yield strength","elongation","heat_treatment"] if c in df.columns]
    proxy = df[keep]
    proxy.to_csv(out, index=False)
    print(f"  [OK] {out.name}: {len(proxy)} rows")
    for ht_val, cnt in proxy["heat_treatment"].value_counts().items():
        print(f"       {ht_val}: {cnt}")
    print("  [NOTE] R2-C5 fix: heat treatment assigned from composition, NOT from label.")


def status_report():
    section("4. Status Report")
    files = [
        (DATA_RAW/"steel_strength.json", "Primary steel (312)"),
        (DATA_RAW/"MPEA_dataset.csv", "MPEA Citrine (1545)"),
        (DATA_RAW/"steelbench_core.csv", "SteelBench core - PRIORITY 1"),
        (DATA_RAW/"steelbench_full.csv", "SteelBench full - PRIORITY 1"),
        (DATA_RAW/"steelbench_proxy.csv", "SteelBench proxy (generated)"),
        (DATA_RAW/"mendeley_steel_3234.csv", "Mendeley 3234 - MANUAL DL"),
        (DATA_RAW/"kmrp_steel_grades.csv", "KMRP grades - MANUAL DL"),
        (DATA_DATASETS/"matbench_steels.csv", "MatBench steels (312)"),
        (DATA_DATASETS/"expt_formation_enthalpy_kingsbury.csv", "Kingsbury enthalpy (2135)"),
        (DATA_DATASETS/"external_validation_dataset.csv", "External validation (265)"),
    ]
    for p, desc in files:
        status = f"OK  {p.stat().st_size//1024:>6} KB" if p.exists() else "MISSING"
        print(f"  {'OK' if p.exists() else 'XX'}  {p.name:<50} {status}")

    section("5. Manual Download Instructions")
    print("""
  MENDELEY (3,234 steel entries):
    URL: https://data.mendeley.com/datasets/jmwb9ddd43/1
    DOI: 10.17632/jmwb9ddd43.1
    Steps:
      1. Open URL in browser, create free account
      2. Download CSV/Excel file
      3. Save as: data/raw/mendeley_steel_3234.csv

  STEELBENCH (1,636 entries with heat treatment):
    URL: https://zenodo.org/records/18530558
    DOI: 10.5281/zenodo.18530558
    Steps:
      1. Open URL in browser
      2. Download steelbench_core.csv + steelbench_full.csv
      3. Save to: data/raw/
    OR: git clone https://github.com/cornada/steelbench.git

  KMRP STEEL GRADES (standard grades from AISI/ASTM):
    URL: https://zenodo.org/records/14660301
    DOI: 10.5281/zenodo.14660301
    Steps:
      1. Open URL in browser, download CSV
      2. Save as: data/raw/kmrp_steel_grades.csv

  COD'HEM HEA DATABASE (for CALPHAD phase validation R1-C2):
    URL: https://imet-db.ru
    Steps:
      1. Open URL, filter by Fe or Ni base
      2. Export to CSV
      3. Save as: data/raw/codhem_hea_export.csv
    """)


if __name__ == "__main__":
    print("Meta-Scientific Agent - Dataset Acquisition Script")
    download_matminer()
    download_github()
    generate_proxy()
    status_report()
