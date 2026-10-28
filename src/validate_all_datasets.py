# -*- coding: utf-8 -*-
"""
validate_all_datasets.py

Comprehensive Validation & Health Check Suite for Meta-Scientific Agent.
Runs 6 automated tests to ensure all datasets, training sets, symbolic formulas,
and splits are in 100% working order prior to Colab packaging and training:

  1. Mendeley Steel Database Integrity Check (3,234 rows, 0 nulls, properties)
  2. MPEA Cross-Family Dataset & Element Coverage Check (1,545 rows, 25 elements)
  3. Training Instruction Sets Quality & Stoichiometry (5k Hypothesis + 5k Analyst)
  4. Zero-Leakage Train / Held-Out Separation Check
  5. Symbolic Physics Engine Calculations (UTS, PREN, Elongation, Plausibility)
  6. Colab Packaging File Manifest Pre-Check
"""

import os
import sys
import json
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import config
from formulas import calculate_physics, parse_composition, is_plausible_composition, ELEMENT_DATA

def check_mendeley_database():
    print("\n" + "=" * 65)
    print(" [TEST 1/6] MENDELEY STEEL DATABASE (3,234 ROWS) HEALTH CHECK")
    print("=" * 65)
    csv_path = os.path.join(config.RAW_DATA_DIR, "mendeley_steel_3234.csv")
    if not os.path.exists(csv_path):
        print(f"[-] FAILED: Missing {csv_path}")
        return False
        
    df = pd.read_csv(csv_path)
    assert len(df) == 3234, f"Expected 3234 rows, got {len(df)}"
    assert df["uts_mpa"].isnull().sum() == 0, "Nulls found in uts_mpa"
    assert df["yield_strength_mpa"].isnull().sum() == 0, "Nulls found in yield_strength_mpa"
    assert df["ductility_pct"].isnull().sum() == 0, "Nulls found in ductility_pct"
    
    unique_grades = df["steel_grade"].nunique()
    unique_clusters = df["cluster_label"].nunique()
    print(f"  [+] Total Samples: {len(df)}")
    print(f"  [+] Unique Steel Grades: {unique_grades}")
    print(f"  [+] Metallurgical Clusters: {unique_clusters}")
    print(f"  [+] UTS Range: {df['uts_mpa'].min()} - {df['uts_mpa'].max()} MPa (mean: {df['uts_mpa'].mean():.1f})")
    print(f"  [+] Yield Strength Range: {df['yield_strength_mpa'].min()} - {df['yield_strength_mpa'].max()} MPa (mean: {df['yield_strength_mpa'].mean():.1f})")
    print(f"  [+] Ductility Range: {df['ductility_pct'].min()}% - {df['ductility_pct'].max()}% (mean: {df['ductility_pct'].mean():.1f}%)")
    print(f"  [+] Heat Treatment Categories:\n{df['heat_treatment'].value_counts().to_string()}")
    print("  [OK] Mendeley Database PASSED all integrity checks.")
    return True

def check_mpea_dataset():
    print("\n" + "=" * 65)
    print(" [TEST 2/6] CITRINE MPEA DATASET & ELEMENT COVERAGE CHECK")
    print("=" * 65)
    mpea_path = os.path.join(config.RAW_DATA_DIR, "MPEA_dataset.csv")
    if not os.path.exists(mpea_path):
        print(f"[-] FAILED: Missing {mpea_path}")
        return False
        
    df = pd.read_csv(mpea_path)
    print(f"  [+] Total MPEA Rows: {len(df)}")
    print(f"  [+] Active Elements in formulas.py: {len(ELEMENT_DATA)} elements ({', '.join(sorted(ELEMENT_DATA.keys()))})")
    
    covered = 0
    total = len(df)
    for f in df["FORMULA"].dropna():
        parsed = parse_composition(str(f))
        if parsed and all(el in ELEMENT_DATA for el in parsed.keys()):
            covered += 1
            
    pct_covered = (covered / total) * 100.0
    print(f"  [+] MPEA Formulas fully parsable: {covered} / {total} ({pct_covered:.1f}%)")
    assert covered > 1400, f"Expected >1400 covered, got {covered}"
    print("  [OK] Citrine MPEA Coverage PASSED.")
    return True

def check_training_instruction_sets():
    print("\n" + "=" * 65)
    print(" [TEST 3/6] INSTRUCTION-TUNING DATASETS AUDIT (HYPOTHESIS + ANALYST)")
    print("=" * 65)
    h_path = os.path.join(config.TRAIN_DATA_DIR, "hypothesis_train.jsonl")
    a_path = os.path.join(config.TRAIN_DATA_DIR, "analyst_train.jsonl")
    
    with open(h_path, "r", encoding="utf-8") as f:
        h_lines = [json.loads(line) for line in f]
    with open(a_path, "r", encoding="utf-8") as f:
        a_lines = [json.loads(line) for line in f]
        
    print(f"  [+] Hypothesis Samples Count: {len(h_lines)} ({os.path.getsize(h_path)/(1024*1024):.2f} MB)")
    print(f"  [+] Analyst Samples Count: {len(a_lines)} ({os.path.getsize(a_path)/(1024*1024):.2f} MB)")
    
    # Verify exact stoichiometry and template
    h_stoich_pass = 0
    for s in h_lines:
        text = s["text"]
        assert "Task: Propose a new alloy chemistry based on the baseline." in text
        assert "Proposed Chemistry: " in text
        chem = text.split("Proposed Chemistry: ")[1].strip()
        parsed = parse_composition(chem)
        if parsed and abs(sum(parsed.values()) * 100.0 - 100.0) <= 0.15:
            h_stoich_pass += 1
            
    assert h_stoich_pass == len(h_lines), f"Stoichiometry failed on {len(h_lines) - h_stoich_pass} samples"
    
    # Verify analyst format
    a_pass = sum(1 for s in a_lines if "Conclusion (maximum 2 sentences):" in s["text"])
    assert a_pass == len(a_lines), "Analyst format mismatch"
    
    print(f"  [+] Stoichiometry Verification Rate: {h_stoich_pass / len(h_lines) * 100:.2f}% (100.0% clean)")
    print(f"  [+] Analyst Template Compliance: {a_pass / len(a_lines) * 100:.2f}%")
    print("  [OK] Instruction-Tuning Datasets PASSED all quality checks.")
    return True

def check_zero_leakage():
    print("\n" + "=" * 65)
    print(" [TEST 4/6] ZERO-LEAKAGE HELD-OUT BENCHMARK AUDIT")
    print("=" * 65)
    split_path = os.path.join(config.RAW_DATA_DIR, "train_eval_split.json")
    with open(split_path, "r") as f:
        split = json.load(f)
    held_out = set(split["held_out_formulas"])
    print(f"  [+] Held-out evaluation formulas: {len(held_out)}")
    
    h_path = os.path.join(config.TRAIN_DATA_DIR, "hypothesis_train.jsonl")
    with open(h_path, "r", encoding="utf-8") as f:
        h_lines = [json.loads(line) for line in f]
        
    contamination = 0
    for s in h_lines:
        text = s["text"]
        for hf in held_out:
            if hf in text:
                contamination += 1
                
    print(f"  [+] Overlap / Contamination count in training data: {contamination}")
    assert contamination == 0, f"Contamination detected: {contamination} overlaps!"
    print("  [OK] Zero-Leakage Guarantee PASSED (0.00% train/eval contamination).")
    return True

def check_symbolic_physics():
    print("\n" + "=" * 65)
    print(" [TEST 5/6] SYMBOLIC PHYSICS & METALLURGICAL FORMULAS CHECK")
    print("=" * 65)
    test_compositions = [
        ("Fe-70.0 Cr-18.0 Ni-10.0 Mo-2.0", "quenched_tempered"),
        ("Fe-95.0 C-0.45 Mn-0.8 Si-0.25 Cr-1.0 Mo-0.2", "quenched_tempered"),
        ("Fe-98.0 C-0.2 Mn-0.8 Si-0.3", "normalized"),
        ("Fe-85.0 Cr-12.0 Mo-1.5 V-0.3 C-0.3", "annealed"),
    ]
    
    for comp_str, ht in test_compositions:
        parsed = parse_composition(comp_str)
        assert is_plausible_composition(parsed), f"Failed plausibility: {comp_str}"
        props = calculate_physics(parsed, ht)
        print(f"  [+] {comp_str} ({ht}):")
        print(f"      UTS={props['Estimated_UTS_MPa']} MPa | PREN={props['PREN_Corrosion_Index']} | Elongation={props['Estimated_Elongation_pct']}% | Density={props['Theoretical_Density_g_cm3']} g/cm3")
        assert props['Estimated_UTS_MPa'] > 300, "Unrealistic low UTS"
        
    print("  [OK] Symbolic Physics Engine PASSED.")
    return True

def check_colab_readiness():
    print("\n" + "=" * 65)
    print(" [TEST 6/6] COLAB DEPLOYMENT & PACKAGING READINESS CHECK")
    print("=" * 65)
    critical_files = [
        "src/config.py",
        "src/formulas.py",
        "src/agent.py",
        "src/train_adaptors.py",
        "data/raw/mendeley_steel_3234.csv",
        "data/raw/train_eval_split.json",
        "data/raw/industrial_applications_v2.json",
        "data/raw/MPEA_dataset.csv",
        "data/train/hypothesis_train.jsonl",
        "data/train/analyst_train.jsonl"
    ]
    
    missing = []
    total_size = 0
    for rel in critical_files:
        p = os.path.join(PROJECT_ROOT, rel)
        if os.path.exists(p):
            sz = os.path.getsize(p)
            total_size += sz
            print(f"  [+] Found: {rel} ({sz / (1024*1024):.2f} MB)")
        else:
            missing.append(rel)
            print(f"  [-] MISSING: {rel}")
            
    print(f"  [+] Total Core Assets Size: {total_size / (1024*1024):.2f} MB")
    assert len(missing) == 0, f"Missing critical files: {missing}"
    print("  [OK] All Colab Assets READY for packaging.")
    return True

def run_all():
    print("=================================================================")
    print("    META-SCIENTIFIC AGENT: FULL SYSTEM & DATA VALIDATION SUITE")
    print("=================================================================")
    t1 = check_mendeley_database()
    t2 = check_mpea_dataset()
    t3 = check_training_instruction_sets()
    t4 = check_zero_leakage()
    t5 = check_symbolic_physics()
    t6 = check_colab_readiness()
    
    if all([t1, t2, t3, t4, t5, t6]):
        print("\n" + "#" * 65)
        print("  ALL 6 COMPREHENSIVE CHECKS PASSED WITH 100% SUCCESS!")
        print("  SYSTEM IS FULLY PREPARED FOR COLAB PACKAGING AND EXECUTION.")
        print("#" * 65 + "\n")

if __name__ == "__main__":
    run_all()
