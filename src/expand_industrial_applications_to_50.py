# -*- coding: utf-8 -*-
"""
expand_industrial_applications_to_50.py

Expands the industrial applications dataset from 30 to 50 comprehensive
engineering challenge alloys across 5 core metallurgic families:
  - Steels (Structural, Carburizing, High-Strength, Tool & Die)
  - Stainless & Corrosion-Resistant Alloys (Austenitic, Duplex, Martensitic, PH)
  - Nickel & Cobalt High-Temperature Superalloys
  - Titanium Aerospace Alloys (Alpha-Beta, Near-Alpha, Beta)
  - High-Strength Aluminum Alloys & Refractory HEAs

Quality Guarantees:
  - Preserves all 30 existing industrial applications intact
  - Adds 20 new standards with exact stoichiometries summing to 100%
  - Verifies all formulas against formulas.py::parse_composition and ELEMENT_DATA
  - Exports data/raw/industrial_applications_v2.json (with backup) and data/raw/industrial_applications_50.json
"""

import os
import json
import shutil
import sys
import re

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from formulas import parse_composition, ELEMENT_DATA

NEW_APPLICATIONS = [
    # --- TITANIUM AEROSPACE & BIOMEDICAL ALLOYS ---
    {
        "alloy_name": "Ti-6Al-4V (Grade 5)",
        "application": "Aerospace airframe forgings",
        "engineering_focus": ["specific strength", "fatigue resistance", "lightweight structural"],
        "uts_mpa": 950.0,
        "elongation_pct": 14.0,
        "pren": 0.0,
        "specific_strength": 214.46,
        "chemistry": "Ti89.35-Al6.2-V4.1-Fe0.3-C0.05",
        "source": "AMS 4928 / ASTM B348"
    },
    {
        "alloy_name": "Ti-6Al-4V ELI (Grade 23)",
        "application": "Orthopedic surgical implants",
        "engineering_focus": ["biocompatibility", "damage tolerance", "fracture toughness"],
        "uts_mpa": 860.0,
        "elongation_pct": 15.0,
        "pren": 0.0,
        "specific_strength": 194.13,
        "chemistry": "Ti89.85-Al6.0-V4.0-Fe0.1-C0.03-N0.02",
        "source": "ASTM F136 / ISO 5832-3"
    },
    {
        "alloy_name": "Ti-5Al-5V-5Mo-3Cr (Ti-5553)",
        "application": "Heavy landing gear forgings",
        "engineering_focus": ["ultra-high strength", "fracture toughness", "deep hardenability"],
        "uts_mpa": 1240.0,
        "elongation_pct": 8.0,
        "pren": 12.9,
        "specific_strength": 266.67,
        "chemistry": "Ti82.0-Al5.0-V5.0-Mo5.0-Cr3.0",
        "source": "AMS 4999 / Boeing BMS 7-360"
    },
    {
        "alloy_name": "Beta-21S (Ti-15Mo-3Nb-3Al-0.2Si)",
        "application": "Aircraft exhaust nacelle structures",
        "engineering_focus": ["oxidation resistance", "elevated temperature strength", "hydraulic fluid resistance"],
        "uts_mpa": 1100.0,
        "elongation_pct": 12.0,
        "pren": 49.5,
        "specific_strength": 222.67,
        "chemistry": "Ti78.8-Mo15.0-Nb3.0-Al3.0-Si0.2",
        "source": "AMS 4897 / ASTM B265"
    },
    {
        "alloy_name": "Ti-6242 (Ti-6Al-2Sn-4Zr-2Mo-0.1Si)",
        "application": "Gas turbine high-pressure compressor stages",
        "engineering_focus": ["high-temperature creep resistance", "thermal stability", "creep strength"],
        "uts_mpa": 1010.0,
        "elongation_pct": 13.0,
        "pren": 6.6,
        "specific_strength": 222.47,
        "chemistry": "Ti85.9-Al6.0-Sn2.0-Zr4.0-Mo2.0-Si0.1",
        "source": "AMS 4919 / GE C50TF12"
    },

    # --- HIGH-STRENGTH STRUCTURAL ALUMINUM ALLOYS ---
    {
        "alloy_name": "Aluminum 7075-T6",
        "application": "Aircraft upper wing skins and bulkheads",
        "engineering_focus": ["maximum specific strength", "lightweight structural", "compression yield strength"],
        "uts_mpa": 572.0,
        "elongation_pct": 11.0,
        "pren": 0.0,
        "specific_strength": 203.56,
        "chemistry": "Al90.0-Zn5.6-Mg2.5-Cu1.6-Cr0.23-Fe0.07",
        "source": "AMS 4045 / ASTM B209"
    },
    {
        "alloy_name": "Aluminum 2024-T3",
        "application": "Aircraft fuselage tension skins",
        "engineering_focus": ["damage tolerance", "fatigue crack growth resistance", "tensile ductility"],
        "uts_mpa": 483.0,
        "elongation_pct": 18.0,
        "pren": 0.0,
        "specific_strength": 173.74,
        "chemistry": "Al93.5-Cu4.4-Mg1.5-Mn0.6",
        "source": "AMS 4037 / QQ-A-250/4"
    },
    {
        "alloy_name": "Aluminum 6061-T6",
        "application": "Marine and architectural welded frames",
        "engineering_focus": ["weldability", "atmospheric corrosion", "general structural durability"],
        "uts_mpa": 310.0,
        "elongation_pct": 17.0,
        "pren": 0.0,
        "specific_strength": 114.81,
        "chemistry": "Al97.9-Mg1.0-Si0.6-Cu0.28-Cr0.2-Fe0.02",
        "source": "ASTM B221 / AMS 4150"
    },
    {
        "alloy_name": "Aluminum-Lithium 2099",
        "application": "Cryogenic rocket propellant tanks",
        "engineering_focus": ["cryogenic toughness", "high elastic modulus", "ultra-low density"],
        "uts_mpa": 550.0,
        "elongation_pct": 10.0,
        "pren": 0.0,
        "specific_strength": 209.13,
        "chemistry": "Al94.3-Cu2.7-Mg0.7-Zn0.7-Mn0.6-Zr0.1-Ti0.1-Fe0.8",
        "source": "AMS 4457 / NASA-STD-5001"
    },
    {
        "alloy_name": "Aluminum 7050-T7451",
        "application": "Heavy aerospace fuselage frames",
        "engineering_focus": ["stress corrosion cracking resistance", "quench sensitivity", "thick section toughness"],
        "uts_mpa": 524.0,
        "elongation_pct": 11.0,
        "pren": 0.0,
        "specific_strength": 185.16,
        "chemistry": "Al89.0-Zn6.2-Cu2.3-Mg2.2-Zr0.12-Fe0.18",
        "source": "AMS 4050 / ASTM B247"
    },

    # --- ADVANCED NICKEL & COBALT SUPERALLOYS ---
    {
        "alloy_name": "Inconel 939",
        "application": "Industrial gas turbine hot gas vanes",
        "engineering_focus": ["hot corrosion resistance", "high temperature creep", "sulfidation resistance"],
        "uts_mpa": 980.0,
        "elongation_pct": 14.0,
        "pren": 27.8,
        "specific_strength": 119.51,
        "chemistry": "Ni48.0-Cr22.5-Co19.0-Ti3.7-Al1.9-W2.0-Ta1.4-Nb1.0-C0.15-Zr0.1-B0.25",
        "source": "UNS N07939 / ISO 15156"
    },
    {
        "alloy_name": "Haynes 282",
        "application": "Advanced ultra-supercritical steam headers",
        "engineering_focus": ["creep rupture strength", "weldability", "thermal stability"],
        "uts_mpa": 1147.0,
        "elongation_pct": 28.0,
        "pren": 48.0,
        "specific_strength": 138.69,
        "chemistry": "Ni56.4-Cr19.5-Co10.0-Mo8.5-Ti2.1-Al1.5-Fe1.5-Mn0.3-Si0.1-C0.06-B0.02-P0.01-S0.01",
        "source": "AMS 5951 / ASME Code Case 2727"
    },
    {
        "alloy_name": "Hastelloy X",
        "application": "Gas turbine combustion liners",
        "engineering_focus": ["oxidation resistance", "fabricability", "high temperature ductility"],
        "uts_mpa": 755.0,
        "elongation_pct": 43.0,
        "pren": 51.7,
        "specific_strength": 91.85,
        "chemistry": "Ni47.0-Cr22.0-Fe18.0-Mo9.0-Co1.5-W0.6-Mn0.5-Si0.5-C0.1-Al0.4-Ti0.4",
        "source": "AMS 5536 / UNS N06002"
    },
    {
        "alloy_name": "Monel K-500",
        "application": "Subsea non-magnetic drill collars and pump shafts",
        "engineering_focus": ["seawater erosion resistance", "high hardness", "non-magnetic permeability"],
        "uts_mpa": 1100.0,
        "elongation_pct": 20.0,
        "pren": 0.0,
        "specific_strength": 129.41,
        "chemistry": "Ni65.0-Cu29.5-Al2.8-Ti0.6-Fe1.0-Mn0.6-Si0.3-C0.2",
        "source": "UNS N05500 / ASTM B865"
    },
    {
        "alloy_name": "Incoloy 800H",
        "application": "Steam methane reforming furnace pigtails",
        "engineering_focus": ["creep resistance", "carburization resistance", "thermal shock resistance"],
        "uts_mpa": 600.0,
        "elongation_pct": 40.0,
        "pren": 21.0,
        "specific_strength": 75.57,
        "chemistry": "Fe45.0-Ni32.0-Cr21.0-Mn0.8-Si0.4-Al0.38-Ti0.38-C0.04",
        "source": "ASTM B408 / ASME SB-408"
    },

    # --- ULTRA-HIGH-STRENGTH DEFENSE & AEROSPACE STEELS ---
    {
        "alloy_name": "AERMET 100",
        "application": "Carrier aircraft arrestor hooks and landing gear",
        "engineering_focus": ["ultra-high tensile strength", "stress corrosion cracking resistance", "fracture toughness"],
        "uts_mpa": 1965.0,
        "elongation_pct": 14.0,
        "pren": 14.0,
        "specific_strength": 248.73,
        "chemistry": "Fe70.92-Co13.4-Ni11.1-Cr3.1-Mo1.2-C0.23-Ti0.05",
        "source": "AMS 6532 / MIL-S-8964"
    },
    {
        "alloy_name": "300M Steel (4340M)",
        "application": "Military transport aircraft landing gear",
        "engineering_focus": ["high strength", "fatigue resistance", "transverse ductility"],
        "uts_mpa": 2000.0,
        "elongation_pct": 10.0,
        "pren": 3.75,
        "specific_strength": 255.1,
        "chemistry": "Fe94.13-C0.42-Ni1.8-Si1.6-Cr0.8-Mn0.75-Mo0.4-V0.08-P0.01-S0.01",
        "source": "AMS 6417 / MIL-S-8844"
    },
    {
        "alloy_name": "Custom 465 Stainless",
        "application": "Medical endoscopic shafts and aerospace fasteners",
        "engineering_focus": ["precipitation hardening", "marine chloride resistance", "high yield strength"],
        "uts_mpa": 1795.0,
        "elongation_pct": 12.0,
        "pren": 14.5,
        "specific_strength": 230.13,
        "chemistry": "Fe74.19-Cr11.75-Ni11.0-Ti1.6-Mo0.9-Mn0.25-Si0.25-C0.02-Al0.02-P0.01-S0.01",
        "source": "ASTM F899 / AMS 5936"
    },
    {
        "alloy_name": "PH 13-8 Mo Stainless",
        "application": "Missile structural pins and launch components",
        "engineering_focus": ["transverse fracture toughness", "stress corrosion resistance", "deep section hardenability"],
        "uts_mpa": 1520.0,
        "elongation_pct": 14.0,
        "pren": 20.4,
        "specific_strength": 195.88,
        "chemistry": "Fe75.32-Cr12.75-Ni8.25-Mo2.25-Al1.15-Mn0.1-Si0.1-C0.05-N0.01-P0.01-S0.01",
        "source": "AMS 5629 / UNS S13800"
    },

    # --- REFRACTORY HIGH-ENTROPY ALLOY (RHEA) ---
    {
        "alloy_name": "Mo-Nb-Ta-W Refractory HEA",
        "application": "Hypersonic leading edges and rocket nozzles",
        "engineering_focus": ["ultra-high temperature strength", "creep resistance", "thermal stability"],
        "uts_mpa": 1280.0,
        "elongation_pct": 6.0,
        "pren": 0.0,
        "specific_strength": 93.43,
        "chemistry": "Mo25.0-Nb25.0-Ta25.0-W25.0",
        "source": "Senkov et al. / Nature Materials"
    }
]

def expand_applications():
    v2_path = os.path.join(PROJECT_ROOT, "data", "raw", "industrial_applications_v2.json")
    v2_backup_path = os.path.join(PROJECT_ROOT, "data", "raw", "industrial_applications_v2_backup30.json")
    v50_path = os.path.join(PROJECT_ROOT, "data", "raw", "industrial_applications_50.json")

    with open(v2_path, "r", encoding="utf-8") as f:
        existing = json.load(f)
    print(f"[+] Loaded existing applications: {len(existing)}")

    # Backup existing v2
    if not os.path.exists(v2_backup_path):
        shutil.copyfile(v2_path, v2_backup_path)
        print(f"[+] Created safety backup -> {v2_backup_path}")

    # Validate each new entry
    validated_new = []
    for item in NEW_APPLICATIONS:
        comp_str = item["chemistry"]
        clean_text = re.sub(r'[\[\]% \(\)]', '', comp_str)
        matches = re.findall(r'([A-Z][a-z]?)-?([0-9]*\.?[0-9]+)', clean_text)
        raw_sum = sum(float(val) for el, val in matches if el in ELEMENT_DATA)
        diff = abs(raw_sum - 100.0)
        assert diff < 0.2, f"Stoichiometry unbalanced for {item['alloy_name']}: {raw_sum}%"

        parsed = parse_composition(comp_str)
        assert parsed is not None, f"Failed parsing {comp_str}"
        assert abs(sum(parsed.values()) - 1.0) < 1e-4, f"Normalized sum != 1.0 for {item['alloy_name']}"
        
        # Verify all elements in formulas.py
        for el in parsed.keys():
            assert el in ELEMENT_DATA, f"Unknown element {el} in {item['alloy_name']}"

        validated_new.append(item)
        print(f"  [OK] Validated: {item['alloy_name']:<35} | {item['application']:<45} | Raw Sum: {raw_sum:.2f}%")

    combined = existing + validated_new
    print(f"[+] Combined total applications: {len(combined)}")

    # Write out to industrial_applications_50.json and update industrial_applications_v2.json
    with open(v50_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    with open(v2_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"[+] Saved 50-application catalog to:\n  - {v50_path}\n  - {v2_path}")

if __name__ == "__main__":
    expand_applications()
