import os
import json
import faiss
import numpy as np
import pandas as pd

from pymatgen.core import Composition
from sentence_transformers import SentenceTransformer

import config


# ============================================================
# GLOBAL MODEL CACHE
# ============================================================

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_embedding_model = None


def get_embedding_model():

    global _embedding_model

    if _embedding_model is None:
        print(f"[*] Loading embedding model: {EMBED_MODEL_NAME}")
        _embedding_model = SentenceTransformer(EMBED_MODEL_NAME)

    return _embedding_model


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_float(x, default=0.0):

    try:
        if x is None:
            return default

        if pd.isna(x):
            return default

        return float(x)

    except:
        return default


def compute_pren(comp):

    cr = safe_float(comp.get("Cr", 0))
    mo = safe_float(comp.get("Mo", 0))
    n = safe_float(comp.get("N", 0))

    return round(cr + (3.3 * mo) + (16 * n), 2)


def compute_density(comp):

    ni = safe_float(comp.get("Ni", 0))
    cr = safe_float(comp.get("Cr", 0))
    mo = safe_float(comp.get("Mo", 0))
    al = safe_float(comp.get("Al", 0))
    ti = safe_float(comp.get("Ti", 0))

    density = (
        7.87
        + (0.01 * ni)
        + (0.005 * cr)
        + (0.012 * mo)
        - (0.015 * al)
        - (0.01 * ti)
    )

    return round(density, 3)


def compute_specific_strength(uts, density):

    uts = safe_float(uts)
    density = safe_float(density)

    if density <= 0:
        return 0

    return round(uts / density, 2)


def chemistry_string(comp):

    parts = []

    for el, val in comp.items():

        if safe_float(val) > 0:
            parts.append(f"{el}{round(val, 2)}")

    return "-".join(parts)


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

def save_embeddings(
    texts,
    metadata,
    index_name,
    metadata_name
):

    print(f"[*] Generating embeddings for {len(texts)} entries...")

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype("float32")

    index = faiss.IndexFlatL2(
        embeddings.shape[1]
    )

    index.add(embeddings)

    os.makedirs(
        config.EMBEDDINGS_DIR,
        exist_ok=True
    )

    faiss.write_index(
        index,
        os.path.join(
            config.EMBEDDINGS_DIR,
            index_name
        )
    )

    with open(
        os.path.join(
            config.EMBEDDINGS_DIR,
            metadata_name
        ),
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    print(f"[+] Saved: {index_name}")
    print(f"[+] Saved: {metadata_name}")


# ============================================================
# ACADEMIC ALLOY KB
# ============================================================

def build_alloy_kb():

    print("\n" + "=" * 60)
    print("[*] BUILDING ACADEMIC ALLOY KB")
    print("=" * 60)

    json_path = os.path.join(
        config.RAW_DATA_DIR,
        "steel_strength.json"
    )

    try:

        with open(json_path, "r") as f:
            raw_data = json.load(f)

    except FileNotFoundError:

        print(f"❌ ERROR: Missing file -> {json_path}")
        return

    texts = []
    metadata = []

    columns = raw_data["columns"]

    for idx, row in enumerate(raw_data["data"]):

        row_dict = dict(
            zip(columns, row)
        )

        formula = row_dict["formula"]

        try:

            comp_obj = Composition(formula)

        except:
            continue

        composition = {
            str(el): round(
                comp_obj[el],
                4
            )
            for el in comp_obj.elements
        }

        uts = safe_float(
            row_dict.get("tensile strength")
        )

        elong = safe_float(
            row_dict.get("elongation")
        )

        pren = compute_pren(
            composition
        )

        density = compute_density(
            composition
        )

        specific_strength = compute_specific_strength(
            uts,
            density
        )

        chemistry = chemistry_string(
            composition
        )

        alloy_id = f"Alloy_{idx}"

        desc = (
            f"[{alloy_id} | UTS | {uts} MPa], "
            f"[{alloy_id} | elongation | {elong}%], "
            f"[{alloy_id} | PREN | {pren}], "
            f"[{alloy_id} | specific_strength | {specific_strength}], "
            f"[{alloy_id} | chemistry | {chemistry}]"
        )

        texts.append(desc)

        metadata.append(desc)

    save_embeddings(
        texts,
        metadata,
        "academic_aisi.index",
        "academic_aisi_metadata.json"
    )


# ============================================================
# INDUSTRIAL APPLICATION KB
# ============================================================

def build_application_kb():

    print("\n" + "=" * 60)
    print("[*] BUILDING INDUSTRIAL APPLICATION KB")
    print("=" * 60)

    json_path = os.path.join(
        config.RAW_DATA_DIR,
        "industrial_applications.json"
    )

    try:

        with open(json_path, "r") as f:
            app_data = json.load(f)

    except FileNotFoundError:

        print(f"❌ ERROR: Missing file -> {json_path}")
        return

    texts = []
    metadata = []

    for idx, entry in enumerate(app_data):

        alloy_name = entry.get(
            "alloy_name",
            "Unknown"
        )

        application = entry.get(
            "application",
            "Unknown"
        )

        engineering_focus = entry.get(
            "engineering_focus",
            []
        )

        composition = entry.get(
            "composition",
            {}
        )

        uts = safe_float(
            entry.get("uts_mpa")
        )

        elong = safe_float(
            entry.get("elongation_pct")
        )

        density = safe_float(
            entry.get("density_gcc")
        )

        if density <= 0:
            density = compute_density(
                composition
            )

        pren = compute_pren(
            composition
        )

        specific_strength = compute_specific_strength(
            uts,
            density
        )

        chemistry = chemistry_string(
            composition
        )

        focus_string = ", ".join(
            engineering_focus
        )

        alloy_id = f"Industrial_{idx}"

        desc = (
            f"[{alloy_id} | alloy | {alloy_name}], "
            f"[{alloy_id} | application | {application}], "
            f"[{alloy_id} | engineering_focus | {focus_string}], "
            f"[{alloy_id} | UTS | {uts} MPa], "
            f"[{alloy_id} | elongation | {elong}%], "
            f"[{alloy_id} | PREN | {pren}], "
            f"[{alloy_id} | specific_strength | {specific_strength}], "
            f"[{alloy_id} | chemistry | {chemistry}]"
        )

        texts.append(desc)

        metadata.append(desc)

    save_embeddings(
        texts,
        metadata,
        "industrial_applications.index",
        "industrial_applications_metadata.json"
    )


# ============================================================
# METALLURGICAL REASONING KB
# ============================================================

def build_reasoning_kb():

    print("\n" + "=" * 60)
    print("[*] BUILDING REASONING KB")
    print("=" * 60)

    knowledge_data = [

        {"element": "Carbon", "goal": "Strength", "effect": "Principal hardening element. Increases tensile strength up to 0.85 wt%.", "tradeoff": "Decreases ductility and weldability."},

        {"element": "Carbon", "goal": "Hardness", "effect": "Increases hardness significantly via martensite formation.", "tradeoff": "Increases brittleness."},

        {"element": "Manganese", "goal": "Strength", "effect": "Increases strength and hardness. Strong pearlite promoter.", "tradeoff": "Decreases ductility and weldability (less than Carbon)."},

        {"element": "Manganese", "goal": "Hardenability", "effect": "Strongly increases hardenability. Prevents FeS formation by forming MnS.", "tradeoff": "Can attack refractories at high levels."},

        {"element": "Chromium", "goal": "Corrosion Resistance", "effect": "Increases resistance to corrosion and oxidation. Essential for stainless steels.", "tradeoff": "Strong carbide former; requires longer heating time."},

        {"element": "Chromium", "goal": "Hardenability", "effect": "Increases hardenability and high-temperature strength.", "tradeoff": "Can increase hardness but might reduce toughness if not balanced."},

        {"element": "Nickel", "goal": "Toughness", "effect": "Strengthens and toughens the ferrite phase. Does not form carbides.", "tradeoff": "High cost element."},

        {"element": "Nickel", "goal": "Fatigue Resistance", "effect": "Improves fatigue resistance and low-temperature toughness.", "tradeoff": "Austenite stabilizer; might lead to retained austenite if too high."},

        {"element": "Molybdenum", "goal": "High-Temp Strength", "effect": "Enhances creep strength and high-temperature stability.", "tradeoff": "Increases cost."},

        {"element": "Molybdenum", "goal": "Hardenability", "effect": "Greatly increases hardenability. Minimizes temper embrittlement.", "tradeoff": "Often used in conjunction with Cr."},

        {"element": "Niobium", "goal": "Yield Strength", "effect": "Small additions (0.02%) increase yield strength by 70-100 MPa via grain refinement.", "tradeoff": "Impaired notch toughness without grain size control."},

        {"element": "Aluminum", "goal": "Grain Size Control", "effect": "Most effective element for controlling austenite grain growth.", "tradeoff": "Can affect hardenability if too high."},

        {"element": "Copper", "goal": "Atmospheric Corrosion", "effect": "Increases atmospheric corrosion resistance (weathering steels) at levels > 0.20%.", "tradeoff": "Detrimental to hot-working and surface quality."},

        {"element": "Boron", "goal": "Hardenability", "effect": "Powerful hardenability agent in low-carbon fully killed steels (0.0005-0.003%).", "tradeoff": "Only effective in very small amounts; ineffective if nitrogen not controlled."},

        {"element": "Silicon", "goal": "Deoxidation", "effect": "Principal deoxidizer. Increases as-rolled strength.", "tradeoff": "Detrimental to surface quality in low-carbon steels."},

        {"element": "Silicon", "goal": "Corrosion Resistance", "effect": "Very high levels (14-17%) provide resistance to corrosive acids.", "tradeoff": "Makes iron virtually unmachinable and brittle."}
    ]

    texts = []
    metadata = []

    for idx, entry in enumerate(
        knowledge_data
    ):

        rule_id = f"Rule_{idx}"

        desc = (
            f"[{entry['element']} | "
            f"goal: {entry['goal']} | "
            f"effect: {entry['effect']} | "
            f"tradeoff: {entry['tradeoff']}]"
        )

        texts.append(desc)

        metadata.append(desc)

    save_embeddings(
        texts,
        metadata,
        "metallurgical_reasoning.index",
        "metallurgical_reasoning_metadata.json"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("🚀 INITIALIZING META-SCIENTIFIC RAG PIPELINE")
    print("=" * 70)

    build_alloy_kb()

    build_application_kb()

    build_reasoning_kb()

    print("\n" + "=" * 70)
    print("🎯 ALL KNOWLEDGE BASES GENERATED SUCCESSFULLY")
    print("=" * 70)