import json
import os

import config


def write_section_header(f, title, source):
    f.write("=" * 100 + "\n")
    f.write(title + "\n")
    f.write(f"Source: {source}\n")
    f.write("=" * 100 + "\n\n")


def inspect_metadata_file(file_path, section_title, source_name, f):

    try:
        with open(file_path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)

        write_section_header(
            f,
            section_title,
            source_name
        )

        f.write(f"FILE: {file_path}\n")
        f.write(f"TOTAL ENTRIES: {len(data)}\n\n")

        for i, entry in enumerate(data):

            f.write(f"[ENTRY {i}]\n")
            f.write("-" * 100 + "\n")

            if isinstance(entry, str):
                f.write(entry)
            else:
                f.write(json.dumps(entry, indent=4, ensure_ascii=False))

            f.write("\n")
            f.write("-" * 100 + "\n\n")

    except FileNotFoundError:
        f.write(f"❌ FILE NOT FOUND:\n{file_path}\n\n")

    except Exception as e:
        f.write(f"❌ ERROR reading {file_path}\n")
        f.write(f"{str(e)}\n\n")


def show_all_embeddings():

    print("🧠 Peeking into the Meta-Scientific Agent Brain...")

    output_file = "full_brain_peek.txt"

    # ============================================================
    # METADATA FILE LOCATIONS
    # ============================================================

    academic_aisi_path = os.path.join(
        config.EMBEDDINGS_DIR,
        "academic_aisi_metadata.json"
    )

    industrial_apps_path = os.path.join(
        config.EMBEDDINGS_DIR,
        "industrial_applications_metadata.json"
    )

    reasoning_path = os.path.join(
        config.EMBEDDINGS_DIR,
        "metallurgical_reasoning_metadata.json"
    )

    # ============================================================
    # OUTPUT
    # ============================================================

    with open(output_file, 'w', encoding='utf-8') as f:

        f.write("\n")
        f.write("#" * 100 + "\n")
        f.write("META-SCIENTIFIC AGENT : FULL KNOWLEDGE BASE INSPECTION\n")
        f.write("#" * 100 + "\n\n")

        # ========================================================
        # SECTION 1 — ACADEMIC ALLOYS
        # ========================================================

        inspect_metadata_file(
            academic_aisi_path,
            "SECTION 1: EXPERIMENTAL / ACADEMIC ALLOY EMBEDDINGS",
            "Ward et al. Steel Strength Dataset",
            f
        )

        f.write("\n\n")

        # ========================================================
        # SECTION 2 — INDUSTRIAL APPLICATIONS
        # ========================================================

        inspect_metadata_file(
            industrial_apps_path,
            "SECTION 2: INDUSTRIAL APPLICATION-AWARE EMBEDDINGS",
            "Curated Industrial Alloy Dataset",
            f
        )

        f.write("\n\n")

        # ========================================================
        # SECTION 3 — REASONING RULES
        # ========================================================

        inspect_metadata_file(
            reasoning_path,
            "SECTION 3: METALLURGICAL REASONING RULES",
            "ASM Metals Handbook Vol 1",
            f
        )

        f.write("\n\n")

        f.write("#" * 100 + "\n")
        f.write("END OF KNOWLEDGE BASE INSPECTION\n")
        f.write("#" * 100 + "\n")

    print(f"✅ Done! Open '{output_file}' to inspect the complete agent knowledge base.")


if __name__ == "__main__":
    show_all_embeddings()