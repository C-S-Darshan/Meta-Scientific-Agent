import re
import json

# --- REALISTIC METALLURGICAL CONSTANTS ---
# UTS = 450 (Base Iron) + (Solute % * Coefficient)
ELEMENT_DATA = {
    "Fe": {"density": 7.87, "pren_coeff": 0.0,  "uts_gain": 0},    # Matrix Base
    "Cr": {"density": 7.19, "pren_coeff": 1.0,  "uts_gain": 15},   # Corrosion/Strength
    "Ni": {"density": 8.90, "pren_coeff": 0.0,  "uts_gain": 12},   # Toughness
    "Mo": {"density": 10.28,"pren_coeff": 3.3,  "uts_gain": 30},   # Pitting
    "Mn": {"density": 7.21, "pren_coeff": 0.0,  "uts_gain": 20},   # General Strength
    "Si": {"density": 2.33, "pren_coeff": 0.0,  "uts_gain": 10},    
    "Co": {"density": 8.90, "pren_coeff": 0.0,  "uts_gain": 18},   
    "Al": {"density": 2.70, "pren_coeff": 0.0,  "uts_gain": 5},   
    "Ti": {"density": 4.50, "pren_coeff": 0.0,  "uts_gain": 50},   
    "V":  {"density": 6.11, "pren_coeff": 0.0,  "uts_gain": 55},
    "C":  {"density": 2.26, "pren_coeff": 0.0,  "uts_gain": 200},  # Small amount = Big jump
    "Cu": {"density": 8.96, "pren_coeff": 0.0,  "uts_gain": 15},   # Conductivity/Strength
    "Nb": {"density": 8.57, "pren_coeff": 0.0,  "uts_gain": 45},   # Carbide former
    "W":  {"density": 19.25,"pren_coeff": 0.0,  "uts_gain": 40},   # Refractory strength
    "Zr": {"density": 6.52, "pren_coeff": 0.0,  "uts_gain": 40},   # Grain refinement
    "Ta": {"density": 16.65,"pren_coeff": 0.0,  "uts_gain": 35},   # Refractory strength
    "Hf": {"density": 13.31,"pren_coeff": 0.0,  "uts_gain": 45},   # Refractory strength
    "N":  {"density": 1.25, "pren_coeff": 16.0, "uts_gain": 60},   # Interstitial strengthener; PREN=16*%N (standard Sedriks formula)
    "P":  {"density": 1.82, "pren_coeff": 0.0,  "uts_gain": 5},    # Residual/tramp; minor strengthening, detrimental to ductility
    "S":  {"density": 2.07, "pren_coeff": 0.0,  "uts_gain": -5},   # Tramp element; MnS inclusions reduce toughness (negative contribution)
    "B":  {"density": 2.46, "pren_coeff": 0.0,  "uts_gain": 50},   # Grain-boundary strengthener; effective at 0.001-0.003 wt%
    # --- ADDITIONAL MPEA ELEMENTS (minor additions for near-complete HEA coverage) ---
    "Zn": {"density": 7.13, "pren_coeff": 0.0,  "uts_gain": 5},    # Minor solid-solution; rarely >5% in HEAs
    "Sn": {"density": 7.31, "pren_coeff": 0.0,  "uts_gain": 10},   # Tramp in steels; solid-solution in Cu-based MPEAs
    "Mg": {"density": 1.74, "pren_coeff": 0.0,  "uts_gain": 10},   # Grain refinement; Mg-Al-based lightweight HEAs
    "Re": {"density": 21.0, "pren_coeff": 0.0,  "uts_gain": 40},   # Refractory; Ni-superalloy Re for creep resistance
}

HEAT_TREATMENT_EFFECTS = {
    "annealed": {
        "strength_factor_range": (0.80, 0.95),
        "ductility_factor_range": (1.10, 1.35)
    },
    "normalized": {
        "strength_factor_range": (0.95, 1.10),
        "ductility_factor_range": (0.90, 1.10)
    },
    "quenched_tempered": {
        "strength_factor_range": (1.20, 1.60),
        "ductility_factor_range": (0.55, 0.85)
    },
    "age_hardened": {
        "strength_factor_range": (1.15, 1.50),
        "ductility_factor_range": (0.60, 0.90)
    }
}

def parse_composition(text: str):
    """Cleans brackets, percents, and ignores unknown symbols like PTFE."""
    # Step 1: Remove all clutter characters
    clean_text = re.sub(r'[\[\]% \(\)]', '', text)
    # Step 2: Find Element-Number pairs (e.g., Fe-70 or Fe70)
    matches = re.findall(r'([A-Z][a-z]?)-?([0-9]*\.?[0-9]+)', clean_text)
    
    comp = {}
    total_val = 0.0
    for el, val in matches:
        if el in ELEMENT_DATA: # Ignore PTFE or unknown elements
            v = float(val)
            comp[el] = v
            total_val += v
    
    if total_val == 0: return None
    # Normalize to 1.0 (Fixes AI math errors)
    return {el: v / total_val for el, v in comp.items()}

def is_plausible_composition(comp: dict) -> bool:
    """Framework scope is restricted to Fe-based industrial steel alloys
    (see Experimental Evaluation). A composition below 30% Fe is out of
    scope and should be rejected rather than passed to calculate_physics(),
    whose UTS formula is only calibrated for Fe-majority steel chemistries
    and produces physically impossible estimates (>10 GPa) for degenerate,
    non-Fe-based compositions."""
    return comp.get("Fe", 0.0) >= 0.3


def calculate_physics(comp: dict, heat_treatment: str = "annealed"):
    density = 0.0
    pren = 0.0
    uts = 450.0 # Base annealed Iron UTS
    
    for el, fraction in comp.items():
        data = ELEMENT_DATA[el]
        pct = fraction * 100
        density += data["density"] * fraction
        pren += (data["pren_coeff"] * pct)
        uts += (data["uts_gain"] * pct)

    # Realistic Elongation trade-off
    base_elong = max(2.0, 45.0 - (uts / 65.0))
    
    # Apply Heat Treatment Effects (using average of range)
    ht_data = HEAT_TREATMENT_EFFECTS.get(heat_treatment, HEAT_TREATMENT_EFFECTS["annealed"])
    s_min, s_max = ht_data["strength_factor_range"]
    d_min, d_max = ht_data["ductility_factor_range"]
    
    final_uts = uts * ((s_min + s_max) / 2.0)
    final_elong = base_elong * ((d_min + d_max) / 2.0)
    
    return {
        "Estimated_UTS_MPa": round(final_uts, 1),
        "PREN_Corrosion_Index": round(pren, 2),
        "Estimated_Elongation_pct": round(final_elong, 1),
        "Theoretical_Density_g_cm3": round(density, 2),
        "Heat_Treatment": heat_treatment,
        # Store raw (pre-heat-treatment) values for recommendation engine
        "_raw_uts": round(uts, 1),
        "_raw_elongation": round(base_elong, 1),
    }


def recommend_heat_treatment(comp: dict, targets: dict = None):
    """
    Iterate through all heat treatments to find the optimal one
    that meets the industry targets for UTS and Elongation.

    Args:
        comp:    Normalised composition dict (fractions summing to ~1.0)
        targets: Dict with optional keys 'UTS', 'Elongation', 'PREN'

    Returns:
        Dict with:
          - recommended_treatment: str
          - all_treatments: list of dicts (one per treatment with pass/fail)
          - meets_target: bool (True if at least one treatment passes all)
    """
    if targets is None:
        targets = {}

    # Calculate base (raw) properties once
    base = calculate_physics(comp, heat_treatment="annealed")
    raw_uts = base["_raw_uts"]
    raw_elong = base["_raw_elongation"]
    pren = base["PREN_Corrosion_Index"]
    density = base["Theoretical_Density_g_cm3"]

    target_uts = targets.get("UTS")
    target_elong = targets.get("Elongation")
    target_pren = targets.get("PREN")

    results = []

    for ht_name, ht_data in HEAT_TREATMENT_EFFECTS.items():
        s_min, s_max = ht_data["strength_factor_range"]
        d_min, d_max = ht_data["ductility_factor_range"]

        ht_uts = round(raw_uts * ((s_min + s_max) / 2.0), 1)
        ht_elong = round(raw_elong * ((d_min + d_max) / 2.0), 1)

        verdict = {}
        all_pass = True

        if target_uts is not None:
            passed = ht_uts >= target_uts
            verdict["UTS"] = {"value": ht_uts, "target": target_uts,
                              "status": "PASS" if passed else "FAIL",
                              "gap": round(ht_uts - target_uts, 1)}
            if not passed:
                all_pass = False

        if target_elong is not None:
            passed = ht_elong >= target_elong
            verdict["Elongation"] = {"value": ht_elong, "target": target_elong,
                                     "status": "PASS" if passed else "FAIL",
                                     "gap": round(ht_elong - target_elong, 1)}
            if not passed:
                all_pass = False

        if target_pren is not None:
            passed = pren >= target_pren
            verdict["PREN"] = {"value": pren, "target": target_pren,
                               "status": "PASS" if passed else "FAIL (compositional — not affected by heat treatment)",
                               "gap": round(pren - target_pren, 2)}
            # NOTE: PREN is purely compositional (Cr + 3.3*Mo + 16*N).
            # Heat treatment CANNOT change it, so do NOT count it toward all_pass.

        results.append({
            "treatment": ht_name,
            "UTS_MPa": ht_uts,
            "Elongation_pct": ht_elong,
            "PREN": pren,
            "all_pass": all_pass,
            "verdict": verdict,
        })

    # Pick the best treatment: prefer one that passes everything,
    # break ties by highest UTS (strength is usually the priority)
    passing = [r for r in results if r["all_pass"]]
    if passing:
        best = max(passing, key=lambda r: r["UTS_MPa"])
    else:
        # None pass — pick the one with fewest failures
        best = min(results, key=lambda r: sum(1 for v in r["verdict"].values() if v["status"] == "FAIL"))

    return {
        "recommended_treatment": best["treatment"],
        "meets_all_targets": best["all_pass"],
        "recommended_UTS_MPa": best["UTS_MPa"],
        "recommended_Elongation_pct": best["Elongation_pct"],
        "PREN": pren,
        "Density": density,
        "all_treatments": results,
    }

def run_physics_tool(hypothesis_text: str, user_goal: str = "") -> str:
    comp = parse_composition(hypothesis_text)
    if not comp:
        return json.dumps({"error": "Format Error: Use Element-Percentage (e.g., Fe-70, Cr-18)."})

    # Constraint Check
    if comp.get("Fe", 0) < 0.3:
        return json.dumps({"error": "Constraint Failure: Alloy must be Iron-based (>30% Fe)."})

    res = calculate_physics(comp)
    res["Input_Parsed"] = {k: f"{round(v*100, 2)}%" for k, v in comp.items()}

    # Check against targets
    goal_l = user_goal.lower()
    fails = []
    if "corrosion" in goal_l and res["PREN_Corrosion_Index"] < 20:
        fails.append(f"Low PREN ({res['PREN_Corrosion_Index']})")
    if "strong" in goal_l and res["Estimated_UTS_MPa"] < 1200:
        fails.append(f"Low UTS ({res['Estimated_UTS_MPa']} MPa)")

    res["Validation_Note"] = "FAIL: " + ", ".join(fails) if fails else "PASS"
    return json.dumps(res, indent=2)

def generate_phase_data(comp: dict, tdb_path: str):
    """
    Generates phase fraction data vs temperature for the given composition using pycalphad.
    Returns a dictionary of phase names to (temperatures, phase_fractions) tuples.
    """
    try:
        from pycalphad import Database, equilibrium, variables as v
        import numpy as np
        import warnings
        
        # Suppress PyCalphad TDB parsing warnings (like missing TYPE_DEFINITION) to keep console clean
        warnings.filterwarnings('ignore', module='pycalphad.io.tdb')
        warnings.filterwarnings('ignore', category=UserWarning)
        
    except ImportError:
        print("[-] pycalphad not installed. Phase diagram generation skipped.")
        return None

    try:
        dbf = Database(tdb_path)
        
        # Get elements present in the database
        dbf_elements = set(list(dbf.elements))
        
        # Filter AI composition to ONLY include elements that exist in this TDB
        comp_upper = {el.upper(): val for el, val in comp.items() if val > 0.001 and el.upper() in dbf_elements}
        
        if not comp_upper:
            print("[-] None of the proposed elements exist in the thermodynamic database.")
            return None
            
        comps = list(comp_upper.keys()) + ['VA']
        phases = list(dbf.phases.keys())

        dependent_comp = max(comp_upper, key=comp_upper.get)
        
        # Normalize the remaining fractions to 1.0 so pycalphad doesn't complain
        total_remaining = sum(comp_upper.values())
        comp_upper = {el: val/total_remaining for el, val in comp_upper.items()}
        
        conds = {v.N: 1, v.P: 101325, v.T: (700, 1800, 100)}
        for el, val in comp_upper.items():
            if el != dependent_comp:
                conds[v.X(el)] = val

        print(f"[*] PyCalphad: Calculating equilibrium for {list(comp_upper.keys())}...")
        eq = equilibrium(dbf, comps, phases, conds, output=['HM', 'CPM'])
        
        thermo_data = {}
        temperatures = np.atleast_1d(eq.T.values)
        thermo_data['temperatures'] = temperatures.tolist()

        phase_fractions = {}
        phases_present = np.unique(eq.Phase.values)
        
        for phase in phases_present:
            if not phase or phase == '' or phase == 'None': continue
            np_vals = eq.NP.where(eq.Phase == phase).sum(dim='vertex').squeeze().values
            if np.nanmax(np_vals) > 1e-4:
                phase_fractions[str(phase)] = np_vals.tolist()
        
        thermo_data['phase_fractions'] = phase_fractions
        
        # Molar properties
        thermo_data['cpm'] = eq.CPM.values.squeeze().tolist()
        thermo_data['hm'] = eq.HM.values.squeeze().tolist()

        # Chemical potentials
        mu_data = {}
        components = eq.MU.coords['component'].values
        for comp_name in components:
            if comp_name != 'VA':
                mu_vals = eq.MU.sel(component=comp_name).values.squeeze()
                mu_data[str(comp_name)] = mu_vals.tolist()
        thermo_data['mu'] = mu_data

        return thermo_data
    except Exception as e:
        print(f"[-] PyCalphad Error: {e}")
        return None


def extract_thermo_insights(thermo_data, comp):
    """
    Mines raw pycalphad equilibrium data for actionable metallurgical insights.
    Returns a structured dict with solidus/liquidus, phase stability, transitions, etc.
    """
    if not thermo_data:
        return None

    try:
        import numpy as np

        temps = np.array(thermo_data['temperatures'])
        phase_fracs = thermo_data.get('phase_fractions', {})
        insights = {}

        # ── 1. Solidus & Liquidus Temperature ────────────────────────────────
        liquid_key = None
        for phase_name in phase_fracs:
            if 'LIQUID' in phase_name.upper():
                liquid_key = phase_name
                break

        if liquid_key:
            liquid_fracs = np.array(phase_fracs[liquid_key])
            # Solidus: first temperature where liquid appears (fraction > 0.01)
            liquid_present = liquid_fracs > 0.01
            if np.any(liquid_present):
                solidus_idx = np.argmax(liquid_present)
                insights['Solidus_Temperature_K'] = round(float(temps[solidus_idx]), 0)
                insights['Solidus_Temperature_C'] = round(float(temps[solidus_idx]) - 273.15, 0)

            # Liquidus: temperature where liquid fraction > 0.99
            fully_liquid = liquid_fracs > 0.99
            if np.any(fully_liquid):
                liquidus_idx = np.argmax(fully_liquid)
                insights['Liquidus_Temperature_K'] = round(float(temps[liquidus_idx]), 0)
                insights['Liquidus_Temperature_C'] = round(float(temps[liquidus_idx]) - 273.15, 0)

            # Weldability: solidus-liquidus range (ΔT)
            if 'Solidus_Temperature_K' in insights and 'Liquidus_Temperature_K' in insights:
                delta_t = insights['Liquidus_Temperature_K'] - insights['Solidus_Temperature_K']
                insights['Freezing_Range_K'] = round(delta_t, 0)
                if delta_t < 50:
                    insights['Weldability'] = 'Excellent (narrow freezing range — low hot-cracking risk)'
                elif delta_t < 150:
                    insights['Weldability'] = 'Good (moderate freezing range)'
                elif delta_t < 300:
                    insights['Weldability'] = 'Fair (wide freezing range — some hot-cracking risk)'
                else:
                    insights['Weldability'] = 'Poor (very wide freezing range — high hot-cracking risk)'

        # ── 2. Phase Stability at Room Temperature (~300K) ───────────────────
        # Find the closest temperature to 300K (or lowest available)
        room_temp_idx = np.argmin(np.abs(temps - 700))  # Use lowest temp in range
        room_temp = temps[room_temp_idx]
        stable_at_room = {}
        for phase_name, fracs in phase_fracs.items():
            frac_arr = np.array(fracs)
            if len(frac_arr) > room_temp_idx and frac_arr[room_temp_idx] > 0.01:
                stable_at_room[phase_name] = round(float(frac_arr[room_temp_idx]), 3)

        if stable_at_room:
            insights['Stable_Phases_At_Service_Temp'] = stable_at_room
            insights['Service_Temp_Reference_K'] = round(float(room_temp), 0)

        # Flag potentially harmful phases
        harmful_phases = []
        for phase_name in stable_at_room:
            pn_upper = phase_name.upper()
            if 'SIGMA' in pn_upper:
                harmful_phases.append(f'{phase_name} (embrittlement risk)')
            elif 'LAVES' in pn_upper:
                harmful_phases.append(f'{phase_name} (reduces ductility)')
            elif 'CEMENTITE' in pn_upper or 'M23C6' in pn_upper or 'M7C3' in pn_upper:
                frac = stable_at_room[phase_name]
                if frac > 0.05:
                    harmful_phases.append(f'{phase_name} at {frac*100:.1f}% (excessive carbide precipitation)')
        if harmful_phases:
            insights['Harmful_Phase_Warnings'] = harmful_phases

        # ── 3. BCC↔FCC Transition Temperature (Ae3 equivalent) ──────────────
        bcc_key = None
        fcc_key = None
        for phase_name in phase_fracs:
            pn = phase_name.upper()
            if 'BCC' in pn and bcc_key is None:
                bcc_key = phase_name
            elif 'FCC' in pn and fcc_key is None:
                fcc_key = phase_name

        if bcc_key and fcc_key:
            bcc_fracs = np.array(phase_fracs[bcc_key])
            fcc_fracs = np.array(phase_fracs[fcc_key])
            # Find where BCC drops below FCC (heating: ferrite → austenite)
            diff = bcc_fracs - fcc_fracs
            sign_changes = np.where(np.diff(np.sign(diff)))[0]
            if len(sign_changes) > 0:
                # Take the first crossover
                idx = sign_changes[0]
                # Linear interpolation for more accurate temperature
                if idx + 1 < len(temps):
                    t1, t2 = temps[idx], temps[idx + 1]
                    d1, d2 = diff[idx], diff[idx + 1]
                    if d2 != d1:
                        transition_t = t1 + (t2 - t1) * (-d1 / (d2 - d1))
                    else:
                        transition_t = (t1 + t2) / 2
                    insights['BCC_FCC_Transition_K'] = round(float(transition_t), 0)
                    insights['BCC_FCC_Transition_C'] = round(float(transition_t) - 273.15, 0)

        # ── 4. Maximum Service Temperature ───────────────────────────────────
        # Find highest temperature where dominant room-temp phases remain stable
        if stable_at_room:
            dominant_phase = max(stable_at_room, key=stable_at_room.get)
            dominant_fracs = np.array(phase_fracs[dominant_phase])
            # Phase is "stable" as long as fraction > 0.10
            stable_mask = dominant_fracs > 0.10
            if np.any(~stable_mask):
                # Find where it drops below threshold
                unstable_idx = np.argmax(~stable_mask)
                if unstable_idx > 0:
                    max_service_t = temps[unstable_idx - 1]
                    insights['Max_Service_Temperature_K'] = round(float(max_service_t), 0)
                    insights['Max_Service_Temperature_C'] = round(float(max_service_t) - 273.15, 0)
                    insights['Max_Service_Dominant_Phase'] = dominant_phase

        # ── 5. Summary text for the analyst LLM ─────────────────────────────
        summary_parts = []
        if 'Solidus_Temperature_C' in insights:
            summary_parts.append(f"Solidus: {insights['Solidus_Temperature_C']:.0f}°C")
        if 'Liquidus_Temperature_C' in insights:
            summary_parts.append(f"Liquidus: {insights['Liquidus_Temperature_C']:.0f}°C")
        if 'Weldability' in insights:
            summary_parts.append(f"Weldability: {insights['Weldability']}")
        if 'BCC_FCC_Transition_C' in insights:
            summary_parts.append(f"BCC→FCC transition at {insights['BCC_FCC_Transition_C']:.0f}°C")
        if stable_at_room:
            phase_list = ', '.join(f"{k} ({v*100:.1f}%)" for k, v in stable_at_room.items())
            summary_parts.append(f"Stable phases at service temp: {phase_list}")
        if harmful_phases:
            summary_parts.append(f"[WARNING] Harmful phases: {'; '.join(harmful_phases)}")

        insights['_analyst_summary'] = '. '.join(summary_parts) if summary_parts else 'No actionable thermo insights available.'

        return insights

    except Exception as e:
        print(f"[-] Thermo Insights Error: {e}")
        return None