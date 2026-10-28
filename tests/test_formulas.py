"""
Unit tests for the deterministic symbolic engine (src/formulas.py).

Covers the composition parser, the Fe>=30% scope-plausibility gate, and the
physics/property calculator -- the three pure functions that sit between the
LLM's free-text hypothesis output and the reported alloy properties.

Run with:  pytest tests/test_formulas.py -v
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import formulas  # noqa: E402


# ---------------------------------------------------------------------------
# parse_composition
# ---------------------------------------------------------------------------

def test_parse_composition_well_formed():
    comp = formulas.parse_composition("Fe-70 Cr-30")
    assert comp is not None
    assert comp["Fe"] == 0.7
    assert comp["Cr"] == 0.3


def test_parse_composition_empty_string_returns_none():
    assert formulas.parse_composition("") is None


def test_parse_composition_no_recognizable_elements_returns_none():
    assert formulas.parse_composition("no valid stoichiometry here") is None


def test_parse_composition_single_element():
    comp = formulas.parse_composition("Fe-100")
    assert comp == {"Fe": 1.0}


def test_parse_composition_normalizes_when_percentages_dont_sum_to_100():
    # AI generations don't always sum to exactly 100; the parser must
    # renormalize to fractions of 1.0 rather than reject or silently
    # under-count.
    comp = formulas.parse_composition("Fe-50 Cr-25")
    assert comp["Fe"] == 2.0 / 3.0
    assert comp["Cr"] == 1.0 / 3.0
    assert abs(sum(comp.values()) - 1.0) < 1e-9


def test_parse_composition_ignores_unknown_elements():
    # "Xx" is not a known alloying element (e.g. a hallucinated symbol or a
    # non-metallurgical token like PTFE) and must be dropped rather than
    # crash or silently corrupt the normalization of the known elements.
    comp = formulas.parse_composition("Fe-70 Xx-30")
    assert comp == {"Fe": 1.0}


# ---------------------------------------------------------------------------
# is_plausible_composition (Fe >= 30% scope gate)
# ---------------------------------------------------------------------------

def test_is_plausible_pure_iron():
    assert formulas.is_plausible_composition({"Fe": 1.0}) is True


def test_is_plausible_at_exact_boundary():
    assert formulas.is_plausible_composition({"Fe": 0.30}) is True


def test_is_plausible_just_below_boundary():
    assert formulas.is_plausible_composition({"Fe": 0.29}) is False


def test_is_plausible_zero_iron():
    assert formulas.is_plausible_composition({"Fe": 0.0}) is False


def test_is_plausible_missing_fe_key():
    # A degenerate generation that never mentions Fe at all must fail the
    # gate rather than raise a KeyError.
    assert formulas.is_plausible_composition({"Cr": 0.6, "Ni": 0.4}) is False


# ---------------------------------------------------------------------------
# calculate_physics
# ---------------------------------------------------------------------------

def test_calculate_physics_pure_iron_baseline():
    props = formulas.calculate_physics({"Fe": 1.0}, heat_treatment="annealed")
    # Pure iron has no alloying contribution: UTS reduces to the 450 MPa
    # base scaled only by the annealed strength factor, PREN is 0.
    assert props["PREN_Corrosion_Index"] == 0.0
    assert props["Estimated_UTS_MPa"] == round(450.0 * 0.875, 1)
    assert props["Theoretical_Density_g_cm3"] == round(7.87, 2)
    assert props["Heat_Treatment"] == "annealed"


def test_calculate_physics_density_is_composition_weighted_average():
    props = formulas.calculate_physics({"Fe": 0.5, "Ni": 0.5}, heat_treatment="annealed")
    expected_density = round(7.87 * 0.5 + 8.90 * 0.5, 2)
    assert props["Theoretical_Density_g_cm3"] == expected_density


def test_calculate_physics_carbon_raises_uts_more_than_equal_mass_fraction_of_iron():
    # C has by far the highest uts_gain coefficient (200 vs Fe's 0) -- a
    # small carbon addition should raise UTS well above the pure-Fe baseline.
    baseline = formulas.calculate_physics({"Fe": 1.0}, heat_treatment="annealed")
    with_carbon = formulas.calculate_physics({"Fe": 0.98, "C": 0.02}, heat_treatment="annealed")
    assert with_carbon["Estimated_UTS_MPa"] > baseline["Estimated_UTS_MPa"]


def test_calculate_physics_quenched_tempered_increases_strength_reduces_ductility():
    comp = {"Fe": 0.8, "Cr": 0.2}
    annealed = formulas.calculate_physics(comp, heat_treatment="annealed")
    qt = formulas.calculate_physics(comp, heat_treatment="quenched_tempered")
    assert qt["Estimated_UTS_MPa"] > annealed["Estimated_UTS_MPa"]
    assert qt["Estimated_Elongation_pct"] < annealed["Estimated_Elongation_pct"]


def test_calculate_physics_unknown_heat_treatment_falls_back_to_annealed():
    comp = {"Fe": 1.0}
    unknown = formulas.calculate_physics(comp, heat_treatment="not_a_real_treatment")
    annealed = formulas.calculate_physics(comp, heat_treatment="annealed")
    assert unknown["Estimated_UTS_MPa"] == annealed["Estimated_UTS_MPa"]
    assert unknown["Heat_Treatment"] == "not_a_real_treatment"


def test_calculate_physics_elongation_never_below_floor():
    # base_elong = max(2.0, 45 - uts/65); an extremely high-UTS composition
    # must still floor out at 2.0, not go negative.
    high_uts_comp = {"Fe": 0.5, "C": 0.5}
    props = formulas.calculate_physics(high_uts_comp, heat_treatment="quenched_tempered")
    assert props["_raw_elongation"] >= 2.0
