def compute_corrosion_metrics(composition: dict, pren_predicted: float) -> dict:
    """
    Compute PREN and corrosion classification from composition fracs.

    PREN = %Cr + 3.3×%Mo + 16×%N  (in weight percent, not mole fraction)
    Multiply mole fracs by atomic weights to convert to approximate wt%:
      Cr: 52, Mo: 96, N: 14
    """
    cr_wt = composition.get("Cr", 0) * 52 * 100   # approx wt%
    mo_wt = composition.get("Mo", 0) * 96 * 100
    n_wt  = composition.get("N",  0) * 14 * 100

    pren_calc = cr_wt + 3.3 * mo_wt + 16 * n_wt

    # PREN classification (industry standard thresholds)
    if pren_calc >= 40:
        grade = "Super duplex / highly corrosion resistant"
        nace = "Suitable for sour service (NACE MR0175 compliant region)"
    elif pren_calc >= 25:
        grade = "Austenitic stainless (316L class)"
        nace = "Suitable for moderate chloride environments"
    elif pren_calc >= 18:
        grade = "Standard stainless (304 class)"
        nace = "Limited chloride resistance — avoid seawater exposure"
    elif pren_calc >= 10:
        grade = "Low-alloy steel"
        nace = "Surface coating required for corrosive environments"
    else:
        grade = "Plain carbon steel — no passive layer"
        nace = "Not suitable for corrosive service without protective coating"

    return {
        "pren_calculated": round(pren_calc, 2),
        "pren_model_predicted": round(pren_predicted, 2),
        "corrosion_grade": grade,
        "nace_guidance": nace,
        "cr_wt_pct": round(cr_wt, 2),
        "mo_wt_pct": round(mo_wt, 2),
    }
