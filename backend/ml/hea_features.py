"""
Physics-informed thermodynamic descriptors for High-Entropy Alloys.
These four features capture the key HEA design criteria:
  - ΔSmix: configurational entropy (>1.5R = HEA regime)
  - δ: atomic size mismatch (drives solid-solution strengthening)
  - ΔHmix: mixing enthalpy (from Miedema's model)
  - VEC: valence electron concentration (predicts FCC vs BCC phase)
"""
import numpy as np
from pymatgen.core.composition import Composition

# Miedema's model pairwise interaction parameters H_AB (kJ/mol)
# Source: Takeuchi & Inoue, 2005 — Materials Transactions
MIEDEMA_H = {
    frozenset({"Fe","Ni"}): -2.0, frozenset({"Fe","Cr"}): -1.0,
    frozenset({"Fe","Co"}): -1.0, frozenset({"Ni","Cr"}): -7.0,
    frozenset({"Ni","Co"}): 0.0,  frozenset({"Co","Cr"}): -4.0,
    frozenset({"Al","Fe"}): -11.0,frozenset({"Al","Ni"}): -22.0,
    frozenset({"Al","Cr"}): -10.0,frozenset({"Al","Co"}): -19.0,
    frozenset({"Mo","Fe"}): -2.0, frozenset({"Mo","Ni"}): -7.0,
    frozenset({"Ti","Fe"}): -17.0,frozenset({"Ti","Ni"}): -35.0,
    frozenset({"Ti","Al"}): -30.0,frozenset({"Nb","Fe"}): -16.0,
    frozenset({"Ta","Fe"}): -15.0,frozenset({"W","Fe"}): -0.0,
}

# d-electron VEC per element (for BCC/FCC stability prediction)
VEC_D = {
    "Fe":8,"Co":9,"Ni":10,"Cu":11,"Cr":6,"Mn":7,"V":5,"Ti":4,
    "Al":3,"Si":4,"Mo":6,"W":6,"Nb":5,"Ta":5,"Hf":4,"Zr":4,
    "C":4,"N":5,"Sc":3,
}

# Goldschmidt atomic radii (pm)
ATOMIC_RADII = {
    "Fe":126,"Co":125,"Ni":124,"Cu":128,"Cr":128,"Mn":127,"V":135,
    "Al":143,"Ti":147,"Mo":139,"W":141,"Nb":146,"Ta":146,"Hf":158,
    "Zr":160,"Si":117,"C":77,"N":75,"Zn":137,"Mg":160,"Sc":162,
}

def compute_hea_features(comp: Composition) -> dict:
    """
    Compute all four HEA thermodynamic descriptors from a pymatgen Composition.

    Returns dict with keys:
        feat_hea_mixing_entropy   — ΔSmix (J/mol/K), threshold 1.5R = HEA
        feat_hea_mixing_enthalpy  — ΔHmix (kJ/mol), from Miedema model
        feat_hea_atomic_mismatch  — δ (%), atomic radius mismatch
        feat_hea_vec              — VEC, valence electron concentration
        feat_hea_n_elements       — number of principal elements
        feat_hea_omega            — Ω = Tmelt × ΔSmix / |ΔHmix| (phase stability criterion)
    """
    R = 8.314   # J/(mol·K)
    elements = [str(el) for el in comp.elements]
    fracs = {el: float(comp.get_atomic_fraction(el)) for el in elements}

    # 1. Configurational mixing entropy
    ds_mix = -R * sum(x * np.log(x) for x in fracs.values() if x > 1e-10)

    # 2. Mixing enthalpy (Miedema pairwise)
    dh_mix = 0.0
    for i, el_i in enumerate(elements):
        for el_j in elements[i+1:]:
            key = frozenset({el_i, el_j})
            h_ij = MIEDEMA_H.get(key, 0.0)   # 0 if pair not tabulated
            dh_mix += 4 * h_ij * fracs[el_i] * fracs[el_j]

    # 3. Atomic size mismatch δ
    r_bar = sum(fracs[el] * ATOMIC_RADII.get(el, 130) for el in elements)
    delta_sq = sum(fracs[el] * (1 - ATOMIC_RADII.get(el, 130) / r_bar)**2 for el in elements)
    delta = np.sqrt(delta_sq) * 100  # express as percentage

    # 4. Valence electron concentration
    vec = sum(fracs[el] * VEC_D.get(el, 6) for el in elements)

    # 5. Omega parameter (phase stability criterion)
    # Estimate Tmelt as composition-weighted average
    tmelt_ref = {
        "Fe":1811,"Co":1768,"Ni":1728,"Cr":2180,"Mn":1519,"V":2183,
        "Al":933,"Ti":1941,"Mo":2896,"W":3695,"Nb":2750,"Ta":3290,
        "Cu":1358,"Zn":693,"Mg":923,"Si":1687,
    }
    t_melt = sum(fracs[el] * tmelt_ref.get(el, 1800) for el in elements)
    omega = (t_melt * ds_mix) / (abs(dh_mix * 1000) + 1e-6)  # dh in J

    return {
        "feat_hea_mixing_entropy":  round(ds_mix, 4),
        "feat_hea_mixing_enthalpy": round(dh_mix, 4),
        "feat_hea_atomic_mismatch": round(delta, 4),
        "feat_hea_vec":             round(vec, 4),
        "feat_hea_n_elements":      len(elements),
        "feat_hea_omega":           round(omega, 4),
    }
