# ALLOY IQ — Improvement Prompt: Claude Sonnet 4.6
**Role**: ML Engine Overhaul · SHAP Narrative · UQ · PDP · Corrosion Physics · MLflow  
**Priority gaps you own**: UQ (CRITICAL), SHAP Narrative (HIGH), Stacking (HIGH), PDP (HIGH), HEA features (HIGH), Corrosion physics (MEDIUM), MLflow (MEDIUM)

---

## CONTEXT: WHAT ALREADY EXISTS

The project at `backend/ml/model_engine.py` trains XGBoost, RandomForest, and MLP with Optuna HPO and outputs SHAP values. The existing structure is:

```
backend/
  ml/
    model_engine.py        ← trains models, computes SHAP, saves .pkl
  main.py                  ← FastAPI app with /predict/mechanical and /predict/explain
  sync.py                  ← schema sync utility
```

The ingestion pipeline produces `data/processed/train.parquet`, `val.parquet`, `test.parquet` with these column conventions:
- `frac_{El}` — elemental mole fractions (e.g. `frac_Fe`, `frac_C`, `frac_Cr`)
- `feat_carbon_equivalent` — IIW CE index (already computed)
- `MagpieData {stat} {property}` — 132 Magpie descriptors
- Target columns: `yield_strength_mpa`, `tensile_strength_mpa`, `hardness_hv`, `elongation_pct`
- **NOT YET in data**: `corrosion_pren`, `fatigue_limit_mpa`, `fracture_toughness_kic`

---

## YOUR TASKS — EXECUTE IN THIS ORDER

---

### TASK 1 — Fix the multi-output stacking ensemble in `model_engine.py`

**Problem**: It is unclear whether the three base models (XGBoost, RF, MLP) predict all four targets jointly or independently. If independently, cross-property correlations are lost (YS and elongation are physically anti-correlated; a model that doesn't know this will make physically impossible predictions).

**What to implement**:

Open `backend/ml/model_engine.py` and restructure the training as a proper stacking ensemble:

```python
from sklearn.ensemble import StackingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge

# Base estimators — each must handle multi-output natively or be wrapped
base_estimators = [
    ("xgb", MultiOutputRegressor(XGBRegressor(tree_method="hist", **best_xgb_params))),
    ("rf",  RandomForestRegressor(**best_rf_params)),      # RF natively multi-output
    ("mlp", MultiOutputRegressor(MLPRegressor(**best_mlp_params))),
]

# Meta-learner sees concatenated predictions from all 3 base models
# For 4 targets × 3 models = 12 features going into Ridge
meta_learner = MultiOutputRegressor(Ridge(alpha=1.0))

stacking = StackingRegressor(
    estimators=base_estimators,
    final_estimator=meta_learner,
    cv=5,
    passthrough=True,   # also pass original features to meta-learner
    n_jobs=-1,
)
```

**Log-transform all targets before fitting**:
```python
import numpy as np
# Property distributions are right-skewed — log transforms improve R²
y_train_log = np.log1p(y_train)
stacking.fit(X_train_scaled, y_train_log)
# Inverse-transform after prediction:
y_pred = np.expm1(stacking.predict(X_test_scaled))
```

**After fitting, print a cross-property correlation check**:
```python
# Sanity check: in the predictions, YS and elongation should be negatively correlated
from scipy.stats import pearsonr
r, p = pearsonr(y_pred[:, 0], y_pred[:, 3])   # YS vs elongation
assert r < -0.2, f"Expected negative YS-elongation correlation, got r={r:.2f}"
```

If this assertion fails, the model has learned spurious relationships — log a `WARNING` and investigate feature leakage.

**Save the stacking ensemble as**:
- `models/stacking_ensemble.pkl` (joblib)
- `models/scaler.pkl` (the RobustScaler fitted on train)
- `models/target_names.json` — `["yield_strength_mpa", "tensile_strength_mpa", "hardness_hv", "elongation_pct"]`
- `models/eval_report.json` — RMSE, MAE, R² per target on test set

---

### TASK 2 — Add Uncertainty Quantification: `backend/ml/uncertainty.py`

**Problem**: The model returns a single point prediction. No engineer uses a single number to make safety-critical material decisions. This is the single biggest trust barrier for a commercial product.

**Create `backend/ml/uncertainty.py`**:

```python
"""
Conformal prediction wrapper for ALLOY IQ.
Uses MAPIE (pip install mapie) to produce calibrated prediction intervals.
Coverage guarantee: 90% of true values fall within the returned interval.
"""
from mapie.regression import MapieRegressor
from mapie.multi_output import MapieMultiOutputRegressor
import numpy as np, joblib
from pathlib import Path

class AlloyUncertainty:
    def __init__(self, base_model_path: str):
        self.base_model = joblib.load(base_model_path)
        self.mapie = None

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray, alpha: float = 0.10):
        """
        Calibrate conformal predictor on a held-out calibration set.
        alpha=0.10 → 90% coverage guarantee.
        X_cal: scaled feature matrix (run through RobustScaler first)
        y_cal: log-transformed target matrix
        """
        self.mapie = MapieMultiOutputRegressor(
            estimator=self.base_model,
            method="naive",     # fast; switch to "jackknife+" for tighter intervals on small data
            cv="prefit",        # base model already fitted — only calibrate conformity scores
        )
        self.mapie.fit(X_cal, y_cal)
        self.alpha = alpha

    def predict(self, X: np.ndarray) -> dict:
        """
        Returns point estimates + 90% prediction intervals for all targets.
        All values inverse-log-transformed back to original units.

        Returns:
            {
              "yield_strength_mpa":    {"mean": 820.0, "lower": 720.0, "upper": 910.0},
              "tensile_strength_mpa":  {"mean": 1020.0, "lower": 890.0, "upper": 1130.0},
              "hardness_hv":           {"mean": 285.0, "lower": 240.0, "upper": 320.0},
              "elongation_pct":        {"mean": 14.2,  "lower": 9.8,   "upper": 18.6},
            }
        """
        y_pred_log, y_intervals_log = self.mapie.predict(X, alpha=self.alpha)
        # y_pred_log: (n_samples, n_targets)
        # y_intervals_log: (n_samples, n_targets, 2) — [lower, upper]

        target_names = ["yield_strength_mpa","tensile_strength_mpa","hardness_hv","elongation_pct"]
        result = {}
        for i, name in enumerate(target_names):
            result[name] = {
                "mean":  float(np.expm1(y_pred_log[0, i])),
                "lower": float(np.expm1(y_intervals_log[0, i, 0])),
                "upper": float(np.expm1(y_intervals_log[0, i, 1])),
            }
        return result

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "AlloyUncertainty":
        return joblib.load(path)
```

**Wire this into `main.py`**: Replace the current prediction response with the uncertainty-aware response. The API response for `/predict/mechanical` must now return:

```json
{
  "predictions": {
    "yield_strength_mpa":   {"mean": 820, "lower": 720, "upper": 910},
    "tensile_strength_mpa": {"mean": 1020, "lower": 890, "upper": 1130},
    "hardness_hv":          {"mean": 285, "lower": 240, "upper": 320},
    "elongation_pct":       {"mean": 14.2, "lower": 9.8, "upper": 18.6}
  },
  "confidence_level": 0.90,
  "data_confidence": "high"
}
```

The `data_confidence` field is a string: `"high"` if interval width < 1.5× average training interval width, `"medium"` if 1.5–3×, `"low"` if >3× (the composition is far from training data).

---

### TASK 3 — SHAP Narrative Generator: `backend/ml/narrative.py`

**Problem**: The `/predict/explain` endpoint currently returns raw SHAP values (e.g. `{"frac_C": 0.312, "frac_Cr": 0.091, ...}`). A materials scientist cannot act on raw floats. The commercial differentiator is plain-English interpretation in metallurgical language.

**Create `backend/ml/narrative.py`**:

```python
"""
Generates plain-English metallurgical narratives from SHAP values.
Output is what a senior materials scientist would say about the prediction.
"""

ELEMENT_NAMES = {
    "C": "carbon", "Cr": "chromium", "Ni": "nickel", "Mo": "molybdenum",
    "Mn": "manganese", "V": "vanadium", "Nb": "niobium", "Si": "silicon",
    "W": "tungsten", "Co": "cobalt", "Ti": "titanium", "Al": "aluminum",
    "Cu": "copper", "N": "nitrogen", "B": "boron", "S": "sulfur", "P": "phosphorus",
    "Fe": "iron (base matrix)", "Zn": "zinc", "Mg": "magnesium",
}

MECHANISMS = {
    # Element → property → mechanism explanation
    ("C",  "yield_strength_mpa"):    "interstitial solid-solution strengthening and carbide precipitation",
    ("C",  "hardness_hv"):           "martensite formation and carbide precipitation hardening",
    ("C",  "elongation_pct"):        "increased brittleness from carbide networks (inverse relationship)",
    ("Cr", "hardness_hv"):           "secondary hardening via M7C3 and M23C6 carbide formation",
    ("Cr", "corrosion_pren"):        "passive oxide layer formation (Cr₂O₃) — PREN increase",
    ("Mo", "yield_strength_mpa"):    "solid-solution strengthening and secondary hardening",
    ("Mo", "corrosion_pren"):        "enhanced pitting resistance in chloride environments",
    ("Ni", "elongation_pct"):        "austenite stabilization improving ductility",
    ("Mn", "yield_strength_mpa"):    "solid-solution strengthening and hardenability improvement",
    ("Nb", "yield_strength_mpa"):    "grain refinement via NbC precipitation (Hall-Petch effect)",
    ("V",  "yield_strength_mpa"):    "vanadium carbide precipitation hardening",
    ("N",  "corrosion_pren"):        "PREN contribution: 16×N coefficient in pitting resistance formula",
    ("S",  "elongation_pct"):        "MnS inclusion embrittlement (negative effect)",
    ("S",  "yield_strength_mpa"):    "MnS inclusion weakening (detrimental)",
}

MAGNITUDE_WORDS = [
    (100, "dominant"),
    (50,  "significant"),
    (20,  "moderate"),
    (5,   "minor"),
    (0,   "negligible"),
]

def _magnitude_word(shap_abs: float) -> str:
    for threshold, word in MAGNITUDE_WORDS:
        if shap_abs >= threshold:
            return word
    return "negligible"

def _element_from_feature(feature_name: str) -> str | None:
    """Extract element symbol from feature name like 'frac_C' or 'frac_Cr'."""
    if feature_name.startswith("frac_"):
        return feature_name[5:]
    return None

def generate_narrative(
    shap_dict: dict[str, float],
    prediction: dict,
    intervals: dict,
    target: str = "yield_strength_mpa",
) -> str:
    """
    Generate a 3-sentence plain-English explanation of a prediction.

    Args:
        shap_dict: {feature_name: shap_value} sorted by abs value descending
        prediction: {"mean": 820.0, "lower": 720.0, "upper": 910.0}
        intervals:  same structure — used for confidence sentence
        target:     which property is being explained

    Returns:
        A 3-sentence string ready for display in the UI.
    """
    target_label = {
        "yield_strength_mpa": "yield strength",
        "tensile_strength_mpa": "tensile strength",
        "hardness_hv": "Vickers hardness",
        "elongation_pct": "elongation",
        "corrosion_pren": "pitting resistance (PREN)",
    }.get(target, target)

    units = {
        "yield_strength_mpa": "MPa",
        "tensile_strength_mpa": "MPa",
        "hardness_hv": "HV",
        "elongation_pct": "%",
        "corrosion_pren": "PREN units",
    }.get(target, "")

    # Find top positive and negative SHAP contributors
    sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    positives = [(k, v) for k, v in sorted_shap if v > 0]
    negatives = [(k, v) for k, v in sorted_shap if v < 0]

    sentences = []

    # Sentence 1: dominant positive driver
    if positives:
        feat, val = positives[0]
        el = _element_from_feature(feat)
        el_name = ELEMENT_NAMES.get(el, feat) if el else feat
        magnitude = _magnitude_word(abs(val))
        mechanism = MECHANISMS.get((el, target), "compositional effect") if el else "compositional effect"
        sentences.append(
            f"{el_name.capitalize()} is the {magnitude} positive driver of {target_label}, "
            f"contributing +{abs(val):.1f} {units} above baseline through {mechanism}."
        )

    # Sentence 2: dominant negative driver (if meaningful)
    if negatives and abs(negatives[0][1]) > 5:
        feat, val = negatives[0]
        el = _element_from_feature(feat)
        el_name = ELEMENT_NAMES.get(el, feat) if el else feat
        mechanism = MECHANISMS.get((el, target), "compositional effect") if el else "compositional effect"
        sentences.append(
            f"Conversely, {el_name} is the largest risk factor, "
            f"reducing predicted {target_label} by {abs(val):.1f} {units} via {mechanism}."
        )
    elif len(positives) > 1:
        # Secondary positive driver instead
        feat, val = positives[1]
        el = _element_from_feature(feat)
        el_name = ELEMENT_NAMES.get(el, feat) if el else feat
        sentences.append(
            f"Secondary reinforcement comes from {el_name} (+{abs(val):.1f} {units}), "
            f"with remaining elements having combined minor influence."
        )

    # Sentence 3: confidence context
    mean_val = prediction["mean"]
    lower = prediction["lower"]
    upper = prediction["upper"]
    interval_width = upper - lower
    avg_width = mean_val * 0.18   # 18% of mean is roughly average interval width for these properties
    if interval_width < avg_width * 1.2:
        confidence_text = (
            f"The 90% prediction interval is [{lower:.0f}, {upper:.0f}] {units} — "
            f"narrow relative to the mean, indicating this composition is well-represented in training data."
        )
    elif interval_width < avg_width * 2.0:
        confidence_text = (
            f"The 90% prediction interval is [{lower:.0f}, {upper:.0f}] {units}. "
            f"Moderate uncertainty suggests this composition sits near the boundary of well-studied alloys — "
            f"physical validation is recommended."
        )
    else:
        confidence_text = (
            f"⚠ Wide prediction interval [{lower:.0f}, {upper:.0f}] {units}. "
            f"This composition is extrapolating beyond well-studied alloy space. "
            f"Treat as directional guidance only — physical testing is required."
        )
    sentences.append(confidence_text)

    return " ".join(sentences)


def generate_full_report(shap_dicts: dict, predictions: dict, intervals: dict) -> dict:
    """Generate narratives for all predicted properties."""
    return {
        target: generate_narrative(shap_dicts.get(target, {}), predictions[target], predictions[target], target)
        for target in predictions
    }
```

**Wire into `/predict/explain`**: The response must now include a `"narrative"` field alongside the raw SHAP values:

```json
{
  "shap_values": {"frac_C": 142.3, "frac_Cr": 38.1, ...},
  "narrative": "Carbon is the dominant positive driver of yield strength, contributing +142.3 MPa above baseline through interstitial solid-solution strengthening and carbide precipitation. Conversely, sulfur is the largest risk factor, reducing predicted yield strength by 18.4 MPa via MnS inclusion weakening. The 90% prediction interval is [720, 910] MPa — narrow relative to the mean, indicating this composition is well-represented in training data."
}
```

---

### TASK 4 — HEA Physics Features: `backend/ml/hea_features.py`

**Problem**: The ingestion pipeline uses Magpie descriptors which are weighted averages of elemental properties. For High-Entropy Alloys, four thermodynamic descriptors computed from composition are equally important and are NOT captured by Magpie.

**Create `backend/ml/hea_features.py`**:

```python
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
```

**Integrate into the ingestion pipeline**: In `cleaner.py` or a new `feature_engineer.py`, call `compute_hea_features()` for every row where `alloy_family == "hea"`. For steels and Al alloys, set these features to 0.0 — the model will learn to ignore them for non-HEA rows via the `alloy_family` one-hot encoding.

**After adding these features, retrain the model** and confirm:
- HEA R² for YS improves by ≥5% on validation set
- `feat_hea_vec` appears in top 10 SHAP features for HEA rows
- `feat_hea_omega` > 1.5 corresponds to rows labeled as single-phase solid solutions

---

### TASK 5 — Partial Dependence Plots: `backend/ml/pdp.py` + API endpoint

**Problem**: There is no way for a user to ask "what happens to yield strength if I increase carbon from 0.1% to 1.0%?" This is the most important interactive feature for design exploration.

**Create `backend/ml/pdp.py`**:

```python
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence

def compute_pdp(
    model,
    scaler,
    X_train: np.ndarray,
    feature_name: str,
    feature_names: list[str],
    n_points: int = 50,
    percentile_range: tuple = (5, 95),
) -> dict:
    """
    Compute partial dependence of all targets on one feature.
    Sweeps the feature from its p5 to p95 in n_points steps,
    holding all others at their training-set median.

    Returns:
        {
          "feature": "frac_C",
          "x_values": [0.001, 0.003, ...],   ← actual element fraction values
          "predictions": {
            "yield_strength_mpa": [410.2, 445.1, ...],
            "tensile_strength_mpa": [...],
            ...
          },
          "x_label": "Carbon fraction",
          "x_unit": "mol fraction",
        }
    """
    feat_idx = feature_names.index(feature_name)
    col_data = X_train[:, feat_idx]
    lo = np.percentile(col_data, percentile_range[0])
    hi = np.percentile(col_data, percentile_range[1])
    sweep_vals = np.linspace(lo, hi, n_points)

    # Build synthetic samples: median for all features, sweep target feature
    X_median = np.median(X_train, axis=0)
    X_sweep = np.tile(X_median, (n_points, 1))
    X_sweep[:, feat_idx] = sweep_vals

    # Predict (model expects scaled input)
    # Note: X_train already scaled — sweep vals are already in scaled space
    y_pred_log = model.predict(X_sweep)
    y_pred = np.expm1(y_pred_log)   # inverse log-transform

    target_names = ["yield_strength_mpa","tensile_strength_mpa","hardness_hv","elongation_pct"]

    # Map feature name to human-readable label
    label_map = {f"frac_{el}": f"{el_name} fraction" for el, el_name in {
        "C":"Carbon","Cr":"Chromium","Ni":"Nickel","Mo":"Molybdenum",
        "Mn":"Manganese","V":"Vanadium","Fe":"Iron","Al":"Aluminum",
    }.items()}

    return {
        "feature": feature_name,
        "x_values": sweep_vals.tolist(),
        "predictions": {
            name: y_pred[:, i].tolist()
            for i, name in enumerate(target_names)
        },
        "x_label": label_map.get(feature_name, feature_name),
        "x_unit": "mol fraction",
    }
```

**Add to `main.py`**:
```python
@app.post("/api/v1/explain/pdp")
async def get_pdp(request: PDPRequest):
    """
    Sweep a single element fraction across its training range,
    returning predicted property values at each point.
    Used by the frontend's 'What-If Explorer' sliders.
    """
    result = compute_pdp(
        model=app.state.model,
        scaler=app.state.scaler,
        X_train=app.state.X_train,
        feature_name=f"frac_{request.element}",
        feature_names=app.state.feature_names,
    )
    return result
```

---

### TASK 6 — Corrosion Physics: `backend/ml/corrosion_features.py`

**Problem**: The current corrosion prediction outputs a dimensionless score. Oil & gas engineers cannot use this. They need PREN index, pitting potential Epit, and NACE MR0175 compliance language.

**Create `backend/ml/corrosion_features.py`**:

```python
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
```

**Add this to the /predict/mechanical response** as a `"corrosion_analysis"` field. The frontend should display the NACE guidance string prominently in the corrosion section.

---

### TASK 7 — MLflow Experiment Tracking: `backend/ml/mlflow_config.py`

**Problem**: Every time the model is retrained, previous metrics are overwritten. There is no way to roll back to the best-performing version.

**Create `backend/ml/mlflow_config.py`**:

```python
import mlflow
import mlflow.sklearn
from pathlib import Path

TRACKING_URI = "sqlite:///mlruns/mlflow.db"
EXPERIMENT_NAME = "alloyiq_property_prediction"

def setup_mlflow():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

def log_training_run(params: dict, metrics: dict, model, artifacts: list[str]):
    """
    Log a complete training run with params, metrics, model binary, and artifact files.
    Call this at the end of every training session.
    """
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)    # {"ys_r2": 0.91, "hv_rmse": 38.2, ...}
        mlflow.sklearn.log_model(model, "stacking_ensemble")
        for artifact_path in artifacts:
            mlflow.log_artifact(artifact_path)
        run_id = mlflow.active_run().info.run_id
        print(f"MLflow run logged: {run_id}")
        return run_id
```

**Integrate into training**: Wrap the full training block in `setup_mlflow()` + `log_training_run()`. Add a `GET /api/v1/model/versions` endpoint that queries MLflow for the last 5 runs and returns their metrics — the frontend can display this in a "Model Version" panel.

---

## QUALITY CHECKS BEFORE MARKING COMPLETE

After implementing all tasks, verify:

1. `python -m pytest tests/test_uncertainty.py` — all conformal coverage tests pass (actual coverage ≥ 88%)
2. `python -m pytest tests/test_narrative.py` — narrative output for a high-carbon steel mentions "carbon" and "interstitial"
3. `python -m pytest tests/test_pdp.py` — PDP for `frac_C` shows monotonically increasing YS trend
4. `curl -X POST /predict/mechanical -d '{"Fe":0.98,"C":0.008}'` — response includes `"lower"`, `"upper"`, `"narrative"`, `"pren_calculated"` fields
5. Stacking ensemble YS R² > 0.85 on test set (log this to MLflow)
6. HEA rows: `feat_hea_vec` in top 10 SHAP features

## HOW TO UPDATE THE AGENT TRACKER

```bash
python -c "
import json, datetime
with open('agent_tracker.json') as f: t = json.load(f)
t['agents']['claude_sonnet']['status'] = 'in_progress'
t['agents']['claude_sonnet']['current_task'] = 'TASK_2_UQ'
t['agents']['claude_sonnet']['last_updated'] = datetime.datetime.utcnow().isoformat()
with open('agent_tracker.json', 'w') as f: json.dump(t, f, indent=2)
print('Tracker updated')
"
```
