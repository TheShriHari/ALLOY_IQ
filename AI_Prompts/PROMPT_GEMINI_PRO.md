# ALLOY IQ — Improvement Prompt: Gemini 3.1 Pro
**Role**: Inverse Design Engine · Pareto Optimization · WebSocket Streaming · Fatigue Model · Transfer Learning  
**Priority gaps you own**: Inverse Design Backend (CRITICAL), WebSocket GA streaming (HIGH), Fatigue & Fracture Toughness (MEDIUM)  
**Why Gemini Pro for these tasks**: These tasks require long-context reasoning across materials science literature, complex algorithm design, and multi-step optimization logic that benefits from your 1M-token context window.

---

## CONTEXT: WHAT ALREADY EXISTS

```
backend/
  main.py                     ← FastAPI app — add your WebSocket endpoint here
  ml/
    model_engine.py            ← trains XGBoost/RF/MLP; exposes predict()
  ingestion/
    matminer_retriever.py      ← fetches steel/HEA/Al datasets
frontend/
  src/app/inverse/page.tsx    ← UI EXISTS but calls a non-existent backend
```

The frontend `/inverse` page currently renders a form with:
- Target property inputs (e.g. "Yield Strength > 900 MPa")
- Element constraint inputs (e.g. "Carbon ≤ 1.0%", "No Tungsten")
- A "Start Optimization" button
- An empty Pareto front chart (Recharts/D3)
- A live generation counter that never updates

**The backend side is 100% missing.** Your job is to build it.

The ML model exports a fast prediction function at `models/predictor.py`:
```python
from models.predictor import predict_composition
# predict_composition({"Fe": 0.98, "C": 0.008, "Cr": 0.18}) 
# → {"yield_strength_mpa": 850.0, "tensile_strength_mpa": 1020.0, "hardness_hv": 285.0, "elongation_pct": 14.2}
```
This function must complete in <5ms — it is your fitness evaluator called ~50,000 times per GA run.

---

## TASK 1 — INVERSE DESIGN ENGINE: `backend/inverse/optimizer.py`

This is the most commercially important feature in ALLOY IQ. It answers: *"Given I need YS > 900 MPa and PREN > 35, what composition should I make?"*

### Architecture

Use the **DEAP** genetic algorithm library (`pip install deap`). The algorithm:
1. Encodes a candidate alloy as a vector of element fractions `[x_Fe, x_C, x_Cr, x_Ni, x_Mo, ...]`
2. Evaluates fitness by calling `predict_composition()` on each candidate
3. Runs NSGA-II (Non-dominated Sorting Genetic Algorithm II) for multi-objective optimization
4. Returns a Pareto front of non-dominated candidates after convergence

### Full implementation: `backend/inverse/optimizer.py`

```python
"""
ALLOY IQ — Inverse Design Engine
Genetic algorithm (NSGA-II via DEAP) for multi-objective alloy composition optimization.

Usage:
    optimizer = AlloyOptimizer(
        targets=[
            ObjectiveTarget("yield_strength_mpa", "maximize", min_val=900),
            ObjectiveTarget("corrosion_pren", "maximize", min_val=35),
        ],
        constraints={
            "frac_C": (0.0, 0.012),     # Carbon: 0–1.2 wt% equivalent
            "frac_Cr": (0.0, 0.30),     # Chromium: 0–30%
            "frac_Ni": (0.0, 0.25),     # Nickel: 0–25%
            "frac_Mo": (0.0, 0.08),     # Molybdenum: 0–8%
            "frac_Mn": (0.0, 0.02),     # Manganese: 0–2%
            # Fe gets remainder: always 1 - sum(others)
        },
        alloy_family="steel",
    )
    for generation_result in optimizer.run(n_generations=100, pop_size=200):
        # yields after each generation — use for WebSocket streaming
        yield generation_result
"""

from dataclasses import dataclass
from typing import Generator, Callable
import numpy as np
import random
from deap import base, creator, tools, algorithms

# Import from models layer (built by Claude Sonnet)
import sys; sys.path.append("..")
from models.predictor import predict_composition


@dataclass
class ObjectiveTarget:
    property_name: str          # e.g. "yield_strength_mpa"
    direction: str              # "maximize" or "minimize"
    min_val: float | None = None  # hard constraint: prediction must exceed this
    max_val: float | None = None  # hard constraint: prediction must be below this
    weight: float = 1.0         # relative importance weight


@dataclass
class GenerationResult:
    generation: int
    best_fitness: list[float]
    pareto_front: list[dict]    # list of {composition, predictions, fitness}
    population_size: int
    constraint_violation_rate: float   # fraction of population violating hard constraints
    elapsed_seconds: float


class AlloyOptimizer:
    # Elements we optimize over (Fe is derived as remainder)
    OPTIMIZABLE_ELEMENTS = [
        "C", "Cr", "Ni", "Mo", "Mn", "V", "Nb", "Si",
        "W", "Co", "Ti", "Al", "Cu", "N"
    ]

    def __init__(
        self,
        targets: list[ObjectiveTarget],
        constraints: dict[str, tuple[float, float]],
        alloy_family: str = "steel",
        predictor_fn: Callable | None = None,
    ):
        self.targets = targets
        self.constraints = constraints   # {element_frac_key: (min, max)}
        self.alloy_family = alloy_family
        self.predictor = predictor_fn or predict_composition

        # Build element list from constraints
        self.elements = [k.replace("frac_", "") for k in constraints]
        self.bounds = [(v[0], v[1]) for v in constraints.values()]
        self.n_dim = len(self.elements)

        # DEAP setup — must happen before creating individuals
        self._setup_deap()

    def _setup_deap(self):
        """Configure DEAP for NSGA-II multi-objective optimization."""
        n_objectives = len(self.targets)

        # Fitness: tuple of weights (negative = minimize, positive = maximize)
        weights = tuple(
            1.0 if t.direction == "maximize" else -1.0
            for t in self.targets
        )

        # Avoid re-registering if called multiple times
        if not hasattr(creator, "FitnessAlloy"):
            creator.create("FitnessAlloy", base.Fitness, weights=weights)
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessAlloy)

        self.toolbox = base.Toolbox()

        # Individual: random composition within bounds
        def random_individual():
            ind = []
            for lo, hi in self.bounds:
                ind.append(random.uniform(lo, hi))
            # Normalise so non-Fe fractions sum to ≤ 0.60 (Fe must be ≥ 40%)
            total = sum(ind)
            if total > 0.60:
                scale = 0.60 / total
                ind = [x * scale for x in ind]
            return creator.Individual(ind)

        self.toolbox.register("individual", random_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Genetic operators
        self.toolbox.register("evaluate", self._evaluate)
        self.toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                              low=[b[0] for b in self.bounds],
                              up=[b[1] for b in self.bounds],
                              eta=20)
        self.toolbox.register("mutate", tools.mutPolynomialBounded,
                              low=[b[0] for b in self.bounds],
                              up=[b[1] for b in self.bounds],
                              eta=20, indpb=1.0/self.n_dim)
        self.toolbox.register("select", tools.selNSGA2)

    def _decode(self, individual: list) -> dict:
        """Decode a DEAP individual into a composition dict."""
        comp = {el: max(0.0, val) for el, val in zip(self.elements, individual)}
        # Fe is the balance element
        non_fe_sum = sum(comp.values())
        comp["Fe"] = max(0.0, 1.0 - non_fe_sum)
        return comp

    def _evaluate(self, individual: list) -> tuple[float, ...]:
        """
        Fitness function: calls the ML model and returns objective values.
        Hard constraint violations return a heavily penalized fitness.
        """
        comp = self._decode(individual)

        # Hard constraint: Fe must be at least 40% (it's a steel/alloy, not a ceramic)
        if comp.get("Fe", 0) < 0.40:
            return tuple(-1e6 * (1 if t.direction == "maximize" else -1) for t in self.targets)

        try:
            predictions = self.predictor(comp)
        except Exception:
            return tuple(-1e6 for _ in self.targets)

        fitness_vals = []
        for target in self.targets:
            val = predictions.get(target.property_name, 0.0)

            # Apply hard bound penalties
            if target.min_val is not None and val < target.min_val:
                val = val - (target.min_val - val) * 10   # penalty
            if target.max_val is not None and val > target.max_val:
                val = val - (val - target.max_val) * 10

            fitness_vals.append(val * target.weight)

        return tuple(fitness_vals)

    def _individual_to_result(self, ind: list) -> dict:
        """Convert a DEAP individual to a serializable result dict."""
        comp = self._decode(ind)
        try:
            preds = self.predictor(comp)
        except Exception:
            preds = {}
        return {
            "composition": {k: round(v, 5) for k, v in comp.items() if v > 1e-5},
            "predictions": preds,
            "fitness": list(ind.fitness.values),
        }

    def run(
        self,
        n_generations: int = 100,
        pop_size: int = 200,
        cxpb: float = 0.9,
        mutpb: float = 0.1,
    ) -> Generator[GenerationResult, None, None]:
        """
        Run NSGA-II optimization, yielding a GenerationResult after each generation.
        This is a generator — iterate over it to get streaming updates.

        Example:
            for result in optimizer.run(n_generations=100):
                send_to_websocket(result)
        """
        import time
        start = time.time()

        population = self.toolbox.population(n=pop_size)

        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Run NSGA-II
        for gen in range(1, n_generations + 1):
            # Select offspring
            offspring = tools.selTournamentDCD(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mutpb:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate offspring that were modified
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid))
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit

            # Select next generation (NSGA-II selection)
            population = self.toolbox.select(population + offspring, pop_size)

            # Extract Pareto front (first non-dominated front)
            pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

            # Compute constraint violation rate
            violations = sum(
                1 for ind in population
                if self._decode(ind).get("Fe", 0) < 0.40
            )

            yield GenerationResult(
                generation=gen,
                best_fitness=[max(ind.fitness.values[i] for ind in pareto_front) for i in range(len(self.targets))],
                pareto_front=[self._individual_to_result(ind) for ind in pareto_front[:20]],  # top 20 only
                population_size=len(population),
                constraint_violation_rate=violations / len(population),
                elapsed_seconds=round(time.time() - start, 2),
            )
```

---

## TASK 2 — PARETO FRONT MODULE: `backend/inverse/pareto.py`

```python
"""
Post-processing for Pareto front analysis.
Filters, ranks, and annotates Pareto-optimal alloy candidates.
"""
import numpy as np

def rank_pareto_candidates(candidates: list[dict], priorities: list[str]) -> list[dict]:
    """
    Rank Pareto-front candidates by weighted priority.
    priorities: ordered list of property names by user importance
               e.g. ["yield_strength_mpa", "corrosion_pren", "elongation_pct"]
    """
    if not candidates:
        return []

    scored = []
    for c in candidates:
        preds = c.get("predictions", {})
        score = 0.0
        for rank, prop in enumerate(priorities):
            weight = 1.0 / (rank + 1)   # higher priority → higher weight
            score += preds.get(prop, 0) * weight
        scored.append({**c, "_priority_score": score})

    return sorted(scored, key=lambda x: x["_priority_score"], reverse=True)


def classify_candidate(candidate: dict) -> dict:
    """
    Add metallurgical classification and application suggestions to a candidate.
    Returns the candidate dict with added 'classification' and 'suggested_applications' fields.
    """
    preds = candidate.get("predictions", {})
    comp = candidate.get("composition", {})

    ys = preds.get("yield_strength_mpa", 0)
    hv = preds.get("hardness_hv", 0)
    cr = comp.get("Cr", 0) * 100  # approximate wt%
    pren = preds.get("corrosion_pren", cr)

    # Steel classification
    if ys > 1500:
        alloy_class = "Ultra-high-strength steel (UHSS)"
        applications = ["Aerospace structural", "Armor plate", "High-performance fasteners"]
    elif ys > 900:
        alloy_class = "High-strength steel (HSS)"
        applications = ["Automotive structural", "Pressure vessels", "Tool steel"]
    elif ys > 500:
        alloy_class = "Medium-strength alloy"
        applications = ["General engineering", "Pipelines", "Construction"]
    else:
        alloy_class = "Low-strength / ductile alloy"
        applications = ["Sheet metal forming", "Deep drawing", "Electrical applications"]

    # Corrosion classification overlay
    if pren >= 40:
        alloy_class += " + Super corrosion resistant"
        applications.append("Offshore oil & gas")
        applications.append("Chemical processing equipment")
    elif pren >= 25:
        alloy_class += " + Corrosion resistant"
        applications.append("Marine environment")

    return {
        **candidate,
        "classification": alloy_class,
        "suggested_applications": applications[:3],
    }


def filter_feasible(candidates: list[dict], constraints: dict) -> list[dict]:
    """
    Remove candidates where element fractions violate user-specified hard constraints.
    constraints: {"frac_C": {"max": 0.012}, "frac_Cr": {"min": 0.10}, ...}
    """
    feasible = []
    for c in candidates:
        comp = c.get("composition", {})
        valid = True
        for el_key, bounds in constraints.items():
            el = el_key.replace("frac_", "")
            val = comp.get(el, 0.0)
            if "min" in bounds and val < bounds["min"]:
                valid = False; break
            if "max" in bounds and val > bounds["max"]:
                valid = False; break
        if valid:
            feasible.append(c)
    return feasible
```

---

## TASK 3 — WEBSOCKET STREAMING: `backend/ws/optimizer_ws.py`

**Problem**: The GA runs for 60-120 seconds. Without streaming, the user sees a dead spinner for 2 minutes then a dump of results. The real-time generation feed — watching the Pareto front evolve — is the most visually impressive feature in the product.

```python
"""
WebSocket endpoint for real-time inverse design optimization streaming.
Streams GenerationResult objects as JSON after each GA generation.

Frontend connects to: ws://localhost:8000/ws/optimize
Frontend sends: {"targets": [...], "constraints": {...}, "n_generations": 100}
Server streams: one JSON message per generation
"""
from fastapi import WebSocket, WebSocketDisconnect
from fastapi import APIRouter
import json
import asyncio
import dataclasses

from inverse.optimizer import AlloyOptimizer, ObjectiveTarget

router = APIRouter()

def _result_to_json(result) -> str:
    """Convert GenerationResult dataclass to JSON string."""
    d = dataclasses.asdict(result)
    return json.dumps(d)


@router.websocket("/ws/optimize")
async def optimize_websocket(websocket: WebSocket):
    """
    WebSocket handler for live inverse design streaming.

    Protocol:
      1. Client connects
      2. Server sends: {"status": "connected", "message": "Ready to optimize"}
      3. Client sends optimization parameters JSON
      4. Server streams one message per generation:
           {"generation": 5, "best_fitness": [...], "pareto_front": [...], ...}
      5. Server sends final: {"status": "complete", "best_candidates": [...]}
      6. Connection closes
    """
    await websocket.accept()
    await websocket.send_json({"status": "connected", "message": "ALLOY IQ optimizer ready"})

    try:
        # Receive optimization parameters
        raw = await websocket.receive_text()
        params = json.loads(raw)

        # Parse targets
        targets = [
            ObjectiveTarget(
                property_name=t["property"],
                direction=t.get("direction", "maximize"),
                min_val=t.get("min_val"),
                max_val=t.get("max_val"),
                weight=t.get("weight", 1.0),
            )
            for t in params.get("targets", [])
        ]

        if not targets:
            await websocket.send_json({"status": "error", "message": "No targets specified"})
            return

        # Parse constraints
        constraints = {
            f"frac_{el}": (bounds.get("min", 0.0), bounds.get("max", 1.0))
            for el, bounds in params.get("constraints", {}).items()
        }

        n_gen = min(params.get("n_generations", 100), 200)   # cap at 200 for server safety
        pop_size = min(params.get("pop_size", 150), 300)

        await websocket.send_json({
            "status": "starting",
            "message": f"Starting NSGA-II: {n_gen} generations, {pop_size} population, {len(targets)} objectives"
        })

        # Run optimizer in thread to avoid blocking event loop
        optimizer = AlloyOptimizer(targets=targets, constraints=constraints)

        loop = asyncio.get_event_loop()
        all_results = []

        def run_sync():
            results = []
            for result in optimizer.run(n_generations=n_gen, pop_size=pop_size):
                results.append(result)
            return results

        # Stream generations — run GA in executor to keep event loop free
        # For production: use Celery. For MVP: use thread executor.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_sync)
            # Poll every 0.5s and send any new results
            last_sent = 0
            while not future.done():
                await asyncio.sleep(0.5)
                # In real implementation, use a queue for inter-thread communication
                # For MVP: check if we can yield partial results
            
            all_results = future.result()

        # Send all generation results
        for result in all_results:
            msg = _result_to_json(result)
            await websocket.send_text(msg)
            await asyncio.sleep(0.01)   # prevent flooding

        # Send completion
        final_pareto = all_results[-1].pareto_front if all_results else []
        await websocket.send_json({
            "status": "complete",
            "total_generations": len(all_results),
            "final_pareto_size": len(final_pareto),
            "best_candidates": final_pareto[:5],   # top 5 for summary
        })

    except WebSocketDisconnect:
        pass   # Client disconnected — clean exit
    except Exception as e:
        await websocket.send_json({"status": "error", "message": str(e)})
    finally:
        await websocket.close()
```

**Register the router in `main.py`**:
```python
from ws.optimizer_ws import router as ws_router
app.include_router(ws_router)
```

---

## TASK 4 — FATIGUE & FRACTURE TOUGHNESS: `backend/ml/fatigue_model.py`

**Problem**: Fatigue limit and fracture toughness (KIc) are listed as target properties in the project spec but are completely absent from `model_engine.py`. This is the primary property needed by aerospace and defense buyers.

**Data strategy** (since KIc data is sparse):

1. **For steels**: Use empirical proxies. The Barsom-Rolfe correlation is widely accepted:
   `KIc ≈ 0.64 × (σy / 1000)^0.5 × (HV × 9.81 / 3 - σy/3)^0.5` (approximate)
   A better approach: use UTS and Charpy impact energy as intermediate targets, then apply the conversion.

2. **Transfer learning from UTS**: Train a base regressor on UTS (large dataset). Fine-tune a shallow network on the ~300 available KIc data points using the base as feature extractor.

**Create `backend/ml/fatigue_model.py`**:

```python
"""
Fatigue limit and fracture toughness prediction module.
Strategy: empirical proxies + transfer learning from UTS model.

Two approaches:
  1. Direct: if KIc data is available, train directly (data from ASM Fracture database)
  2. Proxy: use Barsom-Rolfe / Charpy correlations for steel screening
"""
import numpy as np

# ── Proxy models (no training data required) ──────────────────────

def estimate_fatigue_limit(uts_mpa: float, alloy_family: str) -> dict:
    """
    Estimate fatigue limit from UTS using Wöhler-law approximations.

    For steels: σ_f ≈ 0.45–0.50 × UTS (valid up to UTS ≈ 1400 MPa)
    For Al alloys: σ_f ≈ 0.30–0.40 × UTS (lower fatigue ratio)
    For HEAs: σ_f ≈ 0.40–0.45 × UTS (similar to steels, limited data)

    Returns estimate with uncertainty range (±15% for this proxy approach).
    """
    if alloy_family == "steel":
        if uts_mpa <= 1400:
            ratio_mean, ratio_std = 0.475, 0.025
        else:
            # High-strength steels: ratio drops (hydrogen embrittlement, inclusions)
            ratio_mean, ratio_std = 0.40, 0.04
    elif alloy_family == "aluminum":
        ratio_mean, ratio_std = 0.35, 0.05
    else:   # HEA
        ratio_mean, ratio_std = 0.425, 0.035

    fl_mean = uts_mpa * ratio_mean
    fl_lower = uts_mpa * (ratio_mean - 2 * ratio_std)
    fl_upper = uts_mpa * (ratio_mean + 2 * ratio_std)

    return {
        "fatigue_limit_mpa": round(fl_mean, 1),
        "fatigue_limit_lower": round(fl_lower, 1),
        "fatigue_limit_upper": round(fl_upper, 1),
        "fatigue_ratio": round(ratio_mean, 3),
        "method": "Wöhler proxy from UTS",
        "confidence": "screening only — physical S-N curve testing required for design",
    }


def estimate_fracture_toughness(ys_mpa: float, hv: float, alloy_family: str) -> dict:
    """
    Estimate KIc from yield strength and hardness using Barsom-Rolfe correlation.

    Barsom-Rolfe (1999) for steels:
        KIc ≈ 0.64 × (σy²) / E    [simplified — full form needs CVN impact energy]

    Better proxy using HV (Vickers hardness):
        E_approx = 3 × HV × 9.81   (approximate yield stress from hardness)
        Then apply Barsom-Rolfe

    Returns KIc in MPa√m with wide uncertainty (±30% — proxy only).
    """
    # Young's modulus approximation from alloy family
    E_gpa = {"steel": 210, "aluminum": 70, "hea": 180}.get(alloy_family, 200)

    # Barsom-Rolfe simplified
    kic_estimate = 0.64 * (ys_mpa ** 2) / (E_gpa * 1000)   # in MPa√m

    # Adjust for hardness (high HV → lower toughness due to reduced plastic zone)
    hv_penalty = np.clip((hv - 200) / 600, 0, 0.40)  # up to 40% reduction
    kic_estimate *= (1 - hv_penalty)

    kic_estimate = np.clip(kic_estimate, 20, 200)  # physical bounds for steels

    return {
        "fracture_toughness_kic_mpa_sqrtm": round(kic_estimate, 1),
        "kic_lower": round(kic_estimate * 0.70, 1),
        "kic_upper": round(kic_estimate * 1.30, 1),
        "method": "Barsom-Rolfe proxy (no Charpy data)",
        "ndt_guidance": _ndt_guidance(kic_estimate, ys_mpa),
        "confidence": "screening only — ASTM E399 testing required for design certification",
    }


def _ndt_guidance(kic: float, ys: float) -> str:
    """Generate LEFM (Linear Elastic Fracture Mechanics) design guidance."""
    # NDT (Non-Destructive Testing) detectable crack size from KIc
    # a_NDT = (KIc / (1.12 × σy))² / π
    sigma_design = ys * 0.67   # assume 2/3 yield as design stress
    a_ndt_m = (kic / (1.12 * sigma_design)) ** 2 / np.pi
    a_ndt_mm = a_ndt_m * 1000

    if a_ndt_mm > 10:
        return f"NDT detectable flaw size: ~{a_ndt_mm:.0f} mm — easy to inspect, robust to surface defects"
    elif a_ndt_mm > 1:
        return f"NDT detectable flaw size: ~{a_ndt_mm:.1f} mm — standard UT/dye-penetrant inspection adequate"
    else:
        return f"NDT detectable flaw size: ~{a_ndt_mm:.2f} mm — high-resolution TOFD/phased array UT required"


# ── Integrate into /predict/mechanical ──────────────────────────────

def add_fatigue_fracture(prediction: dict, composition: dict, alloy_family: str) -> dict:
    """
    Add fatigue and fracture toughness estimates to an existing prediction dict.
    Call this after the main ML model runs.
    """
    uts = prediction.get("tensile_strength_mpa", {}).get("mean", 800)
    ys  = prediction.get("yield_strength_mpa",   {}).get("mean", 600)
    hv  = prediction.get("hardness_hv",           {}).get("mean", 250)

    fatigue = estimate_fatigue_limit(uts, alloy_family)
    fracture = estimate_fracture_toughness(ys, hv, alloy_family)

    prediction["fatigue"] = fatigue
    prediction["fracture_toughness"] = fracture
    return prediction
```

**Wire into `/predict/mechanical`** in `main.py`:
```python
from ml.fatigue_model import add_fatigue_fracture

# After existing predictions:
response = add_fatigue_fracture(response, composition_dict, alloy_family)
```

---

## TASK 5 — TRANSFER LEARNING STRATEGY (Document + Implement)

**Problem**: HEA fatigue and corrosion data is sparse (~150-200 points). Standard ML models underfit.

**Create `backend/ml/transfer_learning.py`**:

The strategy: train a base MLP on steels (large dataset, similar feature space). Freeze the first 2 hidden layers. Fine-tune only the last layer on HEA data. This is valid because:
- Lower layers learn element-property relationships (physics is shared across alloy families)  
- Upper layers learn family-specific structure-property mappings

```python
"""
Transfer learning: fine-tune steel-trained MLP on sparse HEA data.
Uses sklearn's MLPRegressor warm_start for the approximation.
For production: use PyTorch with proper layer freezing.
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
import joblib

def finetune_for_hea(
    base_model_path: str,
    X_hea_train: np.ndarray,
    y_hea_train: np.ndarray,
    n_finetune_iter: int = 50,
) -> MLPRegressor:
    """
    Fine-tune the steel-trained MLP on HEA data using warm_start.
    
    Limitation: sklearn MLPRegressor doesn't support layer freezing natively.
    This approximates transfer learning by continuing training from the steel model's
    weights as initialization — still significantly better than training from scratch.
    
    For true layer freezing, use: PyTorch / TensorFlow (see pytorch_transfer.py)
    """
    base_model: MLPRegressor = joblib.load(base_model_path)

    # Copy base model, enable warm_start, reduce learning rate for fine-tuning
    hea_model = MLPRegressor(
        hidden_layer_sizes=base_model.hidden_layer_sizes,
        activation=base_model.activation,
        max_iter=n_finetune_iter,
        warm_start=True,
        learning_rate_init=1e-4,   # 10× smaller than base training LR
        random_state=42,
    )

    # Transfer weights from base model
    hea_model.coefs_ = [c.copy() for c in base_model.coefs_]
    hea_model.intercepts_ = [i.copy() for i in base_model.intercepts_]
    hea_model.n_iter_ = base_model.n_iter_
    hea_model.n_outputs_ = base_model.n_outputs_
    hea_model.out_activation_ = base_model.out_activation_

    hea_model.fit(X_hea_train, y_hea_train)
    return hea_model
```

---

## INTEGRATION CHECKLIST

After completing all tasks, verify:

1. `python -m pytest tests/test_optimizer.py` — GA runs 10 generations, returns Pareto front with ≥3 candidates
2. `wscat -c ws://localhost:8000/ws/optimize` — WebSocket streams JSON messages
3. `curl /predict/mechanical` response includes `"fatigue"` and `"fracture_toughness"` fields
4. Pareto front candidates in `/ws/optimize` all have `Fe > 0.40`
5. `classify_candidate()` returns non-empty `"suggested_applications"` for all candidates

## HOW TO UPDATE THE AGENT TRACKER

```bash
python -c "
import json, datetime
with open('agent_tracker.json') as f: t = json.load(f)
t['agents']['gemini_pro']['status'] = 'in_progress'
t['agents']['gemini_pro']['current_task'] = 'TASK_1_INVERSE_DESIGN'
t['agents']['gemini_pro']['last_updated'] = datetime.datetime.utcnow().isoformat()
with open('agent_tracker.json', 'w') as f: json.dump(t, f, indent=2)
print('Tracker updated')
"
```
