"""
ALLOY IQ — Inverse Design Engine
==================================
Multi-objective genetic algorithm (NSGA-II via DEAP) for alloy composition
optimization given target property constraints.

Usage:
    from backend.engines.inverse_design import InverseDesignEngine
    engine = InverseDesignEngine(model_engine, family="steel")
    result = engine.optimize(
        targets={"yield_strength": (">", 900), "corrosion_pren": (">", 35)},
        constraints={"Cr": (0.15, 0.25), "Ni": (0.08, 0.12)},
        n_generations=100,
        pop_size=200,
    )
"""

from __future__ import annotations

import random
import warnings
from copy import deepcopy
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools

from backend.data.features import FeatureEngineer

# Suppress DEAP creator re-registration warnings on reimport
warnings.filterwarnings("ignore", ".*already defined.*")


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
Operator = Literal[">", "<", ">=", "<=", "=="]
Target = Tuple[Operator, float]
ElementConstraint = Tuple[float, float]  # (min, max) in weight-fraction


# ---------------------------------------------------------------------------
# Pareto front post-processing
# ---------------------------------------------------------------------------
def _extract_pareto_front(
    population,
    fitness_names: List[str],
    family: str,
    element_names: List[str],
) -> Dict:
    """Convert DEAP population individuals into serializable Pareto front."""
    front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    candidates = []
    for ind in front:
        composition = {el: float(w) for el, w in zip(element_names, ind)}
        props = {name: float(f) for name, f in zip(fitness_names, ind.fitness.values)}
        candidates.append({"composition": composition, "properties": props})
    return {
        "pareto_front": candidates,
        "n_candidates": len(candidates),
        "objective_axes": fitness_names,
        "alloy_family": family,
    }


# ---------------------------------------------------------------------------
# InverseDesignEngine
# ---------------------------------------------------------------------------
class InverseDesignEngine:
    """
    NSGA-II multi-objective GA that searches composition space to satisfy
    a user-defined property target profile.
    """

    # Default element pools per family (can be overridden)
    ELEMENT_POOLS = {
        "steel":    ["Fe", "Cr", "Ni", "Mo", "Mn", "C", "Si", "N", "Cu", "V", "Ti", "Nb"],
        "hea":      ["Fe", "Cr", "Ni", "Co", "Al", "Ti", "V",  "Mo", "Cu", "Mn", "Zr"],
        "aluminum": ["Al", "Mg", "Si", "Cu", "Zn", "Mn", "Cr", "Ti", "Zr", "Li"],
    }

    def __init__(self, model_engine, family: str):
        self.model_engine = model_engine
        self.family = family
        self.fe = FeatureEngineer(family)
        self._elements: List[str] = self.ELEMENT_POOLS[family]

    # ------------------------------------------------------------------
    def optimize(
        self,
        targets: Dict[str, Target],
        constraints: Optional[Dict[str, ElementConstraint]] = None,
        n_generations: int = 150,
        pop_size: int = 300,
        seed: int = 42,
    ) -> Dict:
        """
        Parameters
        ----------
        targets : dict
            e.g. {"yield_strength": (">", 900), "corrosion_pren": (">", 35)}
        constraints : dict
            e.g. {"Cr": (0.15, 0.25), "Ni": (0.08, 0.12)}
        n_generations : int
        pop_size : int

        Returns
        -------
        dict with "pareto_front", "n_candidates", "objective_axes"
        """
        random.seed(seed)
        np.random.seed(seed)

        constraints = constraints or {}
        prop_names = list(targets.keys())
        n_obj = len(prop_names)
        n_elem = len(self._elements)

        # Build element bounds array
        bounds_low  = np.zeros(n_elem)
        bounds_high = np.ones(n_elem)
        for i, el in enumerate(self._elements):
            if el in constraints:
                bounds_low[i], bounds_high[i] = constraints[el]

        # --- DEAP setup ---
        # Each objective is maximised (+1) or minimised (-1) based on operator
        weights = tuple(
            1.0 if op in (">", ">=") else -1.0
            for op, _ in targets.values()
        )

        # (Re-)create DEAP types safely
        if "FitnessMulti" not in creator.__dict__:
            creator.create("FitnessMulti", base.Fitness, weights=weights)
        if "Individual" not in creator.__dict__:
            creator.create("Individual", list, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()

        def random_individual() -> creator.Individual:
            """Random composition in [bounds_low, bounds_high] normalized to sum=1."""
            x = np.random.uniform(bounds_low, bounds_high)
            x = np.clip(x, bounds_low, bounds_high)
            x /= x.sum()
            return creator.Individual(x.tolist())

        toolbox.register("individual", random_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # ------------------------------------------------------------------
        def evaluate(ind) -> Tuple[float, ...]:
            """Evaluate one individual; penalize constraint violations."""
            x = np.array(ind)
            # Normalize to sum=1
            x = np.clip(x, 0, 1)
            if x.sum() < 1e-9:
                return tuple([1e9 * (-1 if w > 0 else 1) for w in weights])
            x /= x.sum()

            # Build composition dict → DataFrame
            comp = {el: [float(v)] for el, v in zip(self._elements, x)}
            df = pd.DataFrame(comp)

            # Compute features
            try:
                df_feat = self.fe.transform(df)
            except Exception:
                return tuple([0.0] * n_obj)

            # Predict each target property
            fitness_vals = []
            for prop in prop_names:
                try:
                    result = self.model_engine.predict(self.family, prop, df_feat)
                    fitness_vals.append(result["prediction"])
                except Exception:
                    fitness_vals.append(0.0)

            return tuple(fitness_vals)

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate",    tools.cxSimulatedBinaryBounded,
                         low=list(bounds_low), up=list(bounds_high), eta=20.0)
        toolbox.register("mutate",  tools.mutPolynomialBounded,
                         low=list(bounds_low), up=list(bounds_high),
                         eta=20.0, indpb=1.0 / n_elem)
        toolbox.register("select",  tools.selNSGA2)

        # --- Run NSGA-II ---
        pop = toolbox.population(n=pop_size)
        hof = tools.ParetoFront()

        algorithms.eaMuPlusLambda(
            pop, toolbox,
            mu=pop_size,
            lambda_=pop_size // 2,
            cxpb=0.7,
            mutpb=0.3,
            ngen=n_generations,
            halloffame=hof,
            verbose=False,
        )

        return _extract_pareto_front(hof, prop_names, self.family, self._elements)

    # ------------------------------------------------------------------
    def nearest_compliant_composition(
        self,
        composition: Dict[str, float],
        constraints: Dict[str, ElementConstraint],
    ) -> Dict:
        """
        Given an actual (slightly off-spec) composition, find the nearest
        in-spec composition vector by projection + normalization.
        Useful for the 'batch correction' use case (Cr came in 0.3% low).
        """
        x = np.array([composition.get(el, 0.0) for el in self._elements])

        # Clip to constraint bounds
        for i, el in enumerate(self._elements):
            if el in constraints:
                lo, hi = constraints[el]
                x[i] = np.clip(x[i], lo, hi)

        # Renormalize to sum=1
        if x.sum() > 1e-9:
            x /= x.sum()

        corrected = {el: float(v) for el, v in zip(self._elements, x)}
        l2_dist = float(np.linalg.norm(
            np.array(list(composition.values())) - x[:len(composition)]
        ))
        return {"corrected_composition": corrected, "l2_distance": l2_dist}
