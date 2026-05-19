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
from typing import Generator, Callable, Dict, Tuple, List, Optional
import numpy as np
import random
from deap import base, creator, tools

# Import from models layer
from backend.models.predictor import predict_composition


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
            predictions = self.predictor(comp, self.alloy_family)
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
            preds = self.predictor(comp, self.alloy_family)
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
        initial_state: dict | None = None,
    ) -> Generator[GenerationResult, None, None]:
        """
        Run NSGA-II optimization, yielding a GenerationResult after each generation.
        Supports initial_state to enable rollback-safe checkpoints and recovery.
        """
        import time
        from loguru import logger
        start = time.time()

        # Ensure pop_size is a multiple of 4 (required by DEAP selTournamentDCD)
        if pop_size % 4 != 0:
            pop_size = max(4, ((pop_size // 4) + 1) * 4)

        if initial_state and "population" in initial_state:
            population = []
            for ind_data in initial_state["population"]:
                ind = creator.Individual(ind_data["ind"])
                ind.fitness.values = tuple(ind_data["fitness"])
                population.append(ind)
            if "random_state" in initial_state:
                try:
                    random.setstate(initial_state["random_state"])
                except Exception as r_err:
                    logger.warning("Could not restore random state: {}", r_err)
            start_gen = initial_state.get("start_generation", 1) + 1
            logger.info("Resuming AlloyOptimizer GA loop from generation {}", start_gen)
        else:
            population = self.toolbox.population(n=pop_size)

            # Evaluate initial population
            fitnesses = list(map(self.toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            # Assign crowding distance by selecting initial population via NSGA-II selection
            population = self.toolbox.select(population, pop_size)
            start_gen = 1

        # Run NSGA-II
        for gen in range(start_gen, n_generations + 1):
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

            # Build structural checkpoint package
            # This package gets exported to binary using msgpack + zstd
            current_checkpoint_data = {
                "population": [{"ind": list(ind), "fitness": list(ind.fitness.values)} for ind in population],
                "start_generation": gen,
                "random_state": random.getstate()
            }
            # Attach checkpoint data to generation result so worker can capture and write it safely
            res = GenerationResult(
                generation=gen,
                best_fitness=[max(ind.fitness.values[i] for ind in pareto_front) for i in range(len(self.targets))],
                pareto_front=[self._individual_to_result(ind) for ind in pareto_front[:20]],  # top 20 only
                population_size=len(population),
                constraint_violation_rate=violations / len(population),
                elapsed_seconds=round(time.time() - start, 2),
            )
            # Add custom attribute to pass state data without altering standard parameters
            res.checkpoint_state = current_checkpoint_data
            yield res

