import { useState, useCallback, useRef } from "react";
import { api, GenerationResult, OptimizationTarget } from "@/lib/api";

export function useInverseDesign() {
  const [running, setRunning]   = useState(false);
  const [generation, setGeneration] = useState(0);
  const [paretoFront, setParetoFront] = useState<GenerationResult["pareto_front"]>([]);
  const [bestFitness, setBestFitness] = useState<number[]>([]);
  const [error, setError]       = useState<string | null>(null);
  const cleanupRef = useRef<(() => void) | null>(null);

  const startOptimization = useCallback((
    alloyFamily: string,
    targets: OptimizationTarget[],
    constraints: Record<string, { min?: number; max?: number }>,
  ) => {
    setRunning(true); setError(null); setGeneration(0);

    const cleanup = api.connectOptimizer(
      alloyFamily,
      targets,
      constraints,
      (result: GenerationResult) => {
        setGeneration(result.generation);
        setParetoFront(result.pareto_front);
        setBestFitness(result.best_fitness);
      },
      (finalPareto) => {
        setParetoFront(finalPareto);
        setRunning(false);
      },
      (msg) => {
        setError(msg);
        setRunning(false);
      },
    );
    cleanupRef.current = cleanup;
  }, []);

  const stopOptimization = useCallback(() => {
    cleanupRef.current?.();
    setRunning(false);
  }, []);

  return { startOptimization, stopOptimization, running, generation, paretoFront, bestFitness, error };
}
