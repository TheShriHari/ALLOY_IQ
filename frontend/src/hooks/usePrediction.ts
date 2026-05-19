import { useState, useCallback } from "react";
import { api, ElementComposition, PredictionResponse, ShapResponse } from "@/lib/api";

export function usePrediction() {
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [shap, setShap]       = useState<ShapResponse | null>(null);

  const predict = useCallback(async (alloyFamily: string, property: string, composition: ElementComposition) => {
    setLoading(true); setError(null);
    try {
      const [pred, sh] = await Promise.all([
        api.predict(alloyFamily, property, composition),
        api.explain(alloyFamily, property, composition),
      ]);
      setPrediction(pred);
      setShap(sh);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }, []);

  return { predict, loading, error, prediction, shap, setPrediction, setShap };
}
