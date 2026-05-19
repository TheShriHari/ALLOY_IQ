import { useState, useCallback, useRef, useEffect } from "react";
import { api, ElementComposition, PredictionResponse } from "@/lib/api";

export function useBlenderRender() {
  const [status, setStatus] = useState<"idle" | "queued" | "running" | "complete" | "failed">("idle");
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  const cleanup = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const requestRender = useCallback(async (
    composition: ElementComposition,
    predictions: PredictionResponse,
  ) => {
    cleanup();
    setStatus("queued");
    setError(null);
    setImageUrl(null);
    
    try {
      const { job_id } = await api.requestRender(composition, predictions);

      // Poll every 3s until complete or failed
      pollRef.current = setInterval(async () => {
        try {
          const result = await api.pollRender(job_id);
          setStatus(result.status);
          if (result.status === "complete" && result.image_url) {
            setImageUrl(result.image_url);
            cleanup();
          } else if (result.status === "failed") {
            setError("Blender rendering failed on the server.");
            cleanup();
          }
        } catch (e) {
          setError(e instanceof Error ? e.message : "Polling failed");
          setStatus("failed");
          cleanup();
        }
      }, 3000);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to queue rendering");
      setStatus("failed");
    }
  }, [cleanup]);

  useEffect(() => {
    return cleanup;
  }, [cleanup]);

  return { requestRender, status, imageUrl, error };
}
