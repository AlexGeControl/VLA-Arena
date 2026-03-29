import { useState, useEffect } from 'react';
import { api } from '../api/client';
import type { TsneResponse, SampledLayer } from '../types/api';

export function useTsne(episodeId: string | undefined, timestep: number, layer: SampledLayer) {
  const [data, setData] = useState<TsneResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!episodeId) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    api.getTsne(episodeId, timestep, layer)
      .then((res) => {
        if (!cancelled) setData(res);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [episodeId, timestep, layer]);

  return { data, loading, error };
}
