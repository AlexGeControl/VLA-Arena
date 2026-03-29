import { useState, useEffect } from 'react';
import { api } from '../api/client';
import type { NeighborResponse, SampledLayer } from '../types/api';

export function useNeighbors(
  episodeId: string | undefined,
  timestep: number,
  layer: SampledLayer,
  actionIndex: number,
) {
  const [data, setData] = useState<NeighborResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!episodeId) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    api.getNeighbors(episodeId, timestep, layer, actionIndex)
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
  }, [episodeId, timestep, layer, actionIndex]);

  return { data, loading, error };
}
