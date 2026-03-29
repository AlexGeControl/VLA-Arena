import { useState, useEffect } from 'react';
import { api } from '../api/client';
import type { AttentionResponse, SampledLayer } from '../types/api';

export function useAttention(
  episodeId: string | undefined,
  timestep: number,
  layer: SampledLayer,
  head: number,
  actionIndex: number,
) {
  const [data, setData] = useState<AttentionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!episodeId) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .getAttention(episodeId, timestep, layer, head, actionIndex)
      .then((res) => {
        if (!cancelled) setData(res);
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [episodeId, timestep, layer, head, actionIndex]);

  return { data, loading, error };
}
