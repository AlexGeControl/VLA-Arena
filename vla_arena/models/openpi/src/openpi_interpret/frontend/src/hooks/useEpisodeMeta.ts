import { useState, useEffect } from 'react';
import type { EpisodeMeta } from '../types/api';
import { api } from '../api/client';

interface UseEpisodeMetaResult {
  meta: EpisodeMeta | null;
  loading: boolean;
  error: string | null;
}

/**
 * Fetches episode metadata for the given ID.
 * Manages loading and error state internally.
 */
export function useEpisodeMeta(episodeId: string | undefined): UseEpisodeMetaResult {
  const [meta, setMeta] = useState<EpisodeMeta | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!episodeId) return;

    let cancelled = false;
    setLoading(true);
    setError(null);

    api.getEpisode(episodeId)
      .then((data) => {
        if (!cancelled) {
          setMeta(data);
          setLoading(false);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
          setLoading(false);
        }
      });

    return () => { cancelled = true; };
  }, [episodeId]);

  return { meta, loading, error };
}
