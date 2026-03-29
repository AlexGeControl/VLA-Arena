import { createContext, useContext, useState, type ReactNode } from 'react';
import type { EpisodeMeta, SampledLayer } from '../types/api';
import { useEpisodeMeta } from '../hooks/useEpisodeMeta';

interface EpisodeState {
  meta: EpisodeMeta | null;
  loading: boolean;
  error: string | null;
  timestep: number;
  layer: SampledLayer;
  head: number;
  actionIndex: number;
  highlightedTokenIndex: number | null;
  setTimestep: (t: number) => void;
  setLayer: (l: SampledLayer) => void;
  setHead: (h: number) => void;
  setActionIndex: (a: number) => void;
  setHighlightedTokenIndex: (idx: number | null) => void;
}

const EpisodeContext = createContext<EpisodeState | null>(null);

interface EpisodeProviderProps {
  episodeId: string;
  children: ReactNode;
}

/** Provides shared episode state (selections + metadata) to all descendants. */
export function EpisodeProvider({ episodeId, children }: EpisodeProviderProps) {
  const { meta, loading, error } = useEpisodeMeta(episodeId);
  const [timestep, setTimestep] = useState(0);
  const [layer, setLayer] = useState<SampledLayer>(0);
  const [head, setHead] = useState(0);
  const [actionIndex, setActionIndex] = useState(0);
  const [highlightedTokenIndex, setHighlightedTokenIndex] = useState<number | null>(null);

  const value: EpisodeState = {
    meta,
    loading,
    error,
    timestep,
    layer,
    head,
    actionIndex,
    highlightedTokenIndex,
    setTimestep,
    setLayer,
    setHead,
    setActionIndex,
    setHighlightedTokenIndex,
  };

  return (
    <EpisodeContext.Provider value={value}>
      {children}
    </EpisodeContext.Provider>
  );
}

/**
 * Access the current episode state from context.
 * Must be called within an EpisodeProvider.
 */
export function useEpisode(): EpisodeState {
  const ctx = useContext(EpisodeContext);
  if (!ctx) {
    throw new Error('useEpisode must be used within an EpisodeProvider');
  }
  return ctx;
}
