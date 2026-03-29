import { useState } from 'react';
import type { MouseEvent } from 'react';
import { useEpisode } from '../context/EpisodeContext';
import { useTsne } from '../hooks/useTsne';
import { useNeighbors } from '../hooks/useNeighbors';
import { TsneScatterPlot } from './embedding/TsneScatterPlot';
import { TsneLegend } from './embedding/TsneLegend';
import { TsneTooltip } from './embedding/TsneTooltip';
import { Skeleton } from './common/Skeleton';
import type { TsnePoint } from '../types/api';

export function EmbeddingView() {
  const {
    meta,
    timestep,
    layer,
    actionIndex,
    highlightedTokenIndex,
    setHighlightedTokenIndex,
  } = useEpisode();
  const { data: tsneData, loading, error } = useTsne(meta?.episode_id, timestep, layer);
  const { data: neighborData } = useNeighbors(meta?.episode_id, timestep, layer, actionIndex);
  const [tooltip, setTooltip] = useState<{ point: TsnePoint; x: number; y: number } | null>(
    null,
  );

  const handleHover = (index: number | null, event: MouseEvent<SVGElement>) => {
    if (index !== null && tsneData) {
      const point = tsneData.points[index];
      if (point) {
        setTooltip({ point, x: event.clientX, y: event.clientY });
        setHighlightedTokenIndex(index);
      }
    } else {
      setTooltip(null);
      setHighlightedTokenIndex(null);
    }
  };

  if (error) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
        {error}
      </div>
    );
  }

  if (loading || !tsneData) {
    return (
      <div className="rounded-lg border border-gray-200 bg-white p-4 dark:border-gray-700 dark:bg-gray-950">
        <Skeleton className="mb-3 h-5 w-48" />
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start">
          <Skeleton className="aspect-square min-h-[500px] w-full max-w-full flex-1" />
          <div className="flex w-full flex-col gap-2 lg:w-48">
            <Skeleton className="h-4 w-32" />
            <Skeleton className="h-3 w-full" />
            <Skeleton className="h-3 w-full" />
            <Skeleton className="h-3 w-3/4" />
          </div>
        </div>
        {!loading && (
          <p className="mt-2 text-center text-sm text-gray-500 dark:text-gray-400">
            No embedding data.
          </p>
        )}
      </div>
    );
  }

  return (
    <div className="relative rounded-lg border border-gray-200 bg-white p-4 dark:border-gray-700 dark:bg-gray-950">
      <h3 className="mb-3 text-sm font-semibold text-gray-800 dark:text-gray-100">
        Token embeddings (t-SNE)
      </h3>
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start">
        <div className="min-w-0 flex-1">
          <TsneScatterPlot
            points={tsneData.points}
            selectedActionIndex={actionIndex}
            neighbors={neighborData}
            highlightedTokenIndex={highlightedTokenIndex}
            onHover={handleHover}
          />
        </div>
        <TsneLegend />
      </div>
      <TsneTooltip
        point={tooltip?.point ?? null}
        position={tooltip ? { x: tooltip.x, y: tooltip.y } : null}
        episodeId={meta?.episode_id ?? null}
        timestep={timestep}
        allTokens={meta?.instruction_tokens ?? []}
      />
    </div>
  );
}
