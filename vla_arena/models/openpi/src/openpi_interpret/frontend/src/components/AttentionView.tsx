import { useEpisode } from '../context/EpisodeContext';
import { useAttention } from '../hooks/useAttention';
import { AttentionSummaryBar } from './attention/AttentionSummaryBar';
import { LanguageAttention } from './attention/LanguageAttention';
import { ImageAttention } from './attention/ImageAttention';
import { StateAttention } from './attention/StateAttention';
import { CAMERA_NAMES, PATCHES_PER_CAMERA } from '../types/constants';
import { Skeleton } from './common/Skeleton';

export function AttentionView() {
  const {
    meta,
    loading: metaLoading,
    error: metaError,
    timestep,
    layer,
    head,
    actionIndex,
    highlightedTokenIndex,
    setHighlightedTokenIndex,
  } = useEpisode();
  const { data, loading: attentionLoading, error: attentionError } = useAttention(
    meta?.episode_id,
    timestep,
    layer,
    head,
    actionIndex,
  );

  if (metaLoading || !meta) {
    return (
      <div className="space-y-4 rounded-lg border border-gray-200 p-4">
        <Skeleton className="h-7 w-56" />
        <Skeleton className="h-16 w-full" />
        <Skeleton className="h-4 w-40" />
        <Skeleton className="h-24 w-full max-w-3xl" />
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <Skeleton className="h-72 w-64 max-w-full" />
          <Skeleton className="h-72 w-64 max-w-full" />
          <Skeleton className="h-72 w-64 max-w-full" />
        </div>
        <Skeleton className="h-12 w-48" />
      </div>
    );
  }
  if (metaError) {
    return <div className="p-4 text-red-600">Error: {metaError}</div>;
  }
  if (attentionLoading || !data) {
    return (
      <div className="space-y-4 rounded-lg border border-gray-200 p-4">
        <Skeleton className="h-7 w-56" />
        <Skeleton className="h-16 w-full" />
        <Skeleton className="h-4 w-40" />
        <Skeleton className="h-24 w-full max-w-3xl" />
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <Skeleton className="h-72 w-64 max-w-full" />
          <Skeleton className="h-72 w-64 max-w-full" />
          <Skeleton className="h-72 w-64 max-w-full" />
        </div>
        <Skeleton className="h-12 w-48" />
      </div>
    );
  }
  if (attentionError) {
    return <div className="p-4 text-red-600">Error: {attentionError}</div>;
  }

  const { breakdown } = data;

  return (
    <div className="space-y-4 rounded-lg border border-gray-200 p-4">
      <h2 className="text-lg font-semibold">Attention View</h2>
      <AttentionSummaryBar breakdown={breakdown} />

      <div>
        <h3 className="text-sm font-medium uppercase tracking-wide text-gray-500">
          Language instruction{' '}
          <span className="ml-1 font-normal normal-case text-orange-600">
            {(breakdown.language_total * 100).toFixed(1)}%
          </span>
        </h3>
        <LanguageAttention
          weights={breakdown.language_weights}
          tokens={meta.instruction_tokens}
          highlightedIndex={highlightedTokenIndex}
          onHover={setHighlightedTokenIndex}
        />
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {CAMERA_NAMES.map((cam, i) =>
          meta.camera_names.includes(cam) ? (
            <ImageAttention
              key={cam}
              cameraName={cam}
              weights={breakdown.cameras[cam]}
              totalWeight={breakdown.camera_totals[cam]}
              episodeId={meta.episode_id}
              timestep={timestep}
              highlightedIndex={highlightedTokenIndex}
              cameraOffset={i * PATCHES_PER_CAMERA}
              onHover={setHighlightedTokenIndex}
            />
          ) : (
            <div
              key={cam}
              className="flex h-64 w-64 items-center justify-center rounded-md bg-gray-200 text-sm text-gray-400"
            >
              No image
            </div>
          ),
        )}
      </div>

      <StateAttention weight={breakdown.state_weight} />
    </div>
  );
}
