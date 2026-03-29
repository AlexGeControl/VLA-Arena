import { useParams, Link } from 'react-router-dom';
import { EpisodeProvider, useEpisode } from '../context/EpisodeContext';
import { TimestepSlider } from './TimestepSlider';
import { SharedControls } from './SharedControls';
import { AttentionView } from './AttentionView';
import { EmbeddingView } from './EmbeddingView';

/** Inner content that consumes the EpisodeContext. */
function EpisodeContent() {
  const { meta, loading, error } = useEpisode();

  if (loading) {
    return (
      <div className="flex justify-center py-20">
        <div className="h-10 w-10 animate-spin rounded-full border-4 border-cmu-steel-gray border-t-cmu-red" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg bg-red-50 border border-red-200 p-4 text-red-700">
        Failed to load episode: {error}
      </div>
    );
  }

  if (!meta) return null;

  return (
    <>
      <header className="bg-cmu-red text-white py-6 px-6">
        <div className="max-w-7xl mx-auto">
          <Link to="/" className="text-sm text-white/70 hover:text-white transition-colors">
            &larr; All Episodes
          </Link>
          <h1 className="text-2xl font-bold mt-2 tracking-tight">{meta.episode_id}</h1>
          <p className="mt-1 text-white/80">{meta.task_instruction}</p>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-6 space-y-6">
        <TimestepSlider />
        <SharedControls />

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <AttentionView />
          <EmbeddingView />
        </div>
      </div>
    </>
  );
}

/** Episode exploration page wrapped in the shared episode context provider. */
export function EpisodePage() {
  const { episodeId } = useParams<{ episodeId: string }>();

  if (!episodeId) {
    return (
      <div className="p-8 text-center text-red-600">
        Missing episode ID in URL.
      </div>
    );
  }

  return (
    <EpisodeProvider episodeId={episodeId}>
      <div className="min-h-screen bg-white">
        <EpisodeContent />
      </div>
    </EpisodeProvider>
  );
}
