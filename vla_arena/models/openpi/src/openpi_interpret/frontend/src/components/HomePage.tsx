import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import type { EpisodeSummary } from '../types/api';
import { api } from '../api/client';

/** Truncate text to maxLen characters with ellipsis. */
function truncate(text: string, maxLen: number): string {
  return text.length > maxLen ? text.slice(0, maxLen) + '…' : text;
}

/** Landing page listing all available episodes as clickable cards. */
export function HomePage() {
  const [episodes, setEpisodes] = useState<EpisodeSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.listEpisodes()
      .then(setEpisodes)
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="min-h-screen bg-white">
      <header className="bg-cmu-red text-white py-8 px-6">
        <h1 className="text-3xl font-bold tracking-tight">OpenPI InterpreT</h1>
        <p className="mt-2 text-white/80 text-lg">
          Interactive Pi-Zero Attention &amp; Embedding Explorer
        </p>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-10">
        {loading && (
          <div className="flex justify-center py-20">
            <div className="h-10 w-10 animate-spin rounded-full border-4 border-cmu-steel-gray border-t-cmu-red" />
          </div>
        )}

        {error && (
          <div className="rounded-lg bg-red-50 border border-red-200 p-4 text-red-700">
            Failed to load episodes: {error}
          </div>
        )}

        {!loading && !error && episodes.length === 0 && (
          <p className="text-center text-gray-500 py-20">No episodes available.</p>
        )}

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {episodes.map((ep) => (
            <Link
              key={ep.episode_id}
              to={`/episode/${ep.episode_id}`}
              className="block rounded-lg bg-white shadow hover:shadow-md transition-shadow border border-cmu-steel-gray p-5 hover:border-cmu-red/30"
            >
              <h2 className="text-sm font-semibold text-gray-500 mb-1">
                {ep.episode_id}
              </h2>
              <p className="text-gray-800 font-medium">
                {truncate(ep.task_instruction, 80)}
              </p>
              <p className="mt-2 text-xs text-gray-400">
                {ep.num_timesteps} timesteps
              </p>
            </Link>
          ))}
        </div>
      </main>
    </div>
  );
}
