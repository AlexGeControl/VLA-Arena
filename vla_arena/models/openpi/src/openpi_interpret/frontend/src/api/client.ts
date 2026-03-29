import type {
  EpisodeSummary,
  EpisodeMeta,
  TokenMeta,
  AttentionResponse,
  AttentionSummary,
  TsneResponse,
  NeighborResponse,
  SampledLayer,
  CameraName,
} from '../types/api';

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8080/api";

/**
 * Generic GET helper with error handling.
 * @throws Error with HTTP status and response body on non-OK responses.
 */
async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    throw new Error(`API error ${response.status}: ${await response.text()}`);
  }
  return response.json() as Promise<T>;
}

/** Typed API client for the OpenPI InterpreT backend. */
export const api = {
  /** List all available episodes. */
  listEpisodes: () => get<EpisodeSummary[]>("/episodes"),

  /** Fetch full metadata for a single episode. */
  getEpisode: (id: string) => get<EpisodeMeta>(`/episodes/${id}`),

  /** Build the URL for a camera image (used as an img src). */
  getCameraImageUrl: (id: string, camera: CameraName, timestep?: number) =>
    timestep !== undefined
      ? `${API_BASE}/episodes/${id}/camera/${camera}?timestep=${timestep}`
      : `${API_BASE}/episodes/${id}/camera/${camera}`,

  /** Fetch token metadata for a given timestep. */
  getTokenMeta: (id: string, timestep: number) =>
    get<TokenMeta[]>(`/episodes/${id}/timesteps/${timestep}/token-meta`),

  /** Fetch a single attention row with modality breakdown. */
  getAttention: (
    id: string,
    timestep: number,
    layer: SampledLayer,
    head: number,
    action: number,
  ) =>
    get<AttentionResponse>(
      `/episodes/${id}/timesteps/${timestep}/attention?layer=${layer}&head=${head}&action=${action}`,
    ),

  /** Fetch aggregated attention summary for all actions. */
  getAttentionSummary: (
    id: string,
    timestep: number,
    layer: SampledLayer,
    head: number,
  ) =>
    get<AttentionSummary>(
      `/episodes/${id}/timesteps/${timestep}/attention/summary?layer=${layer}&head=${head}`,
    ),

  /** Fetch t-SNE embedding points for a layer. */
  getTsne: (id: string, timestep: number, layer: SampledLayer) =>
    get<TsneResponse>(
      `/episodes/${id}/timesteps/${timestep}/tsne?layer=${layer}`,
    ),

  /** Fetch nearest neighbors for a selected action token. */
  getNeighbors: (
    id: string,
    timestep: number,
    layer: SampledLayer,
    action: number,
  ) =>
    get<NeighborResponse>(
      `/episodes/${id}/timesteps/${timestep}/tsne/neighbors?layer=${layer}&action=${action}`,
    ),
};
