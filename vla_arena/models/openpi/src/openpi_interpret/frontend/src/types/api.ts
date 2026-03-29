/** Sampled transformer layers available for inspection. */
export const SAMPLED_LAYERS = [0, 3, 6, 9, 12, 15, 17] as const;
export type SampledLayer = (typeof SAMPLED_LAYERS)[number];

/** Camera names available in VLA-Arena episodes. */
export type CameraName = "base_0_rgb" | "left_wrist_0_rgb" | "right_wrist_0_rgb";

/** Compact episode info returned by the list endpoint. */
export interface EpisodeSummary {
  episode_id: string;
  task_instruction: string;
  num_timesteps: number;
}

/** Full episode metadata including token and layer info. */
export interface EpisodeMeta {
  episode_id: string;
  task_instruction: string;
  num_timesteps: number;
  instruction_tokens: string[];
  sampled_layers: SampledLayer[];
  camera_names: CameraName[];
}

/** Metadata for a single token in the sequence. */
export interface TokenMeta {
  index: number;
  type: "image_patch" | "language" | "state" | "action";
  source: string;
  patch_row?: number;
  patch_col?: number;
  token_text?: string;
  token_position?: number;
}

/** Per-modality breakdown of attention weights for one action query. */
export interface AttentionBreakdown {
  cameras: Record<CameraName, number[]>;
  camera_totals: Record<CameraName, number>;
  language_weights: number[];
  language_total: number;
  state_weight: number;
  action_weights: number[];
  action_total: number;
}

/** Full attention row with modality breakdown. */
export interface AttentionResponse {
  row: number[];
  breakdown: AttentionBreakdown;
}

/** Aggregated attention summary across all actions. */
export interface AttentionSummary {
  modality_totals: {
    images: number;
    language: number;
    state: number;
    actions: number;
  };
  per_action: number[];
}

/** Single point in a t-SNE embedding plot. */
export interface TsnePoint {
  index: number;
  x: number;
  y: number;
  type: string;
  source: string;
  color: string;
}

/** t-SNE embedding response for a layer. */
export interface TsneResponse {
  points: TsnePoint[];
}

/** A nearest neighbor in the embedding space. */
export interface NearestNeighbor {
  index: number;
  x: number;
  y: number;
  distance: number;
  modality_group: string;
  type: string;
  source: string;
}

/** Nearest-neighbor response for a selected action token. */
export interface NeighborResponse {
  selected: { index: number; x: number; y: number };
  neighbors: NearestNeighbor[];
}
