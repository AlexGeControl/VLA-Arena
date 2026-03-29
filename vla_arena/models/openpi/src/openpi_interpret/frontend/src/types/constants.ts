export { SAMPLED_LAYERS } from './api';
export type { SampledLayer } from './api';

/** Number of attention heads per transformer layer. */
export const NUM_HEADS = 8;

/** Action prediction horizon (number of predicted actions). */
export const ACTION_HORIZON = 50;

/** Spatial dimension of the patch grid per camera (16x16). */
export const PATCH_GRID_SIZE = 16;

/** Total patches per camera image (PATCH_GRID_SIZE^2). */
export const PATCHES_PER_CAMERA = 256;

/** Ordered list of camera names. */
export const CAMERA_NAMES = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"] as const;

/** Image patch tokens across all cameras (3 × 256). */
export const NUM_IMAGE_PATCH_TOKENS = PATCHES_PER_CAMERA * CAMERA_NAMES.length;

/** First token index of the action horizon in the 867-token layout. */
export const ACTION_TOKEN_INDEX_BASE = 817;

/**
 * Color mapping for token modality groups.
 * Aligned with CMU Brand Standards (https://brand.cmu.edu/visual-identity/colors).
 *
 * - Cameras use cool tones (Blue Thread, Sky Blue, Hornbostel Teal) for spatial data
 * - Language uses Gold Thread for warmth and high visibility on dark backgrounds
 * - State uses Green Thread for a natural, grounded feel
 * - Action uses Carnegie Red — the dominant brand color — for the primary focus of analysis
 */
export const TOKEN_COLORS: Record<string, string> = {
  base_0_rgb: "#043673",       // Blue Thread — deep navy
  left_wrist_0_rgb: "#007BC0", // Highlands Sky Blue
  right_wrist_0_rgb: "#1F4C4C",// Hornbostel Teal — dark teal
  language: "#FDB515",         // Gold Thread
  state: "#009647",            // Green Thread
  action: "#C41230",           // Carnegie Red
};
