import { http, HttpResponse } from 'msw';
import type { EpisodeSummary, EpisodeMeta } from '../types/api';

const MOCK_EPISODE_SUMMARY: EpisodeSummary = {
  episode_id: "mock_001",
  task_instruction: "pick up the red cup and place it on the shelf",
  num_timesteps: 3,
};

const MOCK_EPISODE_META: EpisodeMeta = {
  episode_id: "mock_001",
  task_instruction: "pick up the red cup and place it on the shelf",
  num_timesteps: 3,
  instruction_tokens: ["pick", "up", "the", "red", "cup", "and", "place", "it", "on", "the", "shelf"],
  sampled_layers: [0, 3, 6, 9, 12, 15, 17],
  camera_names: ["base_0_rgb", "left_wrist_0_rgb"],
};

export const handlers = [
  http.get("*/api/episodes", () => {
    return HttpResponse.json([MOCK_EPISODE_SUMMARY]);
  }),

  http.get("*/api/episodes/:id", ({ params }) => {
    const { id } = params;
    if (id === "mock_001") {
      return HttpResponse.json(MOCK_EPISODE_META);
    }
    return new HttpResponse(null, { status: 404 });
  }),
];
