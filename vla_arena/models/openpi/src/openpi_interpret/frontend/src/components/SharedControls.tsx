import { useEpisode } from '../context/EpisodeContext';
import { SAMPLED_LAYERS } from '../types/constants';
import { NUM_HEADS, ACTION_HORIZON } from '../types/constants';
import type { SampledLayer } from '../types/api';

/** Dropdowns for layer, head, and action index selection. */
export function SharedControls() {
  const { layer, head, actionIndex, setLayer, setHead, setActionIndex } = useEpisode();

  return (
    <div className="flex flex-wrap items-end gap-4">
      <div className="flex flex-col">
        <label htmlFor="layer-select" className="text-xs font-medium text-gray-500 mb-1">
          Layer
        </label>
        <select
          id="layer-select"
          value={layer}
          onChange={(e) => setLayer(Number(e.target.value) as SampledLayer)}
          className="rounded border border-gray-300 bg-white px-3 py-1.5 text-sm text-gray-700 shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
        >
          {SAMPLED_LAYERS.map((l) => (
            <option key={l} value={l}>Layer {l}</option>
          ))}
        </select>
      </div>

      <div className="flex flex-col">
        <label htmlFor="head-select" className="text-xs font-medium text-gray-500 mb-1">
          Head <span className="text-gray-400">(Attention View only)</span>
        </label>
        <select
          id="head-select"
          value={head}
          onChange={(e) => setHead(Number(e.target.value))}
          className="rounded border border-gray-300 bg-white px-3 py-1.5 text-sm text-gray-700 shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
        >
          {Array.from({ length: NUM_HEADS }, (_, i) => (
            <option key={i} value={i}>Head {i}</option>
          ))}
        </select>
      </div>

      <div className="flex flex-col">
        <label htmlFor="action-select" className="text-xs font-medium text-gray-500 mb-1">
          Action
        </label>
        <select
          id="action-select"
          value={actionIndex}
          onChange={(e) => setActionIndex(Number(e.target.value))}
          className="rounded border border-gray-300 bg-white px-3 py-1.5 text-sm text-gray-700 shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
        >
          {Array.from({ length: ACTION_HORIZON }, (_, i) => (
            <option key={i} value={i}>Action {i}</option>
          ))}
        </select>
      </div>
    </div>
  );
}
