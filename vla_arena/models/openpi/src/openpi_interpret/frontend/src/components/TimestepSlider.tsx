import { useEpisode } from '../context/EpisodeContext';

/** Range slider for selecting the current timestep within an episode. */
export function TimestepSlider() {
  const { meta, timestep, setTimestep } = useEpisode();

  const maxTimestep = meta ? meta.num_timesteps - 1 : 0;

  return (
    <div className="flex items-center gap-3">
      <label htmlFor="timestep-slider" className="text-sm font-medium text-gray-700 whitespace-nowrap">
        Timestep
      </label>
      <input
        id="timestep-slider"
        type="range"
        min={0}
        max={maxTimestep}
        value={timestep}
        onChange={(e) => setTimestep(Number(e.target.value))}
        className="flex-1"
        disabled={!meta}
      />
      <span className="text-sm tabular-nums text-gray-600 w-16 text-right">
        {timestep} / {maxTimestep}
      </span>
    </div>
  );
}
