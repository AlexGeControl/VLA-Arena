import { TOKEN_COLORS } from '../../types/constants';

const LEGEND_ENTRIES: { label: string; color: string }[] = [
  { label: 'Base camera', color: TOKEN_COLORS.base_0_rgb },
  { label: 'Left wrist', color: TOKEN_COLORS.left_wrist_0_rgb },
  { label: 'Right wrist', color: TOKEN_COLORS.right_wrist_0_rgb },
  { label: 'Language', color: TOKEN_COLORS.language },
  { label: 'State', color: TOKEN_COLORS.state },
  { label: 'Action', color: TOKEN_COLORS.action },
];

export function TsneLegend() {
  return (
    <div className="rounded-md border border-gray-200 bg-white/90 p-3 text-sm shadow-sm dark:border-gray-700 dark:bg-gray-900/90">
      <div className="mb-2 font-medium text-gray-700 dark:text-gray-200">Modality</div>
      <ul className="space-y-1.5">
        {LEGEND_ENTRIES.map(({ label, color }) => (
          <li key={label} className="flex items-center gap-2 text-gray-600 dark:text-gray-300">
            <span
              className="h-3 w-3 shrink-0 rounded-full border border-gray-300 dark:border-gray-600"
              style={{ backgroundColor: color }}
              aria-hidden
            />
            <span>{label}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
