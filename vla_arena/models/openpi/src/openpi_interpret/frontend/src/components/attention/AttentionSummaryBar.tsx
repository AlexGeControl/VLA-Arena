import type { AttentionBreakdown } from '../../types/api';

const COLORS = {
  images: '#043673',   // CMU Blue Thread
  language: '#FDB515', // CMU Gold Thread
  state: '#009647',    // CMU Green Thread
  action: '#C41230',   // CMU Carnegie Red
} as const;

interface AttentionSummaryBarProps {
  breakdown: AttentionBreakdown;
}

export function AttentionSummaryBar({ breakdown }: AttentionSummaryBarProps) {
  const imagesTotal =
    breakdown.camera_totals.base_0_rgb +
    breakdown.camera_totals.left_wrist_0_rgb +
    breakdown.camera_totals.right_wrist_0_rgb;

  const segments = [
    { key: 'images', share: imagesTotal, color: COLORS.images, label: 'Images' },
    { key: 'language', share: breakdown.language_total, color: COLORS.language, label: 'Language' },
    { key: 'state', share: breakdown.state_weight, color: COLORS.state, label: 'State' },
    { key: 'action', share: breakdown.action_total, color: COLORS.action, label: 'Action' },
  ] as const;

  const denom =
    segments.reduce((s, seg) => s + seg.share, 0) || 1;

  return (
    <div className="flex h-9 w-full overflow-hidden rounded-md border border-gray-200 shadow-sm">
      {segments.map((seg) => {
        const pct = (seg.share / denom) * 100;
        if (pct <= 0) return null;
        return (
          <div
            key={seg.key}
            className={`flex min-w-0 items-center justify-center px-0.5 text-[10px] font-semibold drop-shadow-sm ${seg.key === 'language' ? 'text-cmu-black' : 'text-white'}`}
            style={{ width: `${pct}%`, backgroundColor: seg.color }}
            title={`${seg.label}: ${pct.toFixed(1)}%`}
          >
            {pct >= 6 ? `${pct.toFixed(1)}%` : ''}
          </div>
        );
      })}
    </div>
  );
}
