interface StateAttentionProps {
  weight: number;
}

export function StateAttention({ weight }: StateAttentionProps) {
  const pct = Math.min(100, Math.max(0, weight * 100));
  return (
    <div className="space-y-2">
      <h3 className="text-sm font-medium uppercase tracking-wide text-gray-500">
        Proprioceptive state{' '}
        <span className="font-normal normal-case text-green-600">{(weight * 100).toFixed(1)}%</span>
      </h3>
      <div className="h-3 w-full overflow-hidden rounded-full bg-gray-100">
        <div className="h-3 rounded-full bg-green-500 transition-all" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
