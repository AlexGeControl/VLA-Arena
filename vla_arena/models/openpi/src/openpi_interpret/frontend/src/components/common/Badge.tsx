const COLOR_CLASSES: Record<string, string> = {
  gray: 'bg-gray-100 text-gray-700',
  blue: 'bg-blue-100 text-blue-700',
  red: 'bg-red-100 text-red-700',
  orange: 'bg-orange-100 text-orange-700',
  green: 'bg-green-100 text-green-700',
  amber: 'bg-amber-100 text-amber-700',
};

export function Badge({ value, color = 'gray' }: { value: string; color?: string }) {
  const palette = COLOR_CLASSES[color] ?? COLOR_CLASSES.gray;
  return (
    <span
      className={`inline-block rounded px-1.5 py-0.5 font-mono text-xs ${palette}`}
    >
      {value}
    </span>
  );
}
