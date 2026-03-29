const LANGUAGE_TOKEN_INDEX_BASE = 768;

interface LanguageAttentionProps {
  weights: number[];
  tokens: string[];
  highlightedIndex: number | null;
  onHover: (idx: number | null) => void;
}

export function LanguageAttention({
  weights,
  tokens,
  highlightedIndex,
  onHover,
}: LanguageAttentionProps) {
  const n = Math.min(tokens.length, weights.length);
  const sliceWeights = weights.slice(0, n);
  const maxWeight = sliceWeights.length > 0 ? Math.max(...sliceWeights) : 0;

  return (
    <p className="mt-2 flex flex-wrap gap-x-1 gap-y-1 leading-relaxed">
      {tokens.slice(0, n).map((tok, i) => {
        const w = sliceWeights[i] ?? 0;
        const alpha = maxWeight > 0 ? (w / maxWeight) * 0.85 : 0;
        const tokenIndex = LANGUAGE_TOKEN_INDEX_BASE + i;
        const isHighlighted = highlightedIndex === tokenIndex;
        return (
          <span
            key={`${i}-${tok}`}
            className={`cursor-default rounded px-0.5 transition-shadow ${
              isHighlighted ? 'ring-2 ring-orange-400 animate-pulse' : ''
            }`}
            style={{ backgroundColor: `rgba(196, 18, 48, ${alpha})` }}
            onMouseEnter={() => onHover(tokenIndex)}
            onMouseLeave={() => onHover(null)}
          >
            {tok}
          </span>
        );
      })}
    </p>
  );
}
