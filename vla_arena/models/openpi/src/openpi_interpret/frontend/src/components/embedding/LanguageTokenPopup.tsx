export interface LanguageTokenPopupProps {
  tokenText: string;
  position: number;
  allTokens: string[];
}

/** Full instruction with the hovered token emphasized (below t-SNE tooltip). */
export function LanguageTokenPopup({ tokenText, position, allTokens }: LanguageTokenPopupProps) {
  return (
    <div
      className="mt-2 max-w-md border-t border-gray-600 pt-2"
      aria-label={`Instruction context; focused token: ${tokenText}`}
    >
      <p className="mb-1 text-[10px] font-medium uppercase tracking-wide text-gray-400">
        Instruction
      </p>
      <p className="flex flex-wrap gap-x-0.5 gap-y-1 leading-relaxed">
        {allTokens.map((tok, i) => {
          const active = i === position;
          return (
            <span
              key={`${i}-${tok}`}
              className={
                active
                  ? 'rounded px-0.5 bg-orange-500 font-medium text-white'
                  : 'rounded px-0.5 bg-gray-700 text-gray-300'
              }
            >
              {tok}
            </span>
          );
        })}
      </p>
    </div>
  );
}
