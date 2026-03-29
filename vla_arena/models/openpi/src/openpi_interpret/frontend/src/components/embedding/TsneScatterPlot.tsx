import { useMemo } from 'react';
import type { MouseEvent } from 'react';
import type { NeighborResponse, TsnePoint } from '../../types/api';
import { ACTION_TOKEN_INDEX_BASE } from '../../types/constants';

const VIEW_SIZE = 1000;
const PADDING_FRAC = 0.1;

function computeBounds(points: TsnePoint[]) {
  if (points.length === 0) {
    return { xMin: 0, xMax: 1, yMin: 0, yMax: 1 };
  }
  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;
  for (const p of points) {
    xMin = Math.min(xMin, p.x);
    xMax = Math.max(xMax, p.x);
    yMin = Math.min(yMin, p.y);
    yMax = Math.max(yMax, p.y);
  }
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;
  const xPad = xRange * PADDING_FRAC;
  const yPad = yRange * PADDING_FRAC;
  return {
    xMin: xMin - xPad,
    xMax: xMax + xPad,
    yMin: yMin - yPad,
    yMax: yMax + yPad,
  };
}

export interface TsneScatterPlotProps {
  points: TsnePoint[];
  selectedActionIndex: number;
  neighbors: NeighborResponse | null;
  highlightedTokenIndex: number | null;
  onHover: (index: number | null, event: MouseEvent<SVGElement>) => void;
}

export function TsneScatterPlot({
  points,
  selectedActionIndex,
  neighbors,
  highlightedTokenIndex,
  onHover,
}: TsneScatterPlotProps) {
  const { xMin, xMax, yMin, yMax } = useMemo(() => computeBounds(points), [points]);
  const xSpan = xMax - xMin || 1;
  const ySpan = yMax - yMin || 1;

  const mapX = (x: number) => ((x - xMin) / xSpan) * VIEW_SIZE;
  const mapY = (y: number) => VIEW_SIZE - ((y - yMin) / ySpan) * VIEW_SIZE;

  const selectedTokenIndex = ACTION_TOKEN_INDEX_BASE + selectedActionIndex;

  const handleSvgLeave = (e: MouseEvent<SVGSVGElement>) => {
    onHover(null, e);
  };

  const handleCircleLeave = (e: MouseEvent<SVGCircleElement>) => {
    const related = e.relatedTarget;
    if (
      related instanceof Node &&
      e.currentTarget.ownerSVGElement?.contains(related)
    ) {
      return;
    }
    onHover(null, e);
  };

  return (
    <svg
      role="img"
      aria-label="t-SNE embedding of token activations"
      viewBox={`0 0 ${VIEW_SIZE} ${VIEW_SIZE}`}
      className="w-full max-w-full rounded-lg border border-cmu-steel-gray"
      style={{ aspectRatio: '1/1', minWidth: 500, backgroundColor: '#FFFFFF' }}
      onMouseLeave={handleSvgLeave}
    >
      {neighbors && neighbors.neighbors.length > 0 && (
        <g className="neighbor-lines" stroke="rgba(109,110,113,0.6)" strokeWidth={1.5}>
          {neighbors.neighbors.map((n) => (
            <line
              key={n.index}
              x1={mapX(neighbors.selected.x)}
              y1={mapY(neighbors.selected.y)}
              x2={mapX(n.x)}
              y2={mapY(n.y)}
              strokeDasharray="6 4"
            />
          ))}
        </g>
      )}

      <g className="points">
        {points.map((p, index) => {
          const isSelected = index === selectedTokenIndex;
          const isHighlighted = highlightedTokenIndex === index;
          const r = isSelected ? 6 : isHighlighted ? 5 : 3;
          const strokeWhite = isSelected || isHighlighted;
          return (
            <circle
              key={index}
              cx={mapX(p.x)}
              cy={mapY(p.y)}
              r={r}
              fill={p.color}
              opacity={isSelected || isHighlighted ? 1 : 0.6}
              stroke={strokeWhite ? '#000000' : 'none'}
              strokeWidth={strokeWhite ? (isSelected ? 2.5 : 1.5) : 0}
              className="cursor-crosshair"
              onMouseEnter={(e) => onHover(index, e)}
              onMouseLeave={handleCircleLeave}
            />
          );
        })}
      </g>

      {neighbors &&
        neighbors.neighbors.map((n) => (
          <circle
            key={`hl-${n.index}`}
            cx={mapX(n.x)}
            cy={mapY(n.y)}
            r={7}
            fill="none"
            stroke="#000000"
            strokeWidth={2}
            pointerEvents="none"
          />
        ))}
    </svg>
  );
}
