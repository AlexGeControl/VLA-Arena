import { useRef, useEffect, type MouseEvent } from 'react';
import { api } from '../../api/client';
import type { CameraName } from '../../types/api';
import { PATCH_GRID_SIZE } from '../../types/constants';

const DISPLAY_SIZE = 256;
const CELL_PX = DISPLAY_SIZE / PATCH_GRID_SIZE;

interface ImageAttentionProps {
  cameraName: CameraName;
  weights: number[];
  totalWeight: number;
  episodeId: string;
  timestep: number;
  highlightedIndex: number | null;
  cameraOffset: number;
  onHover: (idx: number | null) => void;
}

export function ImageAttention({
  cameraName,
  weights,
  totalWeight,
  episodeId,
  timestep,
  highlightedIndex,
  cameraOffset,
  onHover,
}: ImageAttentionProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const grid = PATCH_GRID_SIZE;
    const off = document.createElement('canvas');
    off.width = grid;
    off.height = grid;
    const octx = off.getContext('2d');
    if (!octx) return;

    const w = weights.length > 0 ? weights : new Array(PATCH_GRID_SIZE * PATCH_GRID_SIZE).fill(0);
    const maxWeight = w.length > 0 ? Math.max(...w) : 0;

    const imgData = octx.createImageData(grid, grid);
    for (let row = 0; row < grid; row++) {
      for (let col = 0; col < grid; col++) {
        const idx = row * grid + col;
        const weight = w[idx] ?? 0;
        const alpha = maxWeight > 0 ? (weight / maxWeight) * 0.85 : 0;
        const p = (row * grid + col) * 4;
        imgData.data[p] = 196;
        imgData.data[p + 1] = 18;
        imgData.data[p + 2] = 48;
        imgData.data[p + 3] = Math.round(alpha * 255);
      }
    }
    octx.putImageData(imgData, 0, 0);

    ctx.clearRect(0, 0, DISPLAY_SIZE, DISPLAY_SIZE);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(off, 0, 0, DISPLAY_SIZE, DISPLAY_SIZE);

    if (highlightedIndex !== null) {
      const local = highlightedIndex - cameraOffset;
      if (local >= 0 && local < PATCH_GRID_SIZE * PATCH_GRID_SIZE) {
        const r = Math.floor(local / PATCH_GRID_SIZE);
        const c = local % PATCH_GRID_SIZE;
        ctx.strokeStyle = '#C41230';
        ctx.lineWidth = 2;
        ctx.strokeRect(c * CELL_PX, r * CELL_PX, CELL_PX, CELL_PX);
      }
    }
  }, [weights, highlightedIndex, cameraOffset]);

  function handleMouseMove(e: MouseEvent<HTMLCanvasElement>) {
    const row = Math.floor(e.nativeEvent.offsetY / CELL_PX);
    const col = Math.floor(e.nativeEvent.offsetX / CELL_PX);
    if (row < 0 || row >= PATCH_GRID_SIZE || col < 0 || col >= PATCH_GRID_SIZE) {
      onHover(null);
      return;
    }
    onHover(cameraOffset + row * PATCH_GRID_SIZE + col);
  }

  function handleMouseLeave() {
    onHover(null);
  }

  return (
    <div className="space-y-1">
      <h4 className="text-xs font-medium text-gray-600">
        {cameraName.replace(/_/g, ' ')}
        <span className="ml-2 text-gray-400">{(totalWeight * 100).toFixed(1)}%</span>
      </h4>
      <div
        className="relative overflow-hidden rounded-md border border-gray-200 shadow-sm"
        style={{ width: DISPLAY_SIZE, height: DISPLAY_SIZE }}
      >
        <img
          src={api.getCameraImageUrl(episodeId, cameraName, timestep)}
          alt={cameraName}
          className="absolute inset-0 h-full w-full object-cover"
          width={DISPLAY_SIZE}
          height={DISPLAY_SIZE}
          draggable={false}
        />
        <canvas
          ref={canvasRef}
          width={DISPLAY_SIZE}
          height={DISPLAY_SIZE}
          className="absolute inset-0 cursor-crosshair"
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        />
      </div>
    </div>
  );
}
