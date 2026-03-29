import type { CameraName, TsnePoint } from '../../types/api';
import {
  ACTION_TOKEN_INDEX_BASE,
  CAMERA_NAMES,
  NUM_IMAGE_PATCH_TOKENS,
  PATCHES_PER_CAMERA,
  PATCH_GRID_SIZE,
} from '../../types/constants';
import { ImagePatchPopup } from './ImagePatchPopup';
import { LanguageTokenPopup } from './LanguageTokenPopup';

export interface TsneTooltipProps {
  point: TsnePoint | null;
  position: { x: number; y: number } | null;
  episodeId: string | null;
  timestep: number;
  allTokens: string[];
}

function tooltipText(point: TsnePoint): string {
  switch (point.type) {
    case 'image_patch':
      return `${point.source} patch`;
    case 'language':
      return `token: '${point.source}' (position ${point.index - NUM_IMAGE_PATCH_TOKENS})`;
    case 'state':
      return 'proprioceptive state';
    case 'action':
      return `action token ${point.index - ACTION_TOKEN_INDEX_BASE}`;
    default:
      return `${point.type}: ${point.source}`;
  }
}

function imagePatchLayout(point: TsnePoint): {
  cameraName: CameraName;
  patchRow: number;
  patchCol: number;
} | null {
  if (point.type !== 'image_patch') return null;
  if (point.index < 0 || point.index >= NUM_IMAGE_PATCH_TOKENS) return null;
  const camIdx = Math.floor(point.index / PATCHES_PER_CAMERA);
  if (camIdx < 0 || camIdx >= CAMERA_NAMES.length) return null;
  const local = point.index % PATCHES_PER_CAMERA;
  return {
    cameraName: CAMERA_NAMES[camIdx],
    patchRow: Math.floor(local / PATCH_GRID_SIZE),
    patchCol: local % PATCH_GRID_SIZE,
  };
}

export function TsneTooltip({ point, position, episodeId, timestep, allTokens }: TsneTooltipProps) {
  if (!point || !position) return null;

  const langPos = point.index - NUM_IMAGE_PATCH_TOKENS;
  const patchLayout = imagePatchLayout(point);

  return (
    <div
      className="pointer-events-none fixed z-50 max-w-md rounded border border-gray-600 bg-gray-900 px-2 py-1 text-xs text-gray-100 shadow-lg"
      style={{
        left: position.x + 12,
        top: position.y + 12,
      }}
    >
      <div>{tooltipText(point)}</div>

      {point.type === 'language' && allTokens.length > 0 && (
        <LanguageTokenPopup
          tokenText={point.source}
          position={langPos}
          allTokens={allTokens}
        />
      )}

      {point.type === 'image_patch' && episodeId && patchLayout && (
        <ImagePatchPopup
          episodeId={episodeId}
          cameraName={patchLayout.cameraName}
          patchRow={patchLayout.patchRow}
          patchCol={patchLayout.patchCol}
          timestep={timestep}
        />
      )}
    </div>
  );
}
