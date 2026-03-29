import { api } from '../../api/client';
import type { CameraName } from '../../types/api';

const THUMB_SIZE = 128;
const PATCH_PX = THUMB_SIZE / 16;

export interface ImagePatchPopupProps {
  episodeId: string;
  cameraName: CameraName;
  patchRow: number;
  patchCol: number;
  timestep: number;
}

/** Camera thumbnail with dim overlay and patch highlight (below t-SNE tooltip). */
export function ImagePatchPopup({
  episodeId,
  cameraName,
  patchRow,
  patchCol,
  timestep,
}: ImagePatchPopupProps) {
  const left = patchCol * PATCH_PX;
  const top = patchRow * PATCH_PX;

  return (
    <div className="mt-2 border-t border-gray-600 pt-2">
      <p className="mb-1 text-[10px] font-medium uppercase tracking-wide text-gray-400">
        {cameraName.replace(/_/g, ' ')} · patch ({patchRow}, {patchCol})
      </p>
      <div
        className="relative overflow-hidden rounded border border-gray-500"
        style={{ width: THUMB_SIZE, height: THUMB_SIZE }}
      >
        <img
          src={api.getCameraImageUrl(episodeId, cameraName, timestep)}
          alt={cameraName}
          className="h-full w-full object-cover"
          width={THUMB_SIZE}
          height={THUMB_SIZE}
          draggable={false}
        />
        <div className="pointer-events-none absolute inset-0 bg-black/50" />
        <div
          className="pointer-events-none absolute box-border border-2 border-cmu-red"
          style={{
            left,
            top,
            width: PATCH_PX,
            height: PATCH_PX,
          }}
        />
      </div>
    </div>
  );
}
