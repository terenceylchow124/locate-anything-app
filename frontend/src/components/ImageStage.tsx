import { useState } from "react";
import type { Detection } from "../types";

interface Props {
  imageUrl: string;
  detections: Detection[];
}

export function ImageStage({ imageUrl, detections }: Props) {
  const [naturalSize, setNaturalSize] = useState<{ width: number; height: number } | null>(null);

  return (
    <div className="image-stage">
      <img
        src={imageUrl}
        alt="scene to detect objects in"
        onLoad={(e) =>
          setNaturalSize({
            width: e.currentTarget.naturalWidth,
            height: e.currentTarget.naturalHeight,
          })
        }
      />
      {naturalSize && (
        <svg
          className="detection-overlay"
          viewBox={`0 0 ${naturalSize.width} ${naturalSize.height}`}
          preserveAspectRatio="none"
        >
          {detections.map((d, i) => {
            const [x1, y1, x2, y2] = d.box;
            return (
              <g key={i}>
                <rect
                  x={x1}
                  y={y1}
                  width={Math.max(0, x2 - x1)}
                  height={Math.max(0, y2 - y1)}
                  className="detection-box"
                />
                <text x={x1} y={Math.max(0, y1 - 4)} className="detection-label">
                  {d.label}
                </text>
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
}
