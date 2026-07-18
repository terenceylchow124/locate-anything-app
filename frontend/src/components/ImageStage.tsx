import { useState } from "react";
import type { Detection } from "../types";

interface Props {
  imageUrl: string;
  detections: Detection[];
}

export function ImageStage({ imageUrl, detections }: Props) {
  const [naturalSize, setNaturalSize] = useState<{ width: number; height: number } | null>(null);

  return (
    <div className="relative w-full overflow-hidden rounded-xl border border-border leading-none">
      <img
        src={imageUrl}
        alt="scene to detect objects in"
        className="block h-auto w-full"
        onLoad={(e) =>
          setNaturalSize({
            width: e.currentTarget.naturalWidth,
            height: e.currentTarget.naturalHeight,
          })
        }
      />
      {naturalSize && (
        <svg
          className="absolute inset-0 h-full w-full"
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
                  className="fill-none stroke-success stroke-[3]"
                  vectorEffect="non-scaling-stroke"
                />
                <text
                  x={x1}
                  y={Math.max(0, y1 - 4)}
                  className="fill-success stroke-bg text-base font-semibold stroke-[3px]"
                  style={{ paintOrder: "stroke" }}
                >
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
