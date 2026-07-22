import { useState } from "react";
import type { Detection } from "../types";

// A detection may carry its own color (combined overlay, one color per
// category); otherwise the canvas-level strokeColor is used (result cards).
type ColoredDetection = Detection & { color?: string };

interface Props {
  imageUrl: string;
  detections: ColoredDetection[];
  /** Box outline + label color when a detection has no explicit color. */
  strokeColor?: string;
  /** Box outline thickness in display px (non-scaling stroke). */
  strokeWidth?: number;
}

// Always fits its parent (object-contain) so the page never scrolls: the <img>
// uses object-contain and the SVG overlay uses the matching preserveAspectRatio
// "meet", so boxes stay aligned with the letterboxed image at any size. The
// parent MUST provide a bounded height (flex-1 min-h-0 / h-full).
export function ImageStage({
  imageUrl,
  detections,
  strokeColor = "#22c55e",
  strokeWidth = 3,
}: Props) {
  const [naturalSize, setNaturalSize] = useState<{ width: number; height: number } | null>(null);

  return (
    <div className="relative h-full w-full overflow-hidden rounded-xl border border-border bg-bg">
      <img
        src={imageUrl}
        alt="scene to detect objects in"
        className="absolute inset-0 h-full w-full object-contain"
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
          preserveAspectRatio="xMidYMid meet"
        >
          {detections.map((d, i) => {
            const [x1, y1, x2, y2] = d.box;
            const color = d.color ?? strokeColor;
            return (
              <g key={i}>
                <rect
                  x={x1}
                  y={y1}
                  width={Math.max(0, x2 - x1)}
                  height={Math.max(0, y2 - y1)}
                  fill="none"
                  stroke={color}
                  strokeWidth={strokeWidth}
                  vectorEffect="non-scaling-stroke"
                />
                <text
                  x={x1}
                  y={Math.max(0, y1 - 4)}
                  fill={color}
                  className="text-base font-semibold"
                  style={{ paintOrder: "stroke", stroke: "var(--bg)", strokeWidth: 3 }}
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
