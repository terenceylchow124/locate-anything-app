import { useMemo, useState } from "react";
import type { ComparisonPanelState } from "../types";
import { ImageStage } from "./ImageStage";

interface Props {
  imageUrl: string;
  prompts: string[];
  results: Record<string, ComparisonPanelState>;
  colorFor: (prompt: string) => string;
}

// Combined overlay (fits the viewport, no scroll): every completed category's
// boxes on one image, each in its tag color. Click a legend tag to show/hide
// that category; the single slider keeps only boxes at or above a confidence.
export function OverlayView({ imageUrl, prompts, results, colorFor }: Props) {
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [threshold, setThreshold] = useState(0);

  const items = prompts.map((prompt) => {
    const state = results[prompt];
    return {
      prompt,
      color: colorFor(prompt),
      count: state?.status === "done" ? state.result.count : null,
      detections: state?.status === "done" ? state.result.detections : [],
    };
  });
  const doneItems = items.filter((it) => it.count !== null);

  const detections = useMemo(
    () =>
      doneItems
        .filter((it) => !hidden.has(it.prompt))
        .flatMap((it) => it.detections.filter((d) => (d.score ?? 0) >= threshold).map((d) => ({ ...d, color: it.color }))),
    [doneItems, hidden, threshold],
  );

  function toggle(prompt: string) {
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(prompt)) next.delete(prompt);
      else next.add(prompt);
      return next;
    });
  }

  return (
    <div className="flex h-full min-h-0 flex-col gap-2">
      <div className="flex shrink-0 flex-wrap items-center gap-x-3 gap-y-1.5">
        {doneItems.map((it) => {
          const isHidden = hidden.has(it.prompt);
          return (
            <button
              key={it.prompt}
              type="button"
              onClick={() => toggle(it.prompt)}
              className="inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium transition-opacity"
              style={{
                borderColor: it.color,
                color: isHidden ? "var(--text)" : it.color,
                backgroundColor: isHidden ? "transparent" : `${it.color}1a`,
                opacity: isHidden ? 0.45 : 1,
              }}
              aria-pressed={!isHidden}
              title={isHidden ? `Show ${it.prompt}` : `Hide ${it.prompt}`}
            >
              <span
                className="inline-block h-2.5 w-2.5 rounded-full"
                style={{ backgroundColor: isHidden ? "transparent" : it.color, border: `1px solid ${it.color}` }}
                aria-hidden="true"
              />
              <span className={isHidden ? "line-through" : ""}>{it.prompt}</span>
              <span className="tabular-nums">({it.count})</span>
            </button>
          );
        })}
      </div>

      <div className="flex shrink-0 items-center gap-2 text-xs text-text">
        <span>min score</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={threshold}
          onChange={(e) => setThreshold(Number(e.target.value))}
          aria-label="Minimum confidence threshold (overlay)"
          className="h-1 min-w-40 flex-1 accent-[var(--accent)]"
        />
        <span className="w-9 shrink-0 tabular-nums">{threshold.toFixed(2)}</span>
        <span className="ml-2 text-text-h">{detections.length} boxes</span>
      </div>

      <div className="min-h-0 flex-1 overflow-hidden rounded-xl border border-border bg-bg">
        <ImageStage imageUrl={imageUrl} detections={detections} strokeColor="#22c55e" />
      </div>
    </div>
  );
}
