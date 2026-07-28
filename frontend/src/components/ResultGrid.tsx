import { Download } from "lucide-react";
import { useState } from "react";
import type { ComparisonPanelState } from "../types";
import { downloadAnnotatedImage } from "../lib/annotatedImage";
import { ImageStage } from "./ImageStage";
import { LatencyBadge } from "./LatencyBadge";
import { WaitIndicator } from "./WaitIndicator";

interface Props {
  imageUrl: string;
  prompts: string[];
  results: Record<string, ComparisonPanelState>;
  colorFor: (prompt: string) => string;
  onRetry: (prompt: string) => void;
  retryDisabled: boolean;
}

// Grid sized to fit the viewport (no scroll): columns/rows are explicit
// minmax(0,1fr) tracks so cells share the available height; each card's image
// area is flex-1 + min-h-0 so it shrinks to whatever's left after its header /
// threshold / status rows.
export function ResultGrid({ imageUrl, prompts, results, colorFor, onRetry, retryDisabled }: Props) {
  const cols = prompts.length <= 1 ? 1 : prompts.length <= 4 ? 2 : 3;
  const rows = Math.max(1, Math.ceil(prompts.length / cols));

  return (
    <div
      className="grid h-full min-h-0 gap-3"
      style={{
        gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`,
        gridTemplateRows: `repeat(${rows}, minmax(0, 1fr))`,
      }}
    >
      {prompts.map((prompt) => (
        <ResultCard
          key={prompt}
          imageUrl={imageUrl}
          prompt={prompt}
          color={colorFor(prompt)}
          state={results[prompt] ?? { status: "pending" }}
          onRetry={onRetry}
          retryDisabled={retryDisabled}
        />
      ))}
    </div>
  );
}

function ResultCard({
  imageUrl,
  prompt,
  color,
  state,
  onRetry,
  retryDisabled,
}: {
  imageUrl: string;
  prompt: string;
  color: string;
  state: ComparisonPanelState;
  onRetry: (prompt: string) => void;
  retryDisabled: boolean;
}) {
  const [threshold, setThreshold] = useState(0);
  const all = state.status === "done" ? state.result.detections : [];
  const shown = all.filter((d) => (d.score ?? 0) >= threshold);
  const done = state.status === "done";

  return (
    <div
      className="flex min-h-0 min-w-0 flex-col gap-1.5 overflow-hidden rounded-xl border-2 bg-bg p-2"
      style={{ borderColor: color }}
    >
      <div className="flex shrink-0 items-center justify-between gap-2">
        <h3 className="flex min-w-0 items-center gap-1.5 truncate text-xs font-semibold text-text-h">
          <span className="inline-block h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: color }} aria-hidden="true" />
          <span className="truncate" title={prompt}>{prompt}</span>
        </h3>
        {done && <span className="shrink-0 text-xs tabular-nums text-text-h">{shown.length}/{all.length}</span>}
      </div>

      <div className="min-h-0 flex-1">
        <ImageStage imageUrl={imageUrl} detections={shown} strokeColor={color} />
      </div>

      {done && (
        <div className="flex shrink-0 items-center gap-1.5 text-[11px] text-text">
          <span>min</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
            aria-label="Minimum confidence threshold"
            className="h-1 min-w-0 flex-1 accent-[var(--accent)]"
          />
          <span className="w-8 shrink-0 tabular-nums">{threshold.toFixed(2)}</span>
          <button
            type="button"
            onClick={() =>
              downloadAnnotatedImage(imageUrl, shown, color, 3, `${prompt.replace(/[^a-z0-9-_]+/gi, "_") || "image"}.png`)
            }
            className="inline-flex shrink-0 items-center gap-1 rounded-md border border-border px-1.5 py-0.5 font-medium text-text-h transition-colors hover:border-accent hover:text-accent"
            title="Download this card's image"
          >
            <Download size={12} />
          </button>
        </div>
      )}

      {state.status === "pending" && <p className="shrink-0 text-xs text-text">Queued...</p>}
      {state.status === "in-flight" && (
        <div className="shrink-0"><WaitIndicator startedAt={state.startedAt} /></div>
      )}
      {state.status === "error" && (
        <div className="flex shrink-0 flex-col gap-1">
          <p role="alert" className="text-xs text-danger">{state.message}</p>
          <button
            type="button"
            onClick={() => onRetry(prompt)}
            disabled={retryDisabled}
            className="w-fit rounded-lg border border-border px-2 py-1 text-xs font-medium text-text-h transition-colors hover:border-accent disabled:cursor-not-allowed disabled:opacity-50"
          >
            Retry
          </button>
        </div>
      )}
      {done && (
        <div className="flex shrink-0 flex-wrap items-center gap-1.5 text-xs">
          {shown.length === 0 ? (
            <p className="text-text">None above threshold.</p>
          ) : (
            <p className="text-text">
              <strong className="text-text-h">{shown.length}</strong> shown
            </p>
          )}
          <LatencyBadge inferenceTimeMs={state.result.inference_time_ms} mode={state.result.mode} />
        </div>
      )}
    </div>
  );
}
