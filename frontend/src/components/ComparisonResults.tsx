import type { ComparisonPanelState } from "../types";
import { ImageStage } from "./ImageStage";
import { LatencyBadge } from "./LatencyBadge";
import { WaitIndicator } from "./WaitIndicator";

interface Props {
  imageUrl: string;
  prompts: string[];
  results: Record<string, ComparisonPanelState>;
  onRetry: (prompt: string) => void;
  retryDisabled: boolean;
}

export function ComparisonResults({ imageUrl, prompts, results, onRetry, retryDisabled }: Props) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {prompts.map((prompt) => {
        const state: ComparisonPanelState = results[prompt] ?? { status: "pending" };
        return (
          <div key={prompt} className="flex flex-col gap-2 rounded-xl border border-border p-3">
            <h3 className="truncate text-sm font-medium text-text-h">{prompt}</h3>
            <ImageStage
              imageUrl={imageUrl}
              detections={state.status === "done" ? state.result.detections : []}
            />

            {state.status === "pending" && <p className="text-sm text-text">Queued...</p>}
            {state.status === "in-flight" && <WaitIndicator startedAt={state.startedAt} />}

            {state.status === "error" && (
              <div className="flex flex-col gap-2">
                <p role="alert" className="text-sm text-danger">
                  {state.message}
                </p>
                <button
                  type="button"
                  onClick={() => onRetry(prompt)}
                  disabled={retryDisabled}
                  className="w-fit rounded-lg border border-border px-3 py-1.5 text-sm font-medium text-text-h transition-colors hover:border-accent disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Retry
                </button>
              </div>
            )}

            {state.status === "done" && (
              <div className="flex flex-wrap items-center gap-2 text-sm">
                {state.result.count === 0 ? (
                  <p>No matches found.</p>
                ) : (
                  <p>
                    <strong>{state.result.count}</strong> match
                    {state.result.count === 1 ? "" : "es"}
                  </p>
                )}
                <LatencyBadge
                  inferenceTimeMs={state.result.inference_time_ms}
                  mode={state.result.mode}
                />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
