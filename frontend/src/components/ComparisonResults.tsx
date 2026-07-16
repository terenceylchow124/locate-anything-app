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
    <div className="comparison-results">
      {prompts.map((prompt) => {
        const state: ComparisonPanelState = results[prompt] ?? { status: "pending" };
        return (
          <div key={prompt} className="comparison-panel">
            <h3>{prompt}</h3>
            <ImageStage
              imageUrl={imageUrl}
              detections={state.status === "done" ? state.result.detections : []}
            />

            {state.status === "pending" && <p className="comparison-status">Queued...</p>}
            {state.status === "in-flight" && <WaitIndicator startedAt={state.startedAt} />}

            {state.status === "error" && (
              <div className="comparison-error">
                <p role="alert">{state.message}</p>
                <button type="button" onClick={() => onRetry(prompt)} disabled={retryDisabled}>
                  Retry
                </button>
              </div>
            )}

            {state.status === "done" && (
              <div className="comparison-panel-summary">
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
