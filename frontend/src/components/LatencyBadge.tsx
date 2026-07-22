interface Props {
  inferenceTimeMs: number;
  mode: string;
}

// Surfaces real CPU-only inference latency as part of the demo's story
// (spec: "framed as part of the demo's story ... rather than hidden").
export function LatencyBadge({ inferenceTimeMs, mode }: Props) {
  const seconds = (inferenceTimeMs / 1000).toFixed(1);
  return (
    <span className="inline-flex items-center rounded-full border border-border px-2.5 py-1 text-xs text-text">
      {seconds}s · {mode}
    </span>
  );
}
