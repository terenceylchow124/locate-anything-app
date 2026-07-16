interface Props {
  inferenceTimeMs: number;
  mode: string;
}

// Surfaces real CPU-only inference latency as part of the demo's story
// (spec: "framed as part of the demo's story ... rather than hidden").
export function LatencyBadge({ inferenceTimeMs, mode }: Props) {
  const seconds = (inferenceTimeMs / 1000).toFixed(1);
  return (
    <span className="latency-badge">
      {seconds}s · {mode} · CPU · no GPU
    </span>
  );
}
