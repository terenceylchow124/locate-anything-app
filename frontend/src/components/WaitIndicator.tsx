import { useEffect, useState } from "react";

interface Props {
  startedAt: number;
}

// Spinner + elapsed-time counter, no fake progress bar -- the engine gives
// no real progress signal, and a fake one would contradict the
// latency-transparency goal (spec Implementation Decisions).
export function WaitIndicator({ startedAt }: Props) {
  const [elapsedSeconds, setElapsedSeconds] = useState(() =>
    Math.floor((Date.now() - startedAt) / 1000)
  );

  useEffect(() => {
    const interval = setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000));
    }, 1000);
    return () => clearInterval(interval);
  }, [startedAt]);

  return (
    <div className="wait-indicator" role="status">
      <span className="spinner" aria-hidden="true" />
      <span>Running on CPU, this can take a few minutes... {elapsedSeconds}s</span>
    </div>
  );
}
