// Mirrors backend/app.py's Pydantic models by hand (see ADR 0002 -- no
// shared schema/codegen is in scope for this project).

export interface Detection {
  label: string;
  box: [number, number, number, number]; // [x1, y1, x2, y2] in image pixels
  score: number | null; // per-box confidence (0..1); null if the engine didn't provide one
}

export interface DetectResponse {
  detections: Detection[];
  count: number;
  inference_time_ms: number;
  mode: string;
  // How long this request waited in the single-worker queue before its own
  // processing started (ticket #02b) -- ~0 when uncontended. Not currently
  // surfaced in the UI, just available on the response.
  queue_wait_ms: number;
}

// Matches the scene-registry shape settled during ticket #04/#05 grilling --
// {id, display_name, category, default_image, default_prompts[],
// expected_count_range} -- so ticket #05 only needs to expand this data
// file, not touch frontend code.
export interface Scene {
  id: string;
  display_name: string;
  category: string;
  default_image: string; // path under /public
  default_prompts: string[];
  expected_count_range: [number, number] | null;
}

// One prompt's state within a side-by-side comparison run (ticket #06).
// "pending" = queued, not yet started; "in-flight" = its /detect call is
// running now; others may already be "done" while this one is still
// pending/in-flight, since the run is sequential (single-worker backend).
export type ComparisonPanelState =
  | { status: "pending" }
  | { status: "in-flight"; startedAt: number }
  | { status: "done"; result: DetectResponse }
  | { status: "error"; message: string };
