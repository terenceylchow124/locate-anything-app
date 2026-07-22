import type { ComparisonPanelState } from "../types";

export interface ExportRow {
  prompt: string;
  label: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  // Per-box confidence score -- the model doesn't expose this yet (its custom
  // generate() discards per-token confidence). Left null in the schema so
  // filling it later is a data change, not a contract change.
  score: number | null;
}

/** Flatten all completed prompts' detections into export rows. */
export function collectRows(
  prompts: string[],
  results: Record<string, ComparisonPanelState>,
): ExportRow[] {
  const rows: ExportRow[] = [];
  for (const prompt of prompts) {
    const state = results[prompt];
    if (state?.status !== "done") continue;
    for (const d of state.result.detections) {
      const [x1, y1, x2, y2] = d.box;
      rows.push({ prompt, label: d.label, x1, y1, x2, y2, score: d.score });
    }
  }
  return rows;
}

function triggerDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function downloadJSON(rows: ExportRow[], filename: string) {
  triggerDownload(new Blob([JSON.stringify(rows, null, 2)], { type: "application/json" }), filename);
}

const CSV_HEADER = "prompt,label,x1,y1,x2,y2,score";

function csvEscape(value: string): string {
  return /[",\n]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value;
}

export function downloadCSV(rows: ExportRow[], filename: string) {
  const body = rows
    .map((r) =>
      [csvEscape(r.prompt), csvEscape(r.label), r.x1, r.y1, r.x2, r.y2, r.score ?? ""].join(","),
    )
    .join("\n");
  triggerDownload(new Blob([`${CSV_HEADER}\n${body}`], { type: "text/csv" }), filename);
}
