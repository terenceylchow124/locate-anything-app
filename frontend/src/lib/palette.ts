// One stable color per prompt (category). The same color is used for the
// prompt tag, the result-card border/title, and that category's boxes in the
// combined overlay, so a category is recognizable at a glance across the UI.
export const PALETTE = ["#ef4444", "#f59e0b", "#22c55e", "#3b82f6", "#a855f7"];

export function colorForIndex(i: number): string {
  return PALETTE[i % PALETTE.length];
}

export function colorForPrompt(prompts: string[], prompt: string): string {
  return colorForIndex(prompts.indexOf(prompt));
}
