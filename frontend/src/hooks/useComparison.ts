import { useState } from "react";
import { detect } from "../api";
import type { ComparisonPanelState } from "../types";

/**
 * Owns all state/logic for the side-by-side comparison feature (ticket #06)
 * -- kept out of App.tsx so it doesn't get tangled with the single-detect
 * flow (ticket #04), which changes for unrelated reasons.
 */
export function useComparison(resolveImageBlob: () => Promise<Blob>, blockedByOther: boolean) {
  const [prompts, setPrompts] = useState<string[]>([]);
  const [freeText, setFreeText] = useState("");
  const [hint, setHint] = useState<string | null>(null);
  const [results, setResults] = useState<Record<string, ComparisonPanelState>>({});
  const [isComparing, setIsComparing] = useState(false);

  function toggleChip(p: string) {
    // Chips at cap are rendered disabled (see ComparisonPicker), so this
    // only ever runs for a toggle that's actually allowed -- the cap itself
    // is shown as a persistent message there, not set reactively here.
    setHint(null);
    setPrompts((prev) => (prev.includes(p) ? prev.filter((x) => x !== p) : [...prev, p]));
  }

  function addFreeText() {
    const value = freeText.trim();
    if (!value) return;
    if (prompts.includes(value)) {
      setHint("Already in comparison");
      return;
    }
    // The Add button is disabled at cap (see ComparisonPicker), so reaching
    // here means there's room.
    setPrompts((prev) => [...prev, value]);
    setFreeText("");
    setHint(null);
  }

  function remove(p: string) {
    setPrompts((prev) => prev.filter((x) => x !== p));
    setHint(null);
  }

  // Runs one prompt against a shared image blob: pending -> in-flight ->
  // done/error. Shared by the full comparison run and single-prompt retry so
  // the state-transition shape lives in exactly one place.
  async function runOne(p: string, imageBlob: Blob) {
    setResults((prev) => ({ ...prev, [p]: { status: "in-flight", startedAt: Date.now() } }));
    try {
      const result = await detect(imageBlob, p);
      setResults((prev) => ({ ...prev, [p]: { status: "done", result } }));
    } catch (err) {
      setResults((prev) => ({
        ...prev,
        [p]: { status: "error", message: err instanceof Error ? err.message : "detection failed" },
      }));
    }
  }

  async function run() {
    if (prompts.length === 0 || blockedByOther || isComparing) return;
    setIsComparing(true);

    const initial: Record<string, ComparisonPanelState> = {};
    for (const p of prompts) initial[p] = { status: "pending" };
    setResults(initial);

    let imageBlob: Blob;
    try {
      imageBlob = await resolveImageBlob();
    } catch {
      const errored: Record<string, ComparisonPanelState> = {};
      for (const p of prompts) errored[p] = { status: "error", message: "failed to load image" };
      setResults(errored);
      setIsComparing(false);
      return;
    }

    // Sequential, not parallel: CPU inference is effectively single-worker
    // (see ticket #02b), so parallel calls would just queue behind each
    // other. One prompt failing doesn't cancel the rest of the run.
    for (const p of prompts) {
      await runOne(p, imageBlob);
    }
    setIsComparing(false);
  }

  async function retry(p: string) {
    if (blockedByOther || isComparing) return;
    setIsComparing(true);
    const imageBlob = await resolveImageBlob();
    await runOne(p, imageBlob);
    setIsComparing(false);
  }

  function resetSelection() {
    setPrompts([]);
    setFreeText("");
    setHint(null);
    setResults({});
  }

  function clearResults() {
    setResults({});
  }

  return {
    prompts,
    freeText,
    setFreeText,
    hint,
    results,
    isComparing,
    toggleChip,
    addFreeText,
    remove,
    run,
    retry,
    resetSelection,
    clearResults,
  };
}
