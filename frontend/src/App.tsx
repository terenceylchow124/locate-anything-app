import { useEffect, useRef, useState } from "react";
import "./App.css";
import { detect } from "./api";
import { ComparisonPicker, MAX_COMPARISON_PROMPTS } from "./components/ComparisonPicker";
import { ComparisonResults } from "./components/ComparisonResults";
import { ImageStage } from "./components/ImageStage";
import { LatencyBadge } from "./components/LatencyBadge";
import { LicensePanel } from "./components/LicensePanel";
import { PromptChips } from "./components/PromptChips";
import { SceneSelector } from "./components/SceneSelector";
import { WaitIndicator } from "./components/WaitIndicator";
import { SCENES } from "./scenes";
import type { ComparisonPanelState, DetectResponse } from "./types";

type RequestState =
  | { status: "idle" }
  | { status: "pending"; startedAt: number }
  | { status: "done"; result: DetectResponse }
  | { status: "error"; message: string };

function App() {
  const [selectedSceneId, setSelectedSceneId] = useState(SCENES[0].id);
  const scene = SCENES.find((s) => s.id === selectedSceneId) ?? SCENES[0];

  const [imageUrl, setImageUrl] = useState(scene.default_image);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState(scene.default_prompts[0] ?? "");
  const [requestState, setRequestState] = useState<RequestState>({ status: "idle" });
  const objectUrlRef = useRef<string | null>(null);

  const [comparisonPrompts, setComparisonPrompts] = useState<string[]>([]);
  const [comparisonFreeText, setComparisonFreeText] = useState("");
  const [comparisonHint, setComparisonHint] = useState<string | null>(null);
  const [comparisonResults, setComparisonResults] = useState<Record<string, ComparisonPanelState>>(
    {}
  );
  const [isComparing, setIsComparing] = useState(false);

  useEffect(() => {
    return () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    };
  }, []);

  const isPending = requestState.status === "pending";
  const anyBusy = isPending || isComparing;

  async function resolveImageBlob(): Promise<Blob> {
    return uploadedFile ?? (await (await fetch(imageUrl)).blob());
  }

  function toggleComparisonChip(p: string) {
    setComparisonHint(null);
    setComparisonPrompts((prev) => {
      if (prev.includes(p)) return prev.filter((x) => x !== p);
      if (prev.length >= MAX_COMPARISON_PROMPTS) {
        setComparisonHint(`Max ${MAX_COMPARISON_PROMPTS} prompts`);
        return prev;
      }
      return [...prev, p];
    });
  }

  function addComparisonFreeText() {
    const value = comparisonFreeText.trim();
    if (!value) return;
    if (comparisonPrompts.includes(value)) {
      setComparisonHint("Already in comparison");
      return;
    }
    if (comparisonPrompts.length >= MAX_COMPARISON_PROMPTS) {
      setComparisonHint(`Max ${MAX_COMPARISON_PROMPTS} prompts`);
      return;
    }
    setComparisonPrompts((prev) => [...prev, value]);
    setComparisonFreeText("");
    setComparisonHint(null);
  }

  function removeComparisonPrompt(p: string) {
    setComparisonPrompts((prev) => prev.filter((x) => x !== p));
    setComparisonHint(null);
  }

  async function runComparison() {
    if (comparisonPrompts.length === 0 || anyBusy) return;
    setIsComparing(true);

    const initial: Record<string, ComparisonPanelState> = {};
    for (const p of comparisonPrompts) initial[p] = { status: "pending" };
    setComparisonResults(initial);

    let imageBlob: Blob;
    try {
      imageBlob = await resolveImageBlob();
    } catch {
      const errored: Record<string, ComparisonPanelState> = {};
      for (const p of comparisonPrompts) {
        errored[p] = { status: "error", message: "failed to load image" };
      }
      setComparisonResults(errored);
      setIsComparing(false);
      return;
    }

    // Sequential, not parallel: CPU inference is effectively single-worker
    // (see ticket #02b), so parallel calls would just queue behind each
    // other. One prompt failing doesn't cancel the rest of the run.
    for (const p of comparisonPrompts) {
      setComparisonResults((prev) => ({ ...prev, [p]: { status: "in-flight", startedAt: Date.now() } }));
      try {
        const result = await detect(imageBlob, p);
        setComparisonResults((prev) => ({ ...prev, [p]: { status: "done", result } }));
      } catch (err) {
        setComparisonResults((prev) => ({
          ...prev,
          [p]: { status: "error", message: err instanceof Error ? err.message : "detection failed" },
        }));
      }
    }
    setIsComparing(false);
  }

  async function retryComparisonPrompt(p: string) {
    if (anyBusy) return;
    setIsComparing(true);
    setComparisonResults((prev) => ({ ...prev, [p]: { status: "in-flight", startedAt: Date.now() } }));
    try {
      const imageBlob = await resolveImageBlob();
      const result = await detect(imageBlob, p);
      setComparisonResults((prev) => ({ ...prev, [p]: { status: "done", result } }));
    } catch (err) {
      setComparisonResults((prev) => ({
        ...prev,
        [p]: { status: "error", message: err instanceof Error ? err.message : "detection failed" },
      }));
    }
    setIsComparing(false);
  }

  function handleSceneSelect(id: string) {
    const nextScene = SCENES.find((s) => s.id === id);
    if (!nextScene) return;
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
    setSelectedSceneId(id);
    setUploadedFile(null);
    setImageUrl(nextScene.default_image);
    setPrompt(nextScene.default_prompts[0] ?? "");
    setRequestState({ status: "idle" });
    setComparisonPrompts([]);
    setComparisonFreeText("");
    setComparisonHint(null);
    setComparisonResults({});
  }

  function handleFileUpload(file: File) {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    const url = URL.createObjectURL(file);
    objectUrlRef.current = url;
    setUploadedFile(file);
    setImageUrl(url);
    setRequestState({ status: "idle" });
    setComparisonResults({});
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!prompt.trim() || anyBusy) return;

    setRequestState({ status: "pending", startedAt: Date.now() });
    try {
      const imageBlob = await resolveImageBlob();
      const result = await detect(imageBlob, prompt);
      setRequestState({ status: "done", result });
    } catch (err) {
      setRequestState({
        status: "error",
        message: err instanceof Error ? err.message : "detection failed",
      });
    }
  }

  const detections = requestState.status === "done" ? requestState.result.detections : [];

  return (
    <div className="app">
      <header>
        <h1>LocateAnything-3B — Interactive Counting Demo</h1>
        <p className="subtitle">
          Open-vocabulary detection: type or click any target, no retraining.
        </p>
      </header>

      <main>
        <SceneSelector
          scenes={SCENES}
          selectedId={selectedSceneId}
          disabled={anyBusy}
          onSelect={handleSceneSelect}
        />

        <ImageStage imageUrl={imageUrl} detections={detections} />

        <form onSubmit={handleSubmit} className="controls">
          <label htmlFor="upload" className="upload-button">
            Upload your own image
            <input
              id="upload"
              type="file"
              accept="image/*"
              disabled={anyBusy}
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFileUpload(file);
              }}
            />
          </label>

          <PromptChips
            prompts={scene.default_prompts}
            activePrompt={prompt}
            disabled={anyBusy}
            onSelect={setPrompt}
          />

          <div className="prompt-row">
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="What should the model look for?"
              disabled={anyBusy}
            />
            <button type="submit" disabled={anyBusy || !prompt.trim()}>
              {isPending ? "Detecting..." : "Detect"}
            </button>
          </div>
        </form>

        {requestState.status === "pending" && <WaitIndicator startedAt={requestState.startedAt} />}

        {requestState.status === "error" && (
          <p className="error" role="alert">
            {requestState.message}
          </p>
        )}

        {requestState.status === "done" && (
          <div className="result-summary">
            {requestState.result.count === 0 ? (
              <p>No matches found for "{prompt}".</p>
            ) : (
              <p>
                Found <strong>{requestState.result.count}</strong> match
                {requestState.result.count === 1 ? "" : "es"} for "{prompt}"
              </p>
            )}
            <LatencyBadge
              inferenceTimeMs={requestState.result.inference_time_ms}
              mode={requestState.result.mode}
            />
          </div>
        )}

        <section className="comparison-section">
          <h2>Compare prompts side by side</h2>
          <p className="comparison-intro">
            Pick 2-3 prompts to run against the same image and see the model swap targets.
          </p>

          <ComparisonPicker
            availablePrompts={scene.default_prompts}
            selected={comparisonPrompts}
            freeText={comparisonFreeText}
            hint={comparisonHint}
            disabled={anyBusy}
            onToggleChip={toggleComparisonChip}
            onFreeTextChange={setComparisonFreeText}
            onAddFreeText={addComparisonFreeText}
            onRemove={removeComparisonPrompt}
          />

          <button
            type="button"
            onClick={runComparison}
            disabled={anyBusy || comparisonPrompts.length === 0}
          >
            {isComparing ? "Comparing..." : `Compare (${comparisonPrompts.length})`}
          </button>

          {comparisonPrompts.length > 0 && Object.keys(comparisonResults).length > 0 && (
            <ComparisonResults
              imageUrl={imageUrl}
              prompts={comparisonPrompts}
              results={comparisonResults}
              onRetry={retryComparisonPrompt}
              retryDisabled={anyBusy}
            />
          )}
        </section>
      </main>

      <LicensePanel />
    </div>
  );
}

export default App;
