import { Loader2, Upload } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { detect } from "./api";
import { ComparisonPicker } from "./components/ComparisonPicker";
import { ComparisonResults } from "./components/ComparisonResults";
import { ImageStage } from "./components/ImageStage";
import { LatencyBadge } from "./components/LatencyBadge";
import { LicensePanel } from "./components/LicensePanel";
import { PromptChips } from "./components/PromptChips";
import { SceneSelector } from "./components/SceneSelector";
import { WaitIndicator } from "./components/WaitIndicator";
import { useComparison } from "./hooks/useComparison";
import { SCENES } from "./scenes";
import type { DetectResponse } from "./types";

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

  useEffect(() => {
    return () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    };
  }, []);

  const isPending = requestState.status === "pending";

  async function resolveImageBlob(): Promise<Blob> {
    return uploadedFile ?? (await (await fetch(imageUrl)).blob());
  }

  const comparison = useComparison(resolveImageBlob, isPending);
  const anyBusy = isPending || comparison.isComparing;

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
    comparison.resetSelection();
  }

  function handleFileUpload(file: File) {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    const url = URL.createObjectURL(file);
    objectUrlRef.current = url;
    setUploadedFile(file);
    setImageUrl(url);
    setRequestState({ status: "idle" });
    comparison.clearResults();
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
    <div className="mx-auto flex max-w-3xl flex-col gap-10 px-4 py-10 sm:px-6 sm:py-14">
      <header className="flex flex-col items-center gap-3 text-center">
        <h1 className="text-2xl tracking-tight sm:text-3xl">
          LocateAnything-3B — Interactive Counting Demo
        </h1>
        <p className="max-w-md text-sm text-text sm:text-base">
          Open-vocabulary detection: type or click any target, no retraining.
        </p>
      </header>

      <main className="flex flex-col gap-8">
        <SceneSelector
          scenes={SCENES}
          selectedId={selectedSceneId}
          disabled={anyBusy}
          onSelect={handleSceneSelect}
        />

        <ImageStage imageUrl={imageUrl} detections={detections} />

        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <label
            htmlFor="upload"
            className="inline-flex w-fit cursor-pointer items-center gap-2 rounded-lg border border-border px-3 py-2 text-sm font-medium text-text-h transition-colors hover:border-accent hover:text-accent has-[input:disabled]:cursor-not-allowed has-[input:disabled]:opacity-50"
          >
            <Upload size={16} />
            Upload your own image
            <input
              id="upload"
              type="file"
              accept="image/*"
              disabled={anyBusy}
              className="hidden"
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

          <div className="flex flex-col gap-2 sm:flex-row">
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="What should the model look for?"
              disabled={anyBusy}
              className="min-w-0 flex-1 rounded-lg border border-border bg-bg px-3 py-2.5 text-sm text-text-h placeholder:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={anyBusy || !prompt.trim()}
              className="inline-flex items-center justify-center whitespace-nowrap rounded-lg bg-accent px-5 py-2.5 text-sm font-medium text-white transition-colors hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isPending && <Loader2 size={16} className="mr-1.5 animate-spin" />}
              {isPending ? "Detecting..." : "Detect"}
            </button>
          </div>
        </form>

        {requestState.status === "pending" && <WaitIndicator startedAt={requestState.startedAt} />}

        {requestState.status === "error" && (
          <p className="rounded-lg border border-danger px-3 py-2 text-sm text-danger" role="alert">
            {requestState.message}
          </p>
        )}

        {requestState.status === "done" && (
          <div className="flex flex-wrap items-center gap-3 text-sm text-text-h">
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

        <section className="mt-2 flex flex-col gap-4 border-t border-border pt-8">
          <h2 className="text-lg">Compare prompts side by side</h2>
          <p className="text-sm text-text">
            Pick 2-3 prompts to run against the same image and see the model swap targets.
          </p>

          <ComparisonPicker
            availablePrompts={scene.default_prompts}
            selected={comparison.prompts}
            freeText={comparison.freeText}
            hint={comparison.hint}
            disabled={anyBusy}
            onToggleChip={comparison.toggleChip}
            onFreeTextChange={comparison.setFreeText}
            onAddFreeText={comparison.addFreeText}
            onRemove={comparison.remove}
          />

          <button
            type="button"
            onClick={comparison.run}
            disabled={anyBusy || comparison.prompts.length === 0}
            className="inline-flex w-fit items-center justify-center whitespace-nowrap rounded-lg border border-accent px-5 py-2.5 text-sm font-medium text-accent transition-colors hover:bg-accent-bg disabled:cursor-not-allowed disabled:opacity-50"
          >
            {comparison.isComparing && <Loader2 size={16} className="mr-1.5 animate-spin" />}
            {comparison.isComparing ? "Comparing..." : `Compare (${comparison.prompts.length})`}
          </button>

          {comparison.prompts.length > 0 && Object.keys(comparison.results).length > 0 && (
            <ComparisonResults
              imageUrl={imageUrl}
              prompts={comparison.prompts}
              results={comparison.results}
              onRetry={comparison.retry}
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
