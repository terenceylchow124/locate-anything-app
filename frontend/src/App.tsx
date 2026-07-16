import { useEffect, useRef, useState } from "react";
import "./App.css";
import { detect } from "./api";
import { ImageStage } from "./components/ImageStage";
import { LatencyBadge } from "./components/LatencyBadge";
import { LicensePanel } from "./components/LicensePanel";
import { PromptChips } from "./components/PromptChips";
import { SceneSelector } from "./components/SceneSelector";
import { WaitIndicator } from "./components/WaitIndicator";
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
  }

  function handleFileUpload(file: File) {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    const url = URL.createObjectURL(file);
    objectUrlRef.current = url;
    setUploadedFile(file);
    setImageUrl(url);
    setRequestState({ status: "idle" });
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!prompt.trim() || isPending) return;

    setRequestState({ status: "pending", startedAt: Date.now() });
    try {
      const imageBlob: Blob = uploadedFile ?? (await (await fetch(imageUrl)).blob());
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
          disabled={isPending}
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
              disabled={isPending}
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFileUpload(file);
              }}
            />
          </label>

          <PromptChips
            prompts={scene.default_prompts}
            activePrompt={prompt}
            disabled={isPending}
            onSelect={setPrompt}
          />

          <div className="prompt-row">
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="What should the model look for?"
              disabled={isPending}
            />
            <button type="submit" disabled={isPending || !prompt.trim()}>
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
      </main>

      <LicensePanel />
    </div>
  );
}

export default App;
