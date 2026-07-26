import { useEffect, useRef, useState } from "react";
import { ImageStage } from "./components/ImageStage";
import { OverlayView } from "./components/OverlayView";
import { ResultGrid } from "./components/ResultGrid";
import { Sidebar } from "./components/Sidebar";
import { useComparison } from "./hooks/useComparison";
import { collectRows, downloadCSV, downloadJSON } from "./lib/export";
import { colorForPrompt } from "./lib/palette";
import { DEFAULT_SCENE, SCENES } from "./scenes";

function App() {
  const [selectedSceneId, setSelectedSceneId] = useState(DEFAULT_SCENE.id);
  const scene = SCENES.find((s) => s.id === selectedSceneId) ?? DEFAULT_SCENE;

  const [imageUrl, setImageUrl] = useState(scene.default_image);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const objectUrlRef = useRef<string | null>(null);

  const [overlay, setOverlay] = useState(false);

  useEffect(() => {
    return () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    };
  }, []);

  async function resolveImageBlob(): Promise<Blob> {
    return uploadedFile ?? (await (await fetch(imageUrl)).blob());
  }

  const comparison = useComparison(resolveImageBlob, false);
  const prompts = comparison.prompts;
  const colorFor = (p: string) => colorForPrompt(prompts, p);

  const busy = comparison.isComparing;
  const hasResults = Object.keys(comparison.results).length > 0;
  const started = busy || hasResults;

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
    comparison.resetSelection();
  }

  function handleFileUpload(file: File) {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    const url = URL.createObjectURL(file);
    objectUrlRef.current = url;
    setUploadedFile(file);
    setImageUrl(url);
    comparison.clearResults();
  }

  function handleRun() {
    if (busy || prompts.length === 0) return;
    comparison.run();
  }

  function handleExport(kind: "json" | "csv") {
    const rows = collectRows(prompts, comparison.results);
    if (rows.length === 0) return;
    const base = (selectedSceneId || "detections").replace(/[^a-z0-9-_]+/gi, "_");
    if (kind === "json") downloadJSON(rows, `${base}.json`);
    else downloadCSV(rows, `${base}.csv`);
  }

  const exportDisabled = collectRows(prompts, comparison.results).length === 0;

  return (
    <div className="flex h-screen w-full overflow-hidden">
      <Sidebar
        scenes={SCENES}
        selectedSceneId={selectedSceneId}
        onSelectScene={handleSceneSelect}
        onUpload={handleFileUpload}
        disabled={busy}
        availablePrompts={scene.default_prompts}
        selected={prompts}
        freeText={comparison.freeText}
        hint={comparison.hint}
        colorFor={colorFor}
        onToggleChip={comparison.toggleChip}
        onFreeTextChange={comparison.setFreeText}
        onAddFreeText={comparison.addFreeText}
        onRemove={comparison.remove}
        runLabel={`Detect (${prompts.length})`}
        runBusy={busy}
        onRun={handleRun}
        overlay={overlay}
        onToggleOverlay={() => setOverlay((v) => !v)}
        onExportJSON={() => handleExport("json")}
        onExportCSV={() => handleExport("csv")}
        exportDisabled={exportDisabled}
      />

      <main className="flex min-h-0 flex-1 flex-col gap-2 overflow-hidden bg-bg p-4">
        {!started ? (
          <>
            <div className="min-h-0 flex-1">
              <ImageStage imageUrl={imageUrl} detections={[]} />
            </div>
            <p className="shrink-0 text-sm text-text">
              Add prompts in the sidebar and run detection to see results here.
            </p>
          </>
        ) : overlay ? (
          <OverlayView
            imageUrl={imageUrl}
            prompts={prompts}
            results={comparison.results}
            colorFor={colorFor}
          />
        ) : (
          <ResultGrid
            imageUrl={imageUrl}
            prompts={prompts}
            results={comparison.results}
            colorFor={colorFor}
            onRetry={comparison.retry}
            retryDisabled={busy}
          />
        )}
      </main>
    </div>
  );
}

export default App;
