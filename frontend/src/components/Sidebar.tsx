import { Download, FileJson, FileText, Layers, LayoutGrid, Play, Upload } from "lucide-react";
import type { ChangeEvent } from "react";
import { PromptTags } from "./PromptTags";
import { SceneSelector } from "./SceneSelector";

interface Props {
  // scene & image
  scenes: import("../types").Scene[];
  selectedSceneId: string;
  onSelectScene: (id: string) => void;
  onUpload: (file: File) => void;
  disabled: boolean;
  // prompts
  availablePrompts: string[];
  selected: string[];
  freeText: string;
  hint: string | null;
  colorFor: (prompt: string) => string;
  onToggleChip: (prompt: string) => void;
  onFreeTextChange: (value: string) => void;
  onAddFreeText: () => void;
  onRemove: (prompt: string) => void;
  // run + view
  runLabel: string;
  runBusy: boolean;
  onRun: () => void;
  overlay: boolean;
  onToggleOverlay: () => void;
  // export
  onExportJSON: () => void;
  onExportCSV: () => void;
  exportDisabled: boolean;
}

// Persistent left control panel. Sized to fit the viewport without scrolling
// (overflow-hidden); the analysis pane on the right does the auto-fitting.
export function Sidebar(p: Props) {
  return (
    <aside className="flex w-72 shrink-0 flex-col gap-4 overflow-hidden border-r border-border bg-bg px-4 py-5">
      <h1 className="shrink-0 text-base font-semibold tracking-tight text-text-h">
        LocateAnything-3B
      </h1>

      <section className="flex shrink-0 flex-col gap-2">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-text">Scene &amp; Image</h2>
        <SceneSelector scenes={p.scenes} selectedId={p.selectedSceneId} disabled={p.disabled} onSelect={p.onSelectScene} />
        <label
          htmlFor="upload"
          className="inline-flex cursor-pointer items-center justify-center gap-2 rounded-lg border border-border px-3 py-2 text-sm font-medium text-text-h transition-colors hover:border-accent hover:text-accent has-[input:disabled]:cursor-not-allowed has-[input:disabled]:opacity-50"
        >
          <Upload size={16} />
          Upload image
          <input
            id="upload"
            type="file"
            accept="image/*"
            disabled={p.disabled}
            className="hidden"
            onChange={(e: ChangeEvent<HTMLInputElement>) => {
              const file = e.target.files?.[0];
              if (file) p.onUpload(file);
            }}
          />
        </label>
      </section>

      <section className="flex min-h-0 flex-col gap-2">
        <h2 className="shrink-0 text-xs font-semibold uppercase tracking-wide text-text">Prompts</h2>
        <div className="min-h-0 overflow-y-auto pr-1">
          <PromptTags
            availablePrompts={p.availablePrompts}
            selected={p.selected}
            freeText={p.freeText}
            hint={p.hint}
            disabled={p.disabled}
            colorFor={p.colorFor}
            onToggleChip={p.onToggleChip}
            onFreeTextChange={p.onFreeTextChange}
            onAddFreeText={p.onAddFreeText}
            onRemove={p.onRemove}
          />
        </div>
      </section>

      <button
        type="button"
        onClick={p.onRun}
        disabled={p.disabled || p.selected.length === 0}
        className="inline-flex shrink-0 items-center justify-center gap-2 rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white transition-colors hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
      >
        <Play size={16} />
        {p.runBusy ? "Detecting..." : p.runLabel}
      </button>

      <section className="flex shrink-0 flex-col gap-2">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-text">View</h2>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => p.overlay && p.onToggleOverlay()}
            disabled={!p.overlay}
            className="inline-flex flex-1 items-center justify-center gap-1.5 rounded-lg border border-border px-3 py-1.5 text-sm font-medium transition-colors disabled:border-accent disabled:bg-accent-bg disabled:text-accent"
          >
            <LayoutGrid size={15} /> Grid
          </button>
          <button
            type="button"
            onClick={() => !p.overlay && p.onToggleOverlay()}
            disabled={p.overlay}
            className="inline-flex flex-1 items-center justify-center gap-1.5 rounded-lg border border-border px-3 py-1.5 text-sm font-medium transition-colors disabled:border-accent disabled:bg-accent-bg disabled:text-accent"
          >
            <Layers size={15} /> Overlay
          </button>
        </div>
      </section>

      <section className="mt-auto flex shrink-0 flex-col gap-2">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-text">Export</h2>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={p.onExportJSON}
            disabled={p.exportDisabled}
            className="inline-flex flex-1 items-center justify-center gap-1.5 rounded-lg border border-border px-3 py-1.5 text-sm font-medium text-text-h transition-colors hover:border-accent disabled:cursor-not-allowed disabled:opacity-50"
          >
            <FileJson size={15} /> JSON
          </button>
          <button
            type="button"
            onClick={p.onExportCSV}
            disabled={p.exportDisabled}
            className="inline-flex flex-1 items-center justify-center gap-1.5 rounded-lg border border-border px-3 py-1.5 text-sm font-medium text-text-h transition-colors hover:border-accent disabled:cursor-not-allowed disabled:opacity-50"
          >
            <FileText size={15} /> CSV
          </button>
        </div>
        <p className="flex items-center gap-1.5 text-xs text-text">
          <Download size={12} /> Label, coordinates &amp; confidence.
        </p>
      </section>
    </aside>
  );
}
