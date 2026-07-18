import { ChevronDown } from "lucide-react";
import type { Scene } from "../types";

interface Props {
  scenes: Scene[];
  selectedId: string;
  disabled: boolean;
  onSelect: (id: string) => void;
}

export function SceneSelector({ scenes, selectedId, disabled, onSelect }: Props) {
  return (
    <div className="flex items-center gap-3">
      <label htmlFor="scene-select" className="text-sm font-medium text-text">
        Scene
      </label>
      <div className="relative flex-1">
        <select
          id="scene-select"
          value={selectedId}
          disabled={disabled}
          onChange={(e) => onSelect(e.target.value)}
          className="w-full appearance-none rounded-lg border border-border bg-bg py-2 pl-3 pr-9 text-sm text-text-h transition-colors hover:border-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg disabled:cursor-not-allowed disabled:opacity-50"
        >
          {scenes.map((scene) => (
            <option key={scene.id} value={scene.id}>
              {scene.display_name}
            </option>
          ))}
        </select>
        <ChevronDown
          size={16}
          className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-text"
        />
      </div>
    </div>
  );
}
