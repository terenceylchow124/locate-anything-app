import type { Scene } from "../types";

interface Props {
  scenes: Scene[];
  selectedId: string;
  disabled: boolean;
  onSelect: (id: string) => void;
}

export function SceneSelector({ scenes, selectedId, disabled, onSelect }: Props) {
  return (
    <div className="scene-selector">
      <label htmlFor="scene-select">Scene</label>
      <select
        id="scene-select"
        value={selectedId}
        disabled={disabled}
        onChange={(e) => onSelect(e.target.value)}
      >
        {scenes.map((scene) => (
          <option key={scene.id} value={scene.id}>
            {scene.display_name}
          </option>
        ))}
      </select>
    </div>
  );
}
