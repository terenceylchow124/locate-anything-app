import { Check, Plus, X } from "lucide-react";

export const MAX_PROMPTS = 5;

interface Props {
  availablePrompts: string[];
  selected: string[];
  freeText: string;
  hint: string | null;
  disabled: boolean;
  colorFor: (prompt: string) => string;
  onToggleChip: (prompt: string) => void;
  onFreeTextChange: (value: string) => void;
  onAddFreeText: () => void;
  onRemove: (prompt: string) => void;
}

// Color-coded prompt management for multi-object comparison. Each selected
// prompt is a colored tag (its color flows to its result card + overlay boxes);
// suggestions are quick-add chips, and free text supports type-and-add.
export function PromptTags({
  availablePrompts,
  selected,
  freeText,
  hint,
  disabled,
  colorFor,
  onToggleChip,
  onFreeTextChange,
  onAddFreeText,
  onRemove,
}: Props) {
  const atCap = selected.length >= MAX_PROMPTS;

  return (
    <div className="flex flex-col gap-3">
      {availablePrompts.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {availablePrompts.map((p) => {
            const isSelected = selected.includes(p);
            const color = colorFor(p);
            return (
              <button
                key={p}
                type="button"
                disabled={disabled || (!isSelected && atCap)}
                onClick={() => onToggleChip(p)}
                className="inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40"
                style={
                  isSelected
                    ? { borderColor: color, color, backgroundColor: `${color}1a` }
                    : { borderColor: "var(--border)", color: "var(--text-h)" }
                }
              >
                {isSelected && <Check size={12} />}
                <span
                  className="inline-block h-2 w-2 rounded-full"
                  style={{ backgroundColor: isSelected ? color : "transparent", border: `1px solid ${color}` }}
                  aria-hidden="true"
                />
                {p}
              </button>
            );
          })}
        </div>
      )}

      <div className="flex gap-2">
        <input
          type="text"
          value={freeText}
          onChange={(e) => onFreeTextChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              onAddFreeText();
            }
          }}
          placeholder="Type a prompt and add"
          disabled={disabled || atCap}
          className="min-w-0 flex-1 rounded-lg border border-border bg-bg px-2.5 py-2 text-sm text-text-h placeholder:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg disabled:opacity-50"
        />
        <button
          type="button"
          onClick={onAddFreeText}
          disabled={disabled || atCap || !freeText.trim()}
          className="inline-flex items-center justify-center gap-1 whitespace-nowrap rounded-lg border border-border px-3 py-2 text-sm font-medium text-text-h transition-colors hover:border-accent disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Plus size={14} />
          Add
        </button>
      </div>

      {selected.length > 0 && (
        <ul className="flex flex-wrap gap-1.5">
          {selected.map((p) => {
            const color = colorFor(p);
            return (
              <li
                key={p}
                className="flex items-center gap-1.5 rounded-full border py-1 pl-2.5 pr-1 text-xs font-medium"
                style={{ borderColor: color, color, backgroundColor: `${color}1a` }}
              >
                <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: color }} aria-hidden="true" />
                {p}
                <button
                  type="button"
                  onClick={() => onRemove(p)}
                  disabled={disabled}
                  aria-label={`Remove ${p}`}
                  className="flex items-center justify-center rounded-full p-0.5 transition-colors hover:bg-black/10 disabled:opacity-50"
                >
                  <X size={12} />
                </button>
              </li>
            );
          })}
        </ul>
      )}

      {atCap ? (
        <p className="text-xs text-danger">Max {MAX_PROMPTS} prompts selected</p>
      ) : (
        hint && <p className="text-xs text-danger">{hint}</p>
      )}
    </div>
  );
}
