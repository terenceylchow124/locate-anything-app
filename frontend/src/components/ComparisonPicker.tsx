import { Check, Plus, X } from "lucide-react";

export const MAX_COMPARISON_PROMPTS = 3;

interface Props {
  availablePrompts: string[];
  selected: string[];
  freeText: string;
  hint: string | null;
  disabled: boolean;
  onToggleChip: (prompt: string) => void;
  onFreeTextChange: (value: string) => void;
  onAddFreeText: () => void;
  onRemove: (prompt: string) => void;
}

const CHIP = "rounded-full border px-3.5 py-1.5 text-sm transition-colors disabled:cursor-not-allowed disabled:opacity-50";
const CHIP_INACTIVE = `${CHIP} border-border text-text-h hover:border-accent`;
const CHIP_ACTIVE = `${CHIP} inline-flex items-center gap-1 border-accent bg-accent-bg font-medium text-accent`;

export function ComparisonPicker({
  availablePrompts,
  selected,
  freeText,
  hint,
  disabled,
  onToggleChip,
  onFreeTextChange,
  onAddFreeText,
  onRemove,
}: Props) {
  const atCap = selected.length >= MAX_COMPARISON_PROMPTS;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap gap-2">
        {availablePrompts.map((p) => {
          const isSelected = selected.includes(p);
          return (
            <button
              key={p}
              type="button"
              className={isSelected ? CHIP_ACTIVE : CHIP_INACTIVE}
              disabled={disabled || (!isSelected && atCap)}
              onClick={() => onToggleChip(p)}
            >
              {isSelected && <Check size={14} />}
              {p}
            </button>
          );
        })}
      </div>

      <div className="flex flex-col gap-2 sm:flex-row">
        <input
          type="text"
          value={freeText}
          onChange={(e) => onFreeTextChange(e.target.value)}
          placeholder="Add another prompt to compare"
          disabled={disabled || atCap}
          className="min-w-0 flex-1 rounded-lg border border-border bg-bg px-3 py-2.5 text-sm text-text-h placeholder:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg disabled:opacity-50"
        />
        <button
          type="button"
          onClick={onAddFreeText}
          disabled={disabled || atCap || !freeText.trim()}
          className="inline-flex items-center justify-center gap-1.5 whitespace-nowrap rounded-lg border border-border px-4 py-2.5 text-sm font-medium text-text-h transition-colors hover:border-accent disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Plus size={16} />
          Add
        </button>
      </div>

      {/* atCap is derived/persistent (shown whenever true), unlike `hint` --
          disabled controls can never fire the click/change that would set a
          hint reactively, so the cap message can't be event-triggered. */}
      {atCap ? (
        <p className="text-sm text-danger">Max {MAX_COMPARISON_PROMPTS} prompts selected</p>
      ) : (
        hint && <p className="text-sm text-danger">{hint}</p>
      )}

      {selected.length > 0 && (
        <ul className="flex flex-wrap gap-2">
          {selected.map((p) => (
            <li
              key={p}
              className="flex items-center gap-1.5 rounded-full border border-border py-1 pl-3 pr-1.5 text-sm text-text-h"
            >
              {p}
              <button
                type="button"
                onClick={() => onRemove(p)}
                disabled={disabled}
                aria-label={`Remove ${p} from comparison`}
                className="flex items-center justify-center rounded-full p-1 text-text transition-colors hover:bg-accent-bg hover:text-accent disabled:opacity-50"
              >
                <X size={14} />
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
