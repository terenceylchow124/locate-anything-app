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
    <div className="comparison-picker">
      <div className="prompt-chips">
        {availablePrompts.map((p) => {
          const isSelected = selected.includes(p);
          return (
            <button
              key={p}
              type="button"
              className={isSelected ? "chip chip-active" : "chip"}
              disabled={disabled || (!isSelected && atCap)}
              onClick={() => onToggleChip(p)}
            >
              {isSelected ? `✓ ${p}` : p}
            </button>
          );
        })}
      </div>

      <div className="prompt-row">
        <input
          type="text"
          value={freeText}
          onChange={(e) => onFreeTextChange(e.target.value)}
          placeholder="Add another prompt to compare"
          disabled={disabled || atCap}
        />
        <button
          type="button"
          onClick={onAddFreeText}
          disabled={disabled || atCap || !freeText.trim()}
        >
          Add
        </button>
      </div>

      {hint && <p className="comparison-hint">{hint}</p>}

      {selected.length > 0 && (
        <ul className="comparison-selected-list">
          {selected.map((p) => (
            <li key={p}>
              {p}
              <button
                type="button"
                onClick={() => onRemove(p)}
                disabled={disabled}
                aria-label={`Remove ${p} from comparison`}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
