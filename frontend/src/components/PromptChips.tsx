interface Props {
  prompts: string[];
  activePrompt: string;
  disabled: boolean;
  onSelect: (prompt: string) => void;
}

const CHIP = "rounded-full border px-3.5 py-1.5 text-sm transition-colors disabled:cursor-not-allowed disabled:opacity-50";
const CHIP_INACTIVE = `${CHIP} border-border text-text-h hover:border-accent`;
const CHIP_ACTIVE = `${CHIP} border-accent bg-accent-bg font-medium text-accent`;

export function PromptChips({ prompts, activePrompt, disabled, onSelect }: Props) {
  return (
    <div className="flex flex-wrap gap-2">
      {prompts.map((prompt) => (
        <button
          key={prompt}
          type="button"
          className={prompt === activePrompt ? CHIP_ACTIVE : CHIP_INACTIVE}
          disabled={disabled}
          onClick={() => onSelect(prompt)}
        >
          {prompt}
        </button>
      ))}
    </div>
  );
}
