interface Props {
  prompts: string[];
  activePrompt: string;
  disabled: boolean;
  onSelect: (prompt: string) => void;
}

export function PromptChips({ prompts, activePrompt, disabled, onSelect }: Props) {
  return (
    <div className="prompt-chips">
      {prompts.map((prompt) => (
        <button
          key={prompt}
          type="button"
          className={prompt === activePrompt ? "chip chip-active" : "chip"}
          disabled={disabled}
          onClick={() => onSelect(prompt)}
        >
          {prompt}
        </button>
      ))}
    </div>
  );
}
