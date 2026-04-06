interface Props {
  history: string[];
  onSelect: (prompt: string) => void;
}

export function PromptHistory({ history, onSelect }: Props) {
  if (history.length === 0) return null;

  return (
    <div className="border-t border-gray-200 bg-gray-50 px-4 py-2">
      <p className="mb-1 text-xs font-medium text-gray-400">Recent</p>
      <div className="flex gap-2 overflow-x-auto pb-1">
        {history.map((prompt, i) => (
          <button
            key={i}
            onClick={() => onSelect(prompt)}
            className="shrink-0 rounded-full border border-gray-200 bg-white px-3 py-1 text-xs text-gray-600 hover:bg-gray-100 transition-colors"
          >
            {prompt.length > 40 ? `${prompt.slice(0, 40)}…` : prompt}
          </button>
        ))}
      </div>
    </div>
  );
}
