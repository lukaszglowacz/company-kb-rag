"use client";

import { type FormEvent } from "react";

interface Props {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (e: FormEvent) => void;
  isLoading: boolean;
}

export function ChatInput({ value, onChange, onSubmit, isLoading }: Props) {
  return (
    <form
      onSubmit={onSubmit}
      className="flex gap-2 border-t border-gray-200 bg-white px-4 py-3"
    >
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Ask a question…"
        disabled={isLoading}
        className="flex-1 rounded-xl border border-gray-200 bg-gray-50 px-4 py-2 text-sm outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100 disabled:opacity-50"
      />
      <button
        type="submit"
        disabled={isLoading || !value.trim()}
        className="rounded-xl bg-blue-600 px-5 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-40 transition-colors"
      >
        Send
      </button>
    </form>
  );
}
