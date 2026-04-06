"use client";

import { useState, useCallback } from "react";

interface UsePromptHistoryReturn {
  history: string[];
  addToHistory: (prompt: string) => void;
}

const MAX_HISTORY = 20;

export function usePromptHistory(): UsePromptHistoryReturn {
  const [history, setHistory] = useState<string[]>([]);

  const addToHistory = useCallback((prompt: string) => {
    setHistory((prev) => {
      const deduplicated = prev.filter((p) => p !== prompt);
      return [prompt, ...deduplicated].slice(0, MAX_HISTORY);
    });
  }, []);

  return { history, addToHistory };
}
