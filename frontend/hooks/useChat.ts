"use client";

import { useState, useCallback } from "react";
import { query as defaultQuery } from "@/lib/api";
import type { Message, QueryResponse } from "@/types";

type QueryFn = (text: string) => Promise<QueryResponse>;

interface UseChatReturn {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
  sendMessage: (text: string) => Promise<void>;
}

export function useChat(queryFn: QueryFn = defaultQuery): UseChatReturn {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendMessage = useCallback(
    async (text: string) => {
      const userMessage: Message = {
        id: crypto.randomUUID(),
        role: "user",
        content: text,
      };

      setMessages((prev: Message[]) => [...prev, userMessage]);
      setIsLoading(true);
      setError(null);

      try {
        const response = await queryFn(text);
        const assistantMessage: Message = {
          id: crypto.randomUUID(),
          role: "assistant",
          content: response.answer,
          chunks: response.retrieved_chunks,
        };
        setMessages((prev: Message[]) => [...prev, assistantMessage]);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
      } finally {
        setIsLoading(false);
      }
    },
    [queryFn],
  );

  return { messages, isLoading, error, sendMessage };
}
