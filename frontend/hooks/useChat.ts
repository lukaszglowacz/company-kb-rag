"use client";

import { useState, useCallback } from "react";
import { queryStream as defaultQueryStream } from "@/lib/api";
import type { ChunkMetadata, Message, StreamCallbacks } from "@/types";

type QueryStreamFn = (
  question: string,
  callbacks: StreamCallbacks,
) => Promise<void>;

interface UseChatReturn {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
  sendMessage: (text: string) => Promise<void>;
}

export function useChat(
  queryStreamFn: QueryStreamFn = defaultQueryStream,
): UseChatReturn {
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

      const assistantId = crypto.randomUUID();
      const assistantMessage: Message = {
        id: assistantId,
        role: "assistant",
        content: "",
        isStreaming: true,
      };

      setMessages((prev) => [...prev, userMessage, assistantMessage]);
      setIsLoading(true);
      setError(null);

      await queryStreamFn(text, {
        onChunks: (chunks: ChunkMetadata[]) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, chunks } : m,
            ),
          );
        },
        onToken: (token: string) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, content: m.content + token }
                : m,
            ),
          );
        },
        onDone: () => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, isStreaming: false } : m,
            ),
          );
          setIsLoading(false);
        },
        onError: (message: string) => {
          setMessages((prev) => prev.filter((m) => m.id !== assistantId));
          setError(message);
          setIsLoading(false);
        },
      });
    },
    [queryStreamFn],
  );

  return { messages, isLoading, error, sendMessage };
}
