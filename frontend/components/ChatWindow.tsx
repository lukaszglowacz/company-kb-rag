"use client";

import { useRef, useEffect, useState, type FormEvent } from "react";
import { MessageBubble } from "@/components/MessageBubble";
import { ChunksSidebar } from "@/components/ChunksSidebar";
import { PromptHistory } from "@/components/PromptHistory";
import { useChat } from "@/hooks/useChat";
import { usePromptHistory } from "@/hooks/usePromptHistory";
import type { ChunkMetadata } from "@/types";

export function ChatWindow() {
  const { messages, isLoading, error, sendMessage } = useChat();
  const { history, addToHistory } = usePromptHistory();
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  const activeChunks: ChunkMetadata[] =
    [...messages].reverse().find((m) => m.chunks)?.chunks ?? [];

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || isLoading) return;
    setInput("");
    addToHistory(text);
    await sendMessage(text);
  }

  return (
    <div className="flex h-full">
      {/* Main chat column */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          {messages.length === 0 && (
            <div className="flex h-full items-center justify-center text-gray-400 text-sm">
              Ask anything about the company knowledge base.
            </div>
          )}
          {messages.map((m) => (
            <MessageBubble key={m.id} message={m} />
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="rounded-2xl bg-gray-100 px-4 py-3 text-sm text-gray-400">
                Thinking…
              </div>
            </div>
          )}
          {error && (
            <div className="rounded-lg bg-red-50 border border-red-200 px-4 py-3 text-sm text-red-600">
              {error}
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Prompt history chips */}
        <PromptHistory history={history} onSelect={setInput} />

        {/* Input bar */}
        <form
          onSubmit={handleSubmit}
          className="flex gap-2 border-t border-gray-200 bg-white px-4 py-3"
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question…"
            disabled={isLoading}
            className="flex-1 rounded-xl border border-gray-200 bg-gray-50 px-4 py-2 text-sm outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100 disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="rounded-xl bg-blue-600 px-5 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-40 transition-colors"
          >
            Send
          </button>
        </form>
      </div>

      {/* Retrieved chunks sidebar */}
      <ChunksSidebar chunks={activeChunks} />
    </div>
  );
}
