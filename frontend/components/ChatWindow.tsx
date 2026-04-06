"use client";

import { useState, type FormEvent } from "react";
import { ChatMessages } from "@/components/ChatMessages";
import { ChatInput } from "@/components/ChatInput";
import { ChunksSidebar } from "@/components/ChunksSidebar";
import { PromptHistory } from "@/components/PromptHistory";
import { useChat } from "@/hooks/useChat";
import { usePromptHistory } from "@/hooks/usePromptHistory";
import { getLatestChunks } from "@/lib/chunks";

export function ChatWindow() {
  const { messages, isLoading, error, sendMessage } = useChat();
  const { history, addToHistory } = usePromptHistory();
  const [input, setInput] = useState("");

  const activeChunks = getLatestChunks(messages);

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
      <div className="flex flex-1 flex-col overflow-hidden">
        <ChatMessages messages={messages} isLoading={isLoading} error={error} />
        <PromptHistory history={history} onSelect={setInput} />
        <ChatInput
          value={input}
          onChange={setInput}
          onSubmit={handleSubmit}
          isLoading={isLoading}
        />
      </div>
      <ChunksSidebar chunks={activeChunks} />
    </div>
  );
}
