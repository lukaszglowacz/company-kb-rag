import { useRef, useEffect } from "react";
import { MessageBubble } from "@/components/MessageBubble";
import type { Message } from "@/types";

interface Props {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
}

export function ChatMessages({ messages, isLoading, error }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
      {messages.length === 0 && (
        <div className="flex h-full items-center justify-center text-gray-400 text-sm">
          Ask anything about the company knowledge base.
        </div>
      )}
      {messages.map((m) => (
        <MessageBubble key={m.id} message={m} />
      ))}
      {isLoading && !messages.at(-1)?.isStreaming && (
        <div className="flex justify-start">
          <div className="rounded-2xl bg-gray-100 px-4 py-3 text-sm text-gray-400">
            Searching knowledge base…
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
  );
}
