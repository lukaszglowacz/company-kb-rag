import ReactMarkdown from "react-markdown";
import { useTypewriter } from "@/hooks/useTypewriter";
import type { Message } from "@/types";

interface Props {
  message: Message;
}

export function MessageBubble({ message }: Props) {
  const isUser = message.role === "user";
  const displayed = useTypewriter(message.content, message.isStreaming ?? false);

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
          isUser
            ? "bg-blue-600 text-white"
            : "bg-gray-100 text-gray-900"
        }`}
      >
        {isUser ? (
          message.content
        ) : (
          <ReactMarkdown
            components={{
              h1: ({ children }) => <p className="font-bold text-base mb-1">{children}</p>,
              h2: ({ children }) => <p className="font-bold mb-1">{children}</p>,
              h3: ({ children }) => <p className="font-semibold mb-1">{children}</p>,
              strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
              ul: ({ children }) => <ul className="list-disc pl-4 my-1 space-y-0.5">{children}</ul>,
              ol: ({ children }) => <ol className="list-decimal pl-4 my-1 space-y-0.5">{children}</ol>,
              p: ({ children }) => <p className="mb-1 last:mb-0">{children}</p>,
            }}
          >
            {displayed}
          </ReactMarkdown>
        )}
        {message.isStreaming && (
          <span className="inline-block w-1.5 h-4 ml-0.5 bg-current opacity-70 animate-pulse align-middle" />
        )}
      </div>
    </div>
  );
}
