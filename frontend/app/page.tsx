import { ChatWindow } from "@/components/ChatWindow";

export default function Home() {
  return (
    <main className="flex h-screen flex-col">
      <header className="flex items-center gap-3 border-b border-gray-200 px-6 py-3">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-600">
          <span className="text-sm font-bold text-white">KB</span>
        </div>
        <h1 className="text-sm font-semibold text-gray-900">
          Company Knowledge Base
        </h1>
      </header>
      <div className="flex-1 overflow-hidden">
        <ChatWindow />
      </div>
    </main>
  );
}
