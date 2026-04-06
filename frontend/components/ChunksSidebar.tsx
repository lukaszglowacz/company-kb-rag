import type { ChunkMetadata } from "@/types";

interface Props {
  chunks: ChunkMetadata[];
}

export function ChunksSidebar({ chunks }: Props) {
  if (chunks.length === 0) return null;

  return (
    <aside className="w-72 shrink-0 border-l border-gray-200 bg-gray-50 p-4 overflow-y-auto">
      <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-gray-500">
        Retrieved chunks
      </h2>
      <ul className="space-y-3">
        {chunks.map((chunk) => (
          <li key={`${chunk.source}-${chunk.score}`} className="rounded-lg border border-gray-200 bg-white p-3">
            <div className="mb-1 flex items-center justify-between">
              <span className="text-xs font-medium text-blue-600">
                {chunk.source}
              </span>
              <span className="text-xs text-gray-400">
                {(chunk.score * 100).toFixed(0)}%
              </span>
            </div>
            <p className="text-xs text-gray-600 leading-relaxed line-clamp-3">
              {chunk.preview}
            </p>
          </li>
        ))}
      </ul>
    </aside>
  );
}
