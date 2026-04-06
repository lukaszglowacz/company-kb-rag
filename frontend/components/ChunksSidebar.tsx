"use client";

import { useState } from "react";
import type { ChunkMetadata } from "@/types";

interface Props {
  chunks: ChunkMetadata[];
}

function ChunkList({ chunks }: { chunks: ChunkMetadata[] }) {
  return (
    <ul className="space-y-3">
      {chunks.map((chunk) => (
        <li
          key={`${chunk.source}-${chunk.score}`}
          className="rounded-lg border border-gray-200 bg-white p-3"
        >
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
  );
}

export function ChunksSidebar({ chunks }: Props) {
  const [open, setOpen] = useState(false);

  if (chunks.length === 0) return null;

  return (
    <>
      {/* Desktop sidebar */}
      <aside className="hidden md:flex w-72 shrink-0 flex-col border-l border-gray-200 bg-gray-50">
        <div className="p-4 overflow-y-auto flex-1">
          <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-gray-500">
            Retrieved Chunks
          </h2>
          <ChunkList chunks={chunks} />
        </div>
      </aside>

      {/* Mobile: floating pill button */}
      <button
        onClick={() => setOpen(true)}
        className="md:hidden fixed bottom-20 right-4 z-10 flex items-center gap-1.5 rounded-full bg-blue-600 px-3 py-2 text-xs font-medium text-white shadow-lg active:bg-blue-700"
      >
        <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        Sources
        <span className="rounded-full bg-white/25 px-1.5 py-0.5 text-[10px] font-bold leading-none">
          {chunks.length}
        </span>
      </button>

      {/* Mobile: bottom sheet overlay */}
      {open && (
        <div
          className="md:hidden fixed inset-0 z-20 bg-black/40"
          onClick={() => setOpen(false)}
        >
          <div
            className="absolute bottom-0 left-0 right-0 rounded-t-2xl bg-gray-50 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Handle */}
            <div className="flex justify-center pt-3 pb-1">
              <div className="h-1 w-10 rounded-full bg-gray-300" />
            </div>
            {/* Header */}
            <div className="flex items-center justify-between px-4 pb-3">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-gray-500">
                Retrieved Chunks ({chunks.length})
              </h2>
              <button
                onClick={() => setOpen(false)}
                className="rounded-full p-1.5 text-gray-400 hover:bg-gray-200 hover:text-gray-600"
              >
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            {/* Scrollable list */}
            <div className="max-h-[55vh] overflow-y-auto px-4 pb-6">
              <ChunkList chunks={chunks} />
            </div>
          </div>
        </div>
      )}
    </>
  );
}
