import type { ChunkMetadata, Message } from "@/types";

export function getLatestChunks(messages: Message[]): ChunkMetadata[] {
  return [...messages].reverse().find((m) => m.chunks)?.chunks ?? [];
}
