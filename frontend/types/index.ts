export interface ChunkMetadata {
  source: string;
  score: number;
  preview: string;
}

export interface QueryResponse {
  answer: string;
  retrieved_chunks: ChunkMetadata[];
}

export interface IngestResponse {
  chunks_added: number;
  total_chunks: number;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  chunks?: ChunkMetadata[];
}
