import type { ChunkMetadata, IngestResponse, QueryResponse, StreamCallbacks } from "@/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function request<T>(path: string, init: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${await res.text()}`);
  }
  return res.json() as Promise<T>;
}

export function query(question: string): Promise<QueryResponse> {
  return request<QueryResponse>("/query", {
    method: "POST",
    body: JSON.stringify({ question }),
  });
}

export async function queryStream(
  question: string,
  callbacks: StreamCallbacks,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/query/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
  } catch (err) {
    console.error(err);
    callbacks.onError("Network error — could not reach the server.");
    return;
  }

  if (!res.ok || !res.body) {
    callbacks.onError(`API error ${res.status}`);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const events = buffer.split("\n\n");
      buffer = events.pop() ?? "";

      for (const event of events) {
        const typeLine = event.split("\n").find((l) => l.startsWith("event:"));
        const dataLine = event.split("\n").find((l) => l.startsWith("data:"));
        if (!typeLine || !dataLine) continue;

        const type = typeLine.slice("event:".length).trim();
        const data = dataLine.slice("data:".length).trim();

        if (type === "chunks") {
          callbacks.onChunks(JSON.parse(data) as ChunkMetadata[]);
        } else if (type === "token") {
          callbacks.onToken(JSON.parse(data) as string);
        } else if (type === "done") {
          callbacks.onDone();
        }
      }
    }
  } catch (err) {
    console.error(err);
    callbacks.onError("Stream interrupted unexpectedly.");
  }
}

export function ingest(text: string, source: string): Promise<IngestResponse> {
  return request<IngestResponse>("/ingest", {
    method: "POST",
    body: JSON.stringify({ text, source }),
  });
}
