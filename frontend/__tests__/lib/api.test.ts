import { afterEach, describe, expect, it, vi } from "vitest";
import { queryStream } from "@/lib/api";
import type { ChunkMetadata, StreamCallbacks } from "@/types";

function makeCallbacks(): StreamCallbacks & {
  chunks: ChunkMetadata[][];
  tokens: string[];
  doneCount: number;
  errors: string[];
} {
  const chunks: ChunkMetadata[][] = [];
  const tokens: string[] = [];
  const errors: string[] = [];
  let doneCount = 0;
  return {
    chunks,
    tokens,
    errors,
    get doneCount() {
      return doneCount;
    },
    onChunks: (c) => chunks.push(c),
    onToken: (t) => tokens.push(t),
    onDone: () => { doneCount++; },
    onError: (e) => errors.push(e),
  };
}

function makeSSEStream(events: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const event of events) {
        controller.enqueue(encoder.encode(event));
      }
      controller.close();
    },
  });
}

afterEach(() => vi.restoreAllMocks());

describe("queryStream", () => {
  it("calls onError with error message on network failure", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new Error("Failed to fetch")),
    );
    const cb = makeCallbacks();
    await queryStream("test?", cb);
    expect(cb.errors[0]).toBe("Failed to fetch");
    vi.unstubAllGlobals();
  });

  it("calls onError on non-ok HTTP response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 503, body: null }),
    );
    const cb = makeCallbacks();
    await queryStream("test?", cb);
    expect(cb.errors[0]).toContain("503");
    vi.unstubAllGlobals();
  });

  it("dispatches onChunks with parsed metadata", async () => {
    const payload = JSON.stringify([
      { source: "hr.md", score: 0.9, preview: "Leave policy" },
    ]);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        body: makeSSEStream([
          `event: chunks\ndata: ${payload}\n\n`,
          "event: done\ndata: {}\n\n",
        ]),
      }),
    );
    const cb = makeCallbacks();
    await queryStream("vacation?", cb);
    expect(cb.chunks).toHaveLength(1);
    expect(cb.chunks[0][0].source).toBe("hr.md");
    vi.unstubAllGlobals();
  });

  it("dispatches onToken for each token event", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        body: makeSSEStream([
          'event: token\ndata: "Hello"\n\n',
          'event: token\ndata: " world"\n\n',
          "event: done\ndata: {}\n\n",
        ]),
      }),
    );
    const cb = makeCallbacks();
    await queryStream("hi?", cb);
    expect(cb.tokens).toEqual(["Hello", " world"]);
    vi.unstubAllGlobals();
  });

  it("calls onDone exactly once after stream completes", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        body: makeSSEStream(["event: done\ndata: {}\n\n"]),
      }),
    );
    const cb = makeCallbacks();
    await queryStream("test?", cb);
    expect(cb.doneCount).toBe(1);
    vi.unstubAllGlobals();
  });

  it("ignores unknown SSE event types", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        body: makeSSEStream([
          "event: unknown\ndata: something\n\n",
          "event: done\ndata: {}\n\n",
        ]),
      }),
    );
    const cb = makeCallbacks();
    await queryStream("test?", cb);
    expect(cb.errors).toHaveLength(0);
    expect(cb.doneCount).toBe(1);
    vi.unstubAllGlobals();
  });
});
