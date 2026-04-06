import { renderHook, act } from "@testing-library/react";
import { useChat } from "@/hooks/useChat";
import type { StreamCallbacks } from "@/types";

type QueryStreamFn = (
  question: string,
  callbacks: StreamCallbacks,
) => Promise<void>;

function makeSuccessStream(
  answer: string,
  chunks = [{ source: "doc.txt", score: 0.9, preview: "the answer" }],
): QueryStreamFn {
  return async (_question, callbacks) => {
    callbacks.onChunks(chunks);
    for (const token of answer.split("")) {
      callbacks.onToken(token);
    }
    callbacks.onDone();
  };
}

function makeErrorStream(message: string): QueryStreamFn {
  return async (_question, callbacks) => {
    callbacks.onError(message);
  };
}

describe("useChat", () => {
  it("starts with empty messages and no error", () => {
    const { result } = renderHook(() => useChat(makeSuccessStream("hi")));
    expect(result.current.messages).toEqual([]);
    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("appends user message immediately and assistant message on done", async () => {
    const { result } = renderHook(() => useChat(makeSuccessStream("hello")));
    await act(async () => {
      await result.current.sendMessage("hi");
    });
    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages[0].role).toBe("user");
    expect(result.current.messages[0].content).toBe("hi");
    expect(result.current.messages[1].role).toBe("assistant");
    expect(result.current.messages[1].content).toBe("hello");
  });

  it("builds assistant content token by token", async () => {
    const { result } = renderHook(() => useChat(makeSuccessStream("abc")));
    await act(async () => {
      await result.current.sendMessage("question");
    });
    expect(result.current.messages[1].content).toBe("abc");
  });

  it("attaches chunks to assistant message", async () => {
    const chunks = [{ source: "s.md", score: 0.8, preview: "text" }];
    const { result } = renderHook(() =>
      useChat(makeSuccessStream("answer", chunks)),
    );
    await act(async () => {
      await result.current.sendMessage("q");
    });
    expect(result.current.messages[1].chunks).toEqual(chunks);
  });

  it("clears isStreaming flag after done", async () => {
    const { result } = renderHook(() => useChat(makeSuccessStream("hi")));
    await act(async () => {
      await result.current.sendMessage("q");
    });
    expect(result.current.messages[1].isStreaming).toBe(false);
    expect(result.current.isLoading).toBe(false);
  });

  it("removes assistant message and sets error on stream error", async () => {
    const { result } = renderHook(() =>
      useChat(makeErrorStream("network error")),
    );
    await act(async () => {
      await result.current.sendMessage("q");
    });
    expect(result.current.error).toBe("network error");
    expect(result.current.isLoading).toBe(false);
    expect(result.current.messages).toHaveLength(1);
    expect(result.current.messages[0].role).toBe("user");
  });

  it("clears previous error on next successful send", async () => {
    let fail = true;
    const queryStreamFn: QueryStreamFn = async (_q, cb) => {
      if (fail) cb.onError("oops");
      else { cb.onChunks([]); cb.onToken("ok"); cb.onDone(); }
    };
    const { result } = renderHook(() => useChat(queryStreamFn));

    await act(async () => { await result.current.sendMessage("first"); });
    expect(result.current.error).toBe("oops");

    fail = false;
    await act(async () => { await result.current.sendMessage("second"); });
    expect(result.current.error).toBeNull();
  });
});
