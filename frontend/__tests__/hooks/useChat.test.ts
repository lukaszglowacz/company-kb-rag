import { renderHook, act } from "@testing-library/react";
import { useChat } from "@/hooks/useChat";
import type { QueryResponse } from "@/types";

const makeQueryFn =
  (response: QueryResponse) => () => Promise.resolve(response);

const makeFailingQueryFn = (message: string) => () =>
  Promise.reject(new Error(message));

const defaultResponse: QueryResponse = {
  answer: "42",
  retrieved_chunks: [
    { source: "doc.txt", score: 0.9, preview: "the answer" },
  ],
};

describe("useChat", () => {
  it("starts with empty messages and no error", () => {
    const { result } = renderHook(() =>
      useChat(makeQueryFn(defaultResponse)),
    );
    expect(result.current.messages).toEqual([]);
    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("appends user and assistant messages on success", async () => {
    const { result } = renderHook(() =>
      useChat(makeQueryFn(defaultResponse)),
    );
    await act(async () => {
      await result.current.sendMessage("What is the answer?");
    });
    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages[0].role).toBe("user");
    expect(result.current.messages[0].content).toBe("What is the answer?");
    expect(result.current.messages[1].role).toBe("assistant");
    expect(result.current.messages[1].content).toBe("42");
  });

  it("attaches retrieved chunks to the assistant message", async () => {
    const { result } = renderHook(() =>
      useChat(makeQueryFn(defaultResponse)),
    );
    await act(async () => {
      await result.current.sendMessage("question");
    });
    expect(result.current.messages[1].chunks).toEqual(
      defaultResponse.retrieved_chunks,
    );
  });

  it("sets error and keeps isLoading false on failure", async () => {
    const { result } = renderHook(() =>
      useChat(makeFailingQueryFn("network error")),
    );
    await act(async () => {
      await result.current.sendMessage("bad question");
    });
    expect(result.current.error).toBe("network error");
    expect(result.current.isLoading).toBe(false);
  });

  it("clears a previous error on the next successful send", async () => {
    let fail = true;
    const queryFn = () =>
      fail
        ? Promise.reject(new Error("oops"))
        : Promise.resolve(defaultResponse);
    const { result } = renderHook(() => useChat(queryFn));

    await act(async () => {
      await result.current.sendMessage("first");
    });
    expect(result.current.error).toBe("oops");

    fail = false;
    await act(async () => {
      await result.current.sendMessage("second");
    });
    expect(result.current.error).toBeNull();
  });

  it("assigns unique ids to each message", async () => {
    const { result } = renderHook(() =>
      useChat(makeQueryFn(defaultResponse)),
    );
    await act(async () => {
      await result.current.sendMessage("ping");
    });
    const [user, assistant] = result.current.messages;
    expect(user.id).toBeTruthy();
    expect(assistant.id).toBeTruthy();
    expect(user.id).not.toBe(assistant.id);
  });
});
