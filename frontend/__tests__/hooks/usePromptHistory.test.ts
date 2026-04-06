import { renderHook, act } from "@testing-library/react";
import { usePromptHistory } from "@/hooks/usePromptHistory";

describe("usePromptHistory", () => {
  it("starts with an empty history", () => {
    const { result } = renderHook(() => usePromptHistory());
    expect(result.current.history).toEqual([]);
  });

  it("adds a prompt to history", () => {
    const { result } = renderHook(() => usePromptHistory());
    act(() => {
      result.current.addToHistory("hello");
    });
    expect(result.current.history).toEqual(["hello"]);
  });

  it("prepends new prompts so the latest appears first", () => {
    const { result } = renderHook(() => usePromptHistory());
    act(() => {
      result.current.addToHistory("first");
      result.current.addToHistory("second");
    });
    expect(result.current.history[0]).toBe("second");
    expect(result.current.history[1]).toBe("first");
  });

  it("deduplicates: re-adding an existing prompt moves it to the front", () => {
    const { result } = renderHook(() => usePromptHistory());
    act(() => {
      result.current.addToHistory("a");
      result.current.addToHistory("b");
      result.current.addToHistory("a");
    });
    expect(result.current.history).toEqual(["a", "b"]);
  });

  it("caps history at 20 entries", () => {
    const { result } = renderHook(() => usePromptHistory());
    act(() => {
      for (let i = 0; i < 25; i++) {
        result.current.addToHistory(`prompt-${i}`);
      }
    });
    expect(result.current.history).toHaveLength(20);
    expect(result.current.history[0]).toBe("prompt-24");
  });
});
