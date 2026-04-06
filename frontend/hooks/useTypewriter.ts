"use client";

import { useState, useEffect, useRef } from "react";

const CHARS_PER_TICK = 4;
const TICK_MS = 18;

/**
 * Returns a version of `text` that "types itself out" character by character
 * while `active` is true. Once `active` becomes false the full text is shown
 * immediately so finished messages never lag.
 */
export function useTypewriter(text: string, active: boolean): string {
  const [pos, setPos] = useState(0);
  const textRef = useRef(text);
  textRef.current = text;

  // Reset position when a new streaming message starts (text goes "" → growing)
  useEffect(() => {
    if (active) setPos(0);
  }, [active]);

  useEffect(() => {
    if (!active) return;

    const id = setInterval(() => {
      setPos((p) => {
        const target = textRef.current.length;
        return p + CHARS_PER_TICK >= target ? target : p + CHARS_PER_TICK;
      });
    }, TICK_MS);

    return () => clearInterval(id);
  }, [active]);

  if (!active) return text;
  return text.slice(0, pos);
}
