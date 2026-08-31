import { useEffect, useRef, useState } from "react";
import { BEHAVIORS, CATS, type CatBehavior, type CatVariant } from "./catSprites";

function pickWeighted(weights: Record<CatBehavior, number>, exclude?: CatBehavior) {
  const entries = (Object.entries(weights) as [CatBehavior, number][]).filter(
    ([key]) => key !== exclude,
  );
  const total = entries.reduce((sum, [, w]) => sum + w, 0);
  let roll = Math.random() * total;
  for (const [key, w] of entries) {
    roll -= w;
    if (roll <= 0) return key;
  }
  return entries[0]?.[0] ?? "idle";
}

/**
 * Autonomous cat brain: picks the next behavior with weighted randomness and
 * personality-tuned dwell times. Rendering-agnostic — pair it with <Cat /> or
 * anything else.
 */
export function useCatBehavior(variant: CatVariant, enabled = true) {
  const cat = CATS[variant];
  const [behavior, setBehavior] = useState<CatBehavior>("idle");
  const timeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;

    const schedule = (current: CatBehavior) => {
      const [min, max] = cat.dwell;
      const spec = BEHAVIORS[current];
      const base = current === "walk" ? max * 1.6 : min + Math.random() * (max - min);
      const ms = Math.max(spec.duration, base) * 1000;
      timeout.current = setTimeout(() => {
        if (cancelled) return;
        const next = pickWeighted(cat.weights, current);
        setBehavior(next);
        schedule(next);
      }, ms);
    };

    schedule(behavior);
    return () => {
      cancelled = true;
      if (timeout.current) clearTimeout(timeout.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [variant, enabled]);

  /** Interrupt with a one-off behavior, then hand control back to the brain. */
  const interrupt = (next: CatBehavior, ms = 900) => {
    if (timeout.current) clearTimeout(timeout.current);
    setBehavior(next);
    timeout.current = setTimeout(() => setBehavior(pickWeighted(cat.weights, next)), ms);
  };

  return { behavior, setBehavior, interrupt };
}
