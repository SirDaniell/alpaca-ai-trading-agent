import { useEffect, useRef, useState } from "react";
import { Cat } from "./Cat";
import { useCatBehavior } from "./useCatBehavior";
import { BEHAVIORS, CATS, type CatBehavior, type CatVariant } from "./catSprites";
import { cn } from "@/lib/utils";

export type CatWalkerProps = {
  variant?: CatVariant;
  size?: number;
  /** Starting position as a percentage of the track width */
  startPercent?: number;
  /** Force a behavior instead of letting the cat decide */
  behavior?: CatBehavior | "auto";
  speed?: number;
  className?: string;
  onCatClick?: (variant: CatVariant) => void;
};

/**
 * Places a <Cat /> on a horizontal track: handles walking, edge turnarounds
 * and mirroring. Used by the playground stage and the ambient page layer.
 */
export function CatWalker({
  variant = "marmalade",
  size = 96,
  startPercent = 20,
  behavior = "auto",
  speed = 1,
  className,
  onCatClick,
}: CatWalkerProps) {
  const auto = behavior === "auto";
  const brain = useCatBehavior(variant, auto);
  const active: CatBehavior = auto ? brain.behavior : behavior;

  const [x, setX] = useState(startPercent);
  const [direction, setDirection] = useState<"left" | "right">("right");
  const xRef = useRef(startPercent);
  const dirRef = useRef<1 | -1>(1);
  const frame = useRef<number | null>(null);
  const trackRef = useRef<HTMLDivElement | null>(null);
  const nodeRef = useRef<HTMLDivElement | null>(null);

  const moving = BEHAVIORS[active].moves;
  const speedRef = useRef(speed);
  speedRef.current = speed;

  useEffect(() => {
    if (!moving) return;
    let last = performance.now();

    const step = (now: number) => {
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      const width = trackRef.current?.offsetWidth ?? 800;
      const pxPerSec = CATS[variant].walkSpeed * speedRef.current * (size / 96);
      const deltaPercent = ((pxPerSec * dt) / width) * 100;
      const maxPercent = Math.max(0, 100 - (size / width) * 100);

      let next = xRef.current + dirRef.current * deltaPercent;
      if (next >= maxPercent) {
        next = maxPercent;
        dirRef.current = -1;
        setDirection("left");
      } else if (next <= 0) {
        next = 0;
        dirRef.current = 1;
        setDirection("right");
      }
      xRef.current = next;
      if (nodeRef.current) nodeRef.current.style.left = `${next}%`;
      frame.current = requestAnimationFrame(step);
    };

    frame.current = requestAnimationFrame(step);
    return () => {
      if (frame.current) cancelAnimationFrame(frame.current);
      setX(xRef.current);
    };
  }, [moving, size, variant]);


  return (
    <div ref={trackRef} className={cn("pointer-events-none relative h-full w-full", className)}>
      <div
        ref={nodeRef}
        className="absolute bottom-0"
        style={{ left: `${x}%`, transition: "none", willChange: "left" }}
        onDoubleClick={() => {
          dirRef.current = dirRef.current === 1 ? -1 : 1;
          setDirection((d) => (d === "right" ? "left" : "right"));
        }}
      >

        <Cat
          variant={variant}
          behavior={active}
          size={size}
          direction={direction}
          speed={speed}
          onClick={() => {
            if (auto) brain.interrupt("pounce", 1100);
            onCatClick?.(variant);
          }}
        />
      </div>
    </div>
  );
}
