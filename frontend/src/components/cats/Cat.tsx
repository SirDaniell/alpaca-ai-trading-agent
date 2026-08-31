import { cn } from "@/lib/utils";
import { BEHAVIORS, CATS, type CatBehavior, type CatVariant } from "./catSprites";
import type { CSSProperties } from "react";

export type CatProps = {
  variant?: CatVariant;
  behavior?: CatBehavior;
  /** Rendered footprint in px (the cat is drawn inside a square cell) */
  size?: number;
  direction?: "left" | "right";
  /** Animation speed multiplier, 1 = the cat's natural tempo */
  speed?: number;
  className?: string;
  onClick?: () => void;
  title?: string;
};

/**
 * A single sprite-sheet cat. Purely presentational: it plays whatever
 * behavior it is handed, on a CSS steps() animation.
 */
export function Cat({
  variant = "marmalade",
  behavior = "idle",
  size = 96,
  direction = "right",
  speed = 1,
  className,
  onClick,
  title,
}: CatProps) {
  const cat = CATS[variant];
  const spec = BEHAVIORS[behavior];
  const duration = spec.duration / (speed * cat.tempo);
  const motionDuration = (spec.motionDuration ?? spec.duration) / (speed * cat.tempo);
  const flip = direction === "left" ? -1 : 1;

  const style = {
    "--cat-size": `${size}px`,
    "--cat-frames": spec.frames,
    "--cat-flip": flip,
    "--cat-frame-duration": `${duration}s`,
    "--cat-motion": spec.motion ?? "none",
    "--cat-motion-duration": `${motionDuration}s`,
    width: size,
    height: size,
    backgroundImage: `url(${cat.sheet})`,
    backgroundSize: `${size * 4}px ${size * 4}px`,
    backgroundPositionY: `${-spec.row * size}px`,
    animationDuration: spec.motion ? undefined : `${duration}s`,
    transform: spec.motion ? undefined : `scaleX(${flip})`,
  } as CSSProperties;

  return (
    <div
      className={cn(
        "cat-sprite pointer-events-auto select-none",
        spec.motion && "cat-motion",
        onClick && "cursor-pointer",
        className,
      )}
      style={style}
      onClick={onClick}
      role={onClick ? "button" : "img"}
      aria-label={title ?? `${cat.name} the cat, ${spec.label.toLowerCase()}`}
    />
  );

}
