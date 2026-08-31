import { CatWalker } from "./CatWalker";
import { CAT_LIST, type CatVariant } from "./catSprites";

export type AmbientCatsProps = {
  /** Which cats wander the page. Defaults to all of them. */
  variants?: CatVariant[];
  size?: number;
  /** Distance from the bottom of the viewport, in px */
  bottom?: number;
};

/**
 * Drop this once anywhere in the app and cats will wander the bottom edge of
 * the page. The layer ignores pointer events; only the cats themselves click.
 */
export function AmbientCats({ variants = CAT_LIST, size = 72, bottom = 0 }: AmbientCatsProps) {
  return (
    <div
      className="pointer-events-none fixed inset-x-0 z-50 h-24"
      style={{ bottom }}
      aria-hidden="true"
    >
      {variants.map((variant, i) => (
        <div key={variant} className="absolute inset-x-4 bottom-0 h-24">
          <CatWalker variant={variant} size={size} startPercent={12 + i * 28} />
        </div>
      ))}
    </div>
  );
}
