# Interactive Cat Companions (AXE Genesis Integration)

Interactive, animated 16-bit retro sprite cats that wander the dashboard, perching on toasts and ambient interface margins.

## The Cats
1. **Marmalade** — Orange Tabby, brisk walker, restless personality.
2. **Smoke** — Grey/blue point, slow, aloof, long stretches.
3. **Mochi** — Calico, bouncy, grooms and licks paws constantly.

## Shared Behavioral Vocabulary
- `idle` (breathing, tail flicking)
- `walk` (horizontal roaming with auto-mirroring)
- `sit` (attentive posture)
- `lick` (grooming front paw)
- `stretch` (cat arch stretch)
- `pounce` (interactive jump on click)
- `fight` (playful paw throw)
- `knead` (happy paw kneading)
- `yawn` (sleepy mouth stretch)

## Architecture & Components
- `<Cat variant="marmalade" behavior="lick" size={72} />`: Core presentational sprite component using CSS `steps(4)` keyframes.
- `<CatWalker variant="smoke" size={72} />`: Horizontal tracking engine handling bounds, turnarounds, and autonomous state transitions.
- `<AmbientCats />`: Fixed non-blocking background layer roaming the bottom margin of the dashboard shell.
- `catToast(message, variant)`: Custom Sonner toast wrapper rendering a cat perched on top of notification toasts.

## Asset Pipeline
- Sprite sheets: `/public/marmalade.png`, `/public/smoke.png`, `/public/mochi.png` (4x4 transparent grid).
- Header & Favicon logo: `/public/cat-logo.png` (Marmalade/Mochi paw-licking cat mask with zero background).
