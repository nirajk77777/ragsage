/**
 * The ragsage mark: brackets around a reference.
 *
 * Drawn in `currentColor` on a 32-unit grid, so this one definition serves the
 * light header and the dark one with no second export to keep in step. The
 * favicon is deliberately *not* this file — `app/icon.svg` is a filled tile,
 * because a stroked mark on a transparent ground disappears into a browser tab
 * strip at the 16px it is given there.
 *
 * The geometry is what makes it survive small: 3-unit strokes with gaps wider
 * than the strokes themselves. Thinning them is the change that would look like
 * a refinement in a diff and read as a smudge in a tab.
 */
export function Logo({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 32 32"
      width="22"
      height="22"
      fill="none"
      aria-hidden="true"
      className={className}
    >
      <g stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
        <path d="M13 4H8A3 3 0 0 0 5 7V25A3 3 0 0 0 8 28H13" />
        <path d="M19 4H24A3 3 0 0 1 27 7V25A3 3 0 0 1 24 28H19" />
      </g>
      <circle cx="16" cy="16" r="3.2" fill="currentColor" />
    </svg>
  );
}
