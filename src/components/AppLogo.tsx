import { cn } from "@/lib/utils";

/** The app mark: three tiles and a spark, the same glyph as the bundle icon. */
export function AppLogo({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" className={cn("size-4", className)}>
      <rect x="2" y="2" width="8.5" height="8.5" rx="2.2" />
      <rect x="2" y="13.5" width="8.5" height="8.5" rx="2.2" />
      <rect x="13.5" y="13.5" width="8.5" height="8.5" rx="2.2" />
      <path d="M17.75 1.5c.4 2.8 1.6 4 4.4 4.4-2.8.4-4 1.6-4.4 4.4-.4-2.8-1.6-4-4.4-4.4 2.8-.4 4-1.6 4.4-4.4z" />
    </svg>
  );
}
