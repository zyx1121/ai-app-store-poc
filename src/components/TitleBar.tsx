import { useEffect, useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { Copy, Minus, Square, X } from "lucide-react";
import { AppLogo } from "@/components/AppLogo";
import { cn } from "@/lib/utils";

/**
 * Custom window chrome. The native title bar is off (`decorations: false`);
 * this bar carries the drag region and the three window controls so the
 * frame shares the app's type, spacing and colours instead of the OS theme.
 */
export function TitleBar() {
  const [maximized, setMaximized] = useState(false);

  useEffect(() => {
    const win = getCurrentWindow();
    let unlisten: (() => void) | undefined;
    let cancelled = false;
    win.isMaximized().then((m) => {
      if (!cancelled) setMaximized(m);
    });
    win
      .onResized(() => {
        win.isMaximized().then(setMaximized);
      })
      .then((fn) => {
        if (cancelled) fn();
        else unlisten = fn;
      });
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  const controlClass =
    "flex h-full w-11 items-center justify-center text-muted-foreground transition-colors hover:bg-muted hover:text-foreground";

  return (
    <header
      data-tauri-drag-region
      className="flex h-9 shrink-0 select-none items-center justify-between border-b border-border bg-background"
    >
      <div className="pointer-events-none flex items-center gap-2 px-4 font-heading text-sm font-medium">
        <AppLogo />
        AI App Store
      </div>
      <div className="flex h-full">
        <button type="button" aria-label="Minimize" className={controlClass} onClick={() => getCurrentWindow().minimize()}>
          <Minus className="size-4" />
        </button>
        <button
          type="button"
          aria-label={maximized ? "Restore" : "Maximize"}
          className={controlClass}
          onClick={() => getCurrentWindow().toggleMaximize()}
        >
          {maximized ? <Copy className="size-3.5 -scale-x-100" /> : <Square className="size-3.5" />}
        </button>
        <button
          type="button"
          aria-label="Close"
          className={cn(controlClass, "hover:bg-destructive hover:text-white")}
          onClick={() => getCurrentWindow().close()}
        >
          <X className="size-4" />
        </button>
      </div>
    </header>
  );
}
