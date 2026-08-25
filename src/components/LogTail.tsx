import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

/** Collapsible monospace panel showing the last few log lines of an instance. */
export function LogTail({ lines, max = 5 }: { lines: string[]; max?: number }) {
  const [open, setOpen] = useState(false);
  const tail = lines.slice(-max);

  if (tail.length === 0) return null;

  return (
    <div className="rounded-lg border border-border">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between px-2.5 py-1.5 text-xs text-muted-foreground hover:text-foreground"
      >
        <span>Logs</span>
        <ChevronDown className={cn("size-3.5 transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <pre className="max-h-32 overflow-y-auto border-t border-border bg-muted/50 p-2.5 font-mono text-xs whitespace-pre-wrap break-all">
          {tail.join("\n")}
        </pre>
      )}
    </div>
  );
}
