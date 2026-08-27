import { useCallback, useRef, useState, type ReactNode } from "react";
import { gpuPlan, gpuRelease, type GpuPlan, type GpuRequest, type Resident } from "@/lib/api";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";

const gb = (mb: number) => `${(mb / 1024).toFixed(1)} GB`;

export function residentLabel(r: Resident): string {
  return r.vram_mb === null ? "size unknown" : gb(r.vram_mb);
}

type Pending = { plan: GpuPlan; resolve: (proceed: boolean) => void };

/**
 * Ask the store what a launch would evict from GPU memory; when the answer is
 * "nothing", proceed at once. Otherwise show the list and let the user unload it
 * and continue, or cancel. `gate` resolves to whether the caller should launch.
 */
export function useGpuGate() {
  const [pending, setPending] = useState<Pending | null>(null);
  const [releasing, setReleasing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pendingRef = useRef<Pending | null>(null);

  const gate = useCallback(async (request: GpuRequest): Promise<boolean> => {
    const plan = await gpuPlan(request);
    if (plan.evict.length === 0) return true;
    return new Promise<boolean>((resolve) => {
      const p = { plan, resolve };
      pendingRef.current = p;
      setError(null);
      setPending(p);
    });
  }, []);

  function finish(proceed: boolean) {
    const p = pendingRef.current;
    pendingRef.current = null;
    setPending(null);
    p?.resolve(proceed);
  }

  async function handleUnload() {
    const p = pendingRef.current;
    if (!p) return;
    setReleasing(true);
    setError(null);
    try {
      for (const r of p.plan.evict) await gpuRelease(r);
      finish(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setReleasing(false);
    }
  }

  const plan = pending?.plan;
  const dialog: ReactNode = (
    <Dialog
      open={pending !== null}
      onOpenChange={(open) => {
        if (!open && !releasing) finish(false);
      }}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle>GPU memory is in use</DialogTitle>
          <DialogDescription>
            {plan?.need_mb !== null && plan?.need_mb !== undefined
              ? `This needs about ${gb(plan.need_mb)}`
              : "This needs the GPU to itself"}
            {plan?.budget_mb ? ` of ${gb(plan.budget_mb)}` : ""}
            {plan && plan.resident_mb > 0 ? `; ${gb(plan.resident_mb)} stays loaded` : ""}.
            The following will be unloaded first:
          </DialogDescription>
        </DialogHeader>
        <ul className="flex flex-col gap-1.5 text-sm">
          {plan?.evict.map((r) => (
            <li key={`${r.kind}-${r.id}`} className="flex items-center gap-2">
              <Badge variant="outline">{r.kind}</Badge>
              <span className="min-w-0 flex-1 truncate">{r.name}</span>
              <span className="text-xs text-muted-foreground">{residentLabel(r)}</span>
            </li>
          ))}
        </ul>
        {error && <p className="text-xs text-destructive">{error}</p>}
        <DialogFooter>
          <Button variant="outline" disabled={releasing} onClick={() => finish(false)}>
            Cancel
          </Button>
          <Button disabled={releasing} onClick={handleUnload}>
            {releasing ? "Unloading..." : "Unload and continue"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );

  return { gate, dialog };
}
