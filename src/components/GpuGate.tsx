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

type Pending = { plan: GpuPlan; name: string; resolve: (result: GateResult) => void };

/** Join residents into "A, B, and C" for a sentence naming who gets stopped. */
function joinNames(items: string[]): string {
  if (items.length <= 1) return items[0] ?? "";
  if (items.length === 2) return `${items[0]} and ${items[1]}`;
  return `${items.slice(0, -1).join(", ")}, and ${items[items.length - 1]}`;
}

/**
 * `proceed`: whether the caller should launch. `leaseId`: the plan's lease
 * (from `GpuPlan.lease_id`), to pass through to the launch call so the store
 * can hold the same lock from plan to launch (#70); `null` on a CPU machine,
 * where nothing was scheduled and no lease was issued.
 */
export type GateResult = { proceed: boolean; leaseId: string | null };

/**
 * Ask the store what a launch would evict from GPU memory; when the answer is
 * "nothing", proceed at once. Otherwise show the list and let the user unload it
 * and continue, or cancel. `gate` resolves to whether the caller should launch.
 * `name` is the display name of the thing being launched, shown in the dialog.
 */
export function useGpuGate() {
  const [pending, setPending] = useState<Pending | null>(null);
  const [releasing, setReleasing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pendingRef = useRef<Pending | null>(null);

  const gate = useCallback(async (request: GpuRequest, name: string): Promise<GateResult> => {
    const plan = await gpuPlan(request);
    if (plan.evict.length === 0) return { proceed: true, leaseId: plan.lease_id };
    return new Promise<GateResult>((resolve) => {
      const p = { plan, name, resolve };
      pendingRef.current = p;
      setError(null);
      setPending(p);
    });
  }, []);

  function finish(result: GateResult) {
    const p = pendingRef.current;
    pendingRef.current = null;
    setPending(null);
    p?.resolve(result);
  }

  async function handleUnload() {
    const p = pendingRef.current;
    if (!p) return;
    setReleasing(true);
    setError(null);
    try {
      for (const r of p.plan.evict) await gpuRelease(r);
      finish({ proceed: true, leaseId: p.plan.lease_id });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setReleasing(false);
    }
  }

  const plan = pending?.plan;
  const evict = plan?.evict ?? [];
  // Only call it an "unload" when everything being freed is a model; stopping a
  // running service or Space reads better as "stop" to a user who did not
  // install anything.
  const allModels = evict.length > 0 && evict.every((r) => r.kind === "model");
  const verb = allModels ? "Unload" : "Stop";
  const residentSentence =
    evict.length > 0
      ? `${joinNames(
          evict.map((r) => (r.vram_mb !== null ? `${r.name} (${gb(r.vram_mb)})` : r.name)),
        )} will be stopped so ${pending?.name} can use the GPU.`
      : "";
  const dialog: ReactNode = (
    <Dialog
      open={pending !== null}
      onOpenChange={(open) => {
        if (!open && !releasing) finish({ proceed: false, leaseId: null });
      }}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Free GPU memory for {pending?.name}?</DialogTitle>
          <DialogDescription>{residentSentence}</DialogDescription>
        </DialogHeader>
        <ul className="flex flex-col gap-1.5 text-sm">
          {evict.map((r) => (
            <li key={`${r.kind}-${r.id}`} className="flex items-center gap-2">
              <Badge variant="outline">{r.kind}</Badge>
              <span className="min-w-0 flex-1 truncate">{r.name}</span>
              <span className="text-xs text-muted-foreground">{residentLabel(r)}</span>
            </li>
          ))}
        </ul>
        {error && <p className="text-xs text-destructive">{error}</p>}
        <DialogFooter>
          <Button
            variant="outline"
            disabled={releasing}
            onClick={() => finish({ proceed: false, leaseId: null })}
          >
            Cancel
          </Button>
          <Button disabled={releasing} onClick={handleUnload}>
            {releasing ? `${verb === "Unload" ? "Unloading" : "Stopping"}...` : `${verb} and continue`}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );

  return { gate, dialog };
}
