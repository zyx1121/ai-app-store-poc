import { useCallback, useEffect, useState } from "react";
import {
  gpuMemory,
  gpuRelease,
  onInstanceUpdate,
  onServiceUpdate,
  type GpuMemory,
  type Resident,
} from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { residentLabel } from "@/components/GpuGate";

const gb = (mb: number) => `${(mb / 1024).toFixed(1)} GB`;

/** What holds GPU memory right now, with a way to unload each item. */
export function GpuMemoryCard() {
  const [mem, setMem] = useState<GpuMemory | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    gpuMemory()
      .then(setMem)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  useEffect(() => {
    refresh();
    let cancelled = false;
    let timer: number | null = null;
    const schedule = () => {
      if (timer !== null) window.clearTimeout(timer);
      timer = window.setTimeout(refresh, 1500);
    };
    const unlisteners: (() => void)[] = [];
    Promise.all([onInstanceUpdate(schedule), onServiceUpdate(schedule)]).then((fns) => {
      if (cancelled) fns.forEach((fn) => fn());
      else unlisteners.push(...fns);
    });
    return () => {
      cancelled = true;
      if (timer !== null) window.clearTimeout(timer);
      unlisteners.forEach((fn) => fn());
    };
  }, [refresh]);

  async function handleUnload(r: Resident) {
    setBusy(`${r.kind}-${r.id}`);
    setError(null);
    try {
      await gpuRelease(r);
      refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  }

  if (!mem || mem.budget_mb === null) return null;

  const known = mem.residents.reduce((sum, r) => sum + (r.vram_mb ?? 0), 0);
  const used = mem.used_mb ?? known;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <span>GPU memory</span>
          <span className="text-sm font-normal text-muted-foreground">
            {gb(used)} of {gb(mem.budget_mb)} in use
          </span>
        </CardTitle>
        <CardDescription>
          One model family fits at a time; launching another asks before unloading these.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        {mem.residents.length === 0 ? (
          <p className="text-xs text-muted-foreground">Nothing resident.</p>
        ) : (
          <ul className="flex flex-col gap-1.5 text-sm">
            {mem.residents.map((r) => {
              const key = `${r.kind}-${r.id}`;
              return (
                <li key={key} className="flex items-center gap-2">
                  <Badge variant="outline">{r.kind}</Badge>
                  <span className="min-w-0 flex-1 truncate">{r.name}</span>
                  <span className="text-xs text-muted-foreground">{residentLabel(r)}</span>
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={busy !== null}
                    onClick={() => handleUnload(r)}
                  >
                    {busy === key ? "Unloading..." : "Unload"}
                  </Button>
                </li>
              );
            })}
          </ul>
        )}
        {error && <p className="text-xs text-destructive">{error}</p>}
      </CardContent>
    </Card>
  );
}
