import { useEffect, useState } from "react";
import {
  onStorageProgress,
  storageCleanup,
  storageUsage,
  type StorageCleanupOptions,
  type StorageProgressEvent,
  type StorageUsage,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { HardDrive } from "lucide-react";
import { cn } from "@/lib/utils";

const gb = (mb: number) => `${(mb / 1024).toFixed(1)} GB`;

type Options = StorageCleanupOptions;

const DEFAULT_OPTIONS: Options = {
  dangling_images: true,
  build_cache: true,
  unused_space_images: true,
  hf_cache_older_than_days: undefined,
  compact_vhd: false,
};

function eventTone(status: StorageProgressEvent["status"]) {
  switch (status) {
    case "error":
      return "text-destructive";
    case "ok":
      return "text-foreground";
    case "start":
      return "mt-2 font-semibold text-foreground first:mt-0";
    default:
      return "pl-2 text-muted-foreground";
  }
}

/** Docker images, build cache, the HF cache and the WSL virtual disk, with a
 * "Free up space" action. Rendered at the bottom of the Setup screen (#63). */
export function StorageCard() {
  const [usage, setUsage] = useState<StorageUsage | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [options, setOptions] = useState<Options>(DEFAULT_OPTIONS);
  const [trimHfCache, setTrimHfCache] = useState(false);
  const [cleaning, setCleaning] = useState(false);
  const [events, setEvents] = useState<StorageProgressEvent[]>([]);
  const [freedMb, setFreedMb] = useState<number | null>(null);

  function refresh() {
    setLoading(true);
    setError(null);
    storageUsage()
      .then(setUsage)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }

  useEffect(() => {
    refresh();
  }, []);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let cancelled = false;
    onStorageProgress((event) => {
      setEvents((prev) => [...prev, event]);
    }).then((fn) => {
      if (cancelled) fn();
      else unlisten = fn;
    });
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  async function freeUpSpace() {
    setCleaning(true);
    setEvents([]);
    setFreedMb(null);
    setError(null);
    try {
      const result = await storageCleanup({
        ...options,
        hf_cache_older_than_days: trimHfCache ? 30 : undefined,
      });
      setUsage(result.usage);
      setFreedMb(result.freed_mb);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setCleaning(false);
    }
  }

  const nothingSelected =
    !options.dangling_images &&
    !options.build_cache &&
    !options.unused_space_images &&
    !trimHfCache &&
    !options.compact_vhd;

  return (
    <div className="flex flex-col gap-3 rounded-xl border border-border p-3">
      <div className="flex items-center gap-2">
        <HardDrive className="size-4 shrink-0 text-foreground" />
        <span className="flex-1 text-sm font-medium">Storage</span>
        <Button variant="outline" size="sm" onClick={refresh} disabled={loading || cleaning}>
          {loading ? "Checking..." : "Refresh"}
        </Button>
      </div>

      {usage ? (
        <div className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-3 gap-y-1 text-sm">
          <span className="text-muted-foreground">Docker images</span>
          <span>
            {gb(usage.images_total_mb)}
            {usage.images_reclaimable_mb > 0 && (
              <span className="text-xs text-muted-foreground">
                {" "}
                ({gb(usage.images_reclaimable_mb)} reclaimable)
              </span>
            )}
          </span>
          <span className="text-muted-foreground">Build cache</span>
          <span>{gb(usage.build_cache_mb)}</span>
          <span className="text-muted-foreground">HF cache</span>
          <span>{gb(usage.hf_cache_mb)}</span>
          <span className="text-muted-foreground">Build directory</span>
          <span>{gb(usage.build_dir_mb)}</span>
          <span className="text-muted-foreground">WSL virtual disk</span>
          <span>{usage.vhdx_mb !== null ? gb(usage.vhdx_mb) : "-"}</span>
        </div>
      ) : (
        !loading && !error && <p className="text-sm text-muted-foreground">No data yet.</p>
      )}

      {error && <p className="text-xs text-destructive">{error}</p>}

      <div className="flex flex-col gap-1.5 border-t border-border pt-2 text-sm">
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            className="size-4 rounded border-border accent-foreground"
            checked={options.dangling_images}
            disabled={cleaning}
            onChange={(e) =>
              setOptions((o) => ({ ...o, dangling_images: e.target.checked }))
            }
          />
          Dangling images
        </label>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            className="size-4 rounded border-border accent-foreground"
            checked={options.build_cache}
            disabled={cleaning}
            onChange={(e) => setOptions((o) => ({ ...o, build_cache: e.target.checked }))}
          />
          Docker build cache
        </label>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            className="size-4 rounded border-border accent-foreground"
            checked={options.unused_space_images}
            disabled={cleaning}
            onChange={(e) =>
              setOptions((o) => ({ ...o, unused_space_images: e.target.checked }))
            }
          />
          Space images not in use
        </label>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            className="size-4 rounded border-border accent-foreground"
            checked={trimHfCache}
            disabled={cleaning}
            onChange={(e) => setTrimHfCache(e.target.checked)}
          />
          HF cache entries untouched for 30+ days
        </label>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            className="size-4 rounded border-border accent-foreground"
            checked={options.compact_vhd}
            disabled={cleaning}
            onChange={(e) => setOptions((o) => ({ ...o, compact_vhd: e.target.checked }))}
          />
          Compact the WSL virtual disk
        </label>
        {options.compact_vhd && (
          <p className="pl-6 text-xs text-muted-foreground">
            Stops the distro for a moment: Docker, Ollama and every running app or model go
            down and back up. Only runs when nothing is starting or running.
          </p>
        )}
      </div>

      <Button onClick={freeUpSpace} disabled={cleaning || nothingSelected} className="self-start">
        {cleaning ? "Freeing up space..." : "Free up space"}
      </Button>

      {freedMb !== null && !cleaning && (
        <p className="text-sm font-medium">Freed {(freedMb / 1024).toFixed(1)} GB.</p>
      )}

      {(cleaning || events.length > 0) && (
        <div className="max-h-48 overflow-y-auto rounded-lg border border-border bg-muted/50 p-3 font-mono text-xs">
          {events.length === 0 ? (
            <p className="text-muted-foreground">Starting...</p>
          ) : (
            events.map((event, i) => (
              <div key={i} className={cn(eventTone(event.status))}>
                {event.status === "start" ? event.step : event.message}
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
