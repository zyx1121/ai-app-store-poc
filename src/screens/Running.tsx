import { useEffect, useState } from "react";
import {
  buildSpace,
  launchModel,
  launchSpace,
  listLibrary,
  modelFiles,
  onInstanceUpdate,
  onLibraryUpdate,
  openUrl,
  removeInstance,
  spaceSummary,
  stopInstance,
  type LibraryItem,
  type SpaceSummary,
} from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { StatusBadge } from "@/components/StatusBadge";
import { LogTail } from "@/components/LogTail";
import { GpuMemoryCard } from "@/components/GpuMemoryCard";
import { useGpuGate } from "@/components/GpuGate";

const BUILD_DONT_ASK_KEY = "buildLocally.dontAskAgain";
const BUILD_USE_REPO_DOCKERFILE_KEY = "buildLocally.useRepoDockerfile";

function loadBoolPreference(key: string): boolean {
  try {
    return localStorage.getItem(key) === "true";
  } catch {
    return false;
  }
}

function saveBoolPreference(key: string, value: boolean) {
  try {
    localStorage.setItem(key, String(value));
  } catch {
    // ignore (private browsing, storage disabled, ...)
  }
}

type BuildConsent = { proceed: boolean; useRepoDockerfile: boolean };

/** How a Space is started: from the Hub's image, or from one built here. */
type RunKind = "run" | "build";

/** The name Ollama knows the model by; mirrors `model_tag` in instances.rs. */
function modelTag(item: LibraryItem): string {
  const quant = item.quant ?? "";
  return item.repo.includes("/") ? `hf.co/${item.repo}:${quant}` : `${item.repo}:${quant}`;
}

/** Spaces on a GPU tier hold VRAM of their own; CPU tiers run without asking. */
function spaceWantsGpu(space: SpaceSummary): boolean {
  return !!space.hardware && !space.hardware.startsWith("cpu");
}

function ItemCard({
  item,
  busy,
  onRun,
  onBuild,
  onStop,
  onOpenChat,
  onRemove,
}: {
  item: LibraryItem;
  busy: boolean;
  onRun: (item: LibraryItem) => void;
  onBuild: (item: LibraryItem) => void;
  onStop: (id: string) => void;
  onOpenChat: (instanceId: string) => void;
  onRemove: (item: LibraryItem) => void;
}) {
  const live = item.status !== "stopped";
  const canOpen = item.kind === "space" && item.status === "running" && item.url;
  const canChat = item.kind === "model" && item.status === "running";

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Badge variant="outline">{item.kind}</Badge>
          <span className="truncate">{item.display_name}</span>
        </CardTitle>
        <CardDescription>
          {item.repo}
          {item.quant && ` (${item.quant})`}
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <StatusBadge status={item.status} />
          {item.local_build && <Badge variant="outline">Built locally</Badge>}
          {item.port && <span>port {item.port}</span>}
          {item.url && <span className="truncate">{item.url}</span>}
          {item.status === "pulling" && item.pull_size_mb !== null && (
            <span>{(item.pull_size_mb / 1024).toFixed(1)} GB image</span>
          )}
          {item.status === "pulling" && item.progress_pct !== null && (
            <span className="flex min-w-24 flex-1 items-center gap-1.5">
              <span className="h-1 flex-1 overflow-hidden rounded-full bg-muted">
                <span
                  className="block h-full rounded-full bg-primary transition-[width]"
                  style={{ width: `${item.progress_pct}%` }}
                />
              </span>
              {item.progress_pct}%
            </span>
          )}
        </div>
        {item.error && <p className="text-xs text-destructive">{item.error}</p>}
        {item.log_tail.length > 0 && <LogTail lines={item.log_tail} />}
      </CardContent>
      <CardFooter className="flex gap-2">
        {!live && (
          <Button disabled={busy} onClick={() => onRun(item)}>
            Run
          </Button>
        )}
        {!live && item.kind === "space" && (
          <Button variant="outline" disabled={busy} onClick={() => onBuild(item)}>
            Build locally
          </Button>
        )}
        {canOpen && (
          <Button variant="outline" onClick={() => openUrl(item.url!)}>
            Open
          </Button>
        )}
        {canChat && <Button onClick={() => onOpenChat(item.id)}>Chat</Button>}
        {live && (
          <Button variant="outline" disabled={busy} onClick={() => onStop(item.id)}>
            Stop
          </Button>
        )}
        <Button
          variant="destructive"
          className="ml-auto"
          disabled={busy}
          onClick={() => onRemove(item)}
        >
          Remove
        </Button>
      </CardFooter>
    </Card>
  );
}

export function Running({ onOpenChat }: { onOpenChat: (instanceId: string) => void }) {
  const [items, setItems] = useState<LibraryItem[]>([]);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { gate, dialog: gpuDialog } = useGpuGate();

  // A Space that reads secrets from its environment collects them here, right
  // before the launch that needs them (#57).
  const [secretsFor, setSecretsFor] = useState<{
    item: LibraryItem;
    space: SpaceSummary;
    kind: RunKind;
  } | null>(null);
  const [secretValues, setSecretValues] = useState<Record<string, string>>({});

  // Build locally runs code from the Space's own repository; ask once.
  const [consentOpen, setConsentOpen] = useState(false);
  const [consentUseRepoDockerfile, setConsentUseRepoDockerfile] = useState(false);
  const [consentDontAskAgain, setConsentDontAskAgain] = useState(false);
  const [consentResolve, setConsentResolve] = useState<((c: BuildConsent) => void) | null>(null);

  const [removeTarget, setRemoveTarget] = useState<LibraryItem | null>(null);

  useEffect(() => {
    let cancelled = false;
    listLibrary().then((res) => {
      if (!cancelled) setItems(res);
    });

    const unlisteners: (() => void)[] = [];
    const track = (fn: () => void) => {
      if (cancelled) fn();
      else unlisteners.push(fn);
    };
    // The library event carries the whole list (an add or a remove); instance
    // updates carry status and log lines for one item already in it.
    onLibraryUpdate((next) => setItems(next)).then(track);
    onInstanceUpdate((instance) => {
      setItems((prev) =>
        prev.map((item) =>
          item.id === instance.id
            ? {
                ...item,
                display_name: instance.display_name,
                status: instance.status,
                model_tag: instance.model_tag,
                port: instance.port,
                url: instance.url,
                error: instance.error,
                log_tail: instance.log_tail,
                local_build: instance.local_build,
                gpu: instance.gpu,
                pull_size_mb: instance.pull_size_mb,
                progress_pct: instance.progress_pct,
              }
            : item,
        ),
      );
    }).then(track);

    return () => {
      cancelled = true;
      for (const fn of unlisteners) fn();
    };
  }, []);

  function confirmBuildLocally(): Promise<BuildConsent> {
    if (loadBoolPreference(BUILD_DONT_ASK_KEY)) {
      return Promise.resolve({
        proceed: true,
        useRepoDockerfile: loadBoolPreference(BUILD_USE_REPO_DOCKERFILE_KEY),
      });
    }
    return new Promise<BuildConsent>((resolve) => {
      setConsentUseRepoDockerfile(false);
      setConsentDontAskAgain(false);
      setConsentResolve(() => resolve);
      setConsentOpen(true);
    });
  }

  function finishBuildConsent(proceed: boolean) {
    setConsentOpen(false);
    if (proceed && consentDontAskAgain) {
      saveBoolPreference(BUILD_DONT_ASK_KEY, true);
      saveBoolPreference(BUILD_USE_REPO_DOCKERFILE_KEY, consentUseRepoDockerfile);
    }
    consentResolve?.({ proceed, useRepoDockerfile: consentUseRepoDockerfile });
    setConsentResolve(null);
  }

  /** Gate the GPU, then pull (or build) and start. Everything a launch needs
   * that the library entry does not carry comes from the Hub here. */
  async function launchSpaceItem(space: SpaceSummary, kind: RunKind, env: Record<string, string>) {
    let leaseId: string | null = null;
    if (spaceWantsGpu(space)) {
      const result = await gate({ kind: "space", id: space.id }, space.title ?? space.name);
      if (!result.proceed) return;
      leaseId = result.leaseId;
    }
    if (kind === "build") {
      const { proceed, useRepoDockerfile } = await confirmBuildLocally();
      if (!proceed) return;
      await buildSpace(space.id, useRepoDockerfile, env, leaseId);
    } else {
      await launchSpace(space.id, env, leaseId);
    }
  }

  async function launchModelItem(item: LibraryItem) {
    // The GPU plan is only as good as the size it is given; the file list is
    // where the quant's size comes from (a bare Ollama name has none).
    const sizeBytes = await modelFiles(item.repo)
      .then((files) => files.find((f) => f.quant === item.quant)?.size_bytes ?? null)
      .catch(() => null);
    const result = await gate(
      { kind: "model", tag: modelTag(item), size_bytes: sizeBytes },
      item.display_name,
    );
    if (!result.proceed) return;
    await launchModel(item.repo, item.quant ?? "", result.leaseId);
  }

  async function handleRun(item: LibraryItem, kind: RunKind = "run") {
    setError(null);
    setBusyId(item.id);
    try {
      if (item.kind === "model") {
        await launchModelItem(item);
        return;
      }
      const space = await spaceSummary(item.repo);
      if (space.secrets.length > 0) {
        setSecretValues(Object.fromEntries(space.secrets.map((name) => [name, ""])));
        setSecretsFor({ item, space, kind });
        return;
      }
      await launchSpaceItem(space, kind, {});
    } catch (e) {
      setError(String(e));
    } finally {
      setBusyId(null);
    }
  }

  async function confirmSecrets() {
    const pending = secretsFor;
    if (!pending) return;
    setSecretsFor(null);
    setError(null);
    setBusyId(pending.item.id);
    try {
      await launchSpaceItem(pending.space, pending.kind, secretValues);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusyId(null);
    }
  }

  async function handleStop(id: string) {
    setError(null);
    setBusyId(id);
    try {
      await stopInstance(id);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusyId(null);
    }
  }

  async function confirmRemove() {
    const target = removeTarget;
    if (!target) return;
    setRemoveTarget(null);
    setError(null);
    setBusyId(target.id);
    try {
      await removeInstance(target.id);
      setItems((prev) => prev.filter((i) => i.id !== target.id));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusyId(null);
    }
  }

  return (
    <div className="flex flex-col gap-3 p-6">
      {gpuDialog}
      <GpuMemoryCard />

      {error && <p className="text-sm text-destructive">{error}</p>}

      {items.length === 0 && (
        <p className="text-sm text-muted-foreground">Nothing added yet</p>
      )}

      {items.map((item) => (
        <ItemCard
          key={item.id}
          item={item}
          busy={busyId === item.id}
          onRun={(i) => handleRun(i, "run")}
          onBuild={(i) => handleRun(i, "build")}
          onStop={handleStop}
          onOpenChat={onOpenChat}
          onRemove={setRemoveTarget}
        />
      ))}

      <Dialog
        open={secretsFor !== null}
        onOpenChange={(open) => {
          if (!open) setSecretsFor(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              Secrets for {secretsFor?.space.title ?? secretsFor?.space.name}
            </DialogTitle>
          </DialogHeader>

          <div className="flex flex-col gap-2">
            {secretsFor?.space.secrets.map((name) => (
              <div key={name} className="flex flex-col gap-1">
                <label
                  className="text-xs font-medium text-muted-foreground"
                  htmlFor={`secret-${name}`}
                >
                  {name}
                </label>
                <Input
                  id={`secret-${name}`}
                  type="password"
                  value={secretValues[name] ?? ""}
                  onChange={(e) =>
                    setSecretValues((prev) => ({ ...prev, [name]: e.target.value }))
                  }
                />
              </div>
            ))}
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setSecretsFor(null)}>
              Cancel
            </Button>
            <Button onClick={confirmSecrets}>
              {secretsFor?.kind === "build" ? "Build locally" : "Run"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={consentOpen}
        onOpenChange={(open) => {
          if (!open) finishBuildConsent(false);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Build locally?</DialogTitle>
            <DialogDescription>
              Build locally runs code from this app's repository on your PC. Continue?
            </DialogDescription>
          </DialogHeader>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              className="size-4 rounded border-input"
              checked={consentUseRepoDockerfile}
              onChange={(e) => setConsentUseRepoDockerfile(e.target.checked)}
            />
            Use the app's own Dockerfile (runs its `RUN` steps verbatim)
          </label>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              className="size-4 rounded border-input"
              checked={consentDontAskAgain}
              onChange={(e) => setConsentDontAskAgain(e.target.checked)}
            />
            Don't ask again
          </label>
          <DialogFooter>
            <Button variant="outline" onClick={() => finishBuildConsent(false)}>
              Cancel
            </Button>
            <Button onClick={() => finishBuildConsent(true)}>Continue</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={removeTarget !== null}
        onOpenChange={(open) => {
          if (!open) setRemoveTarget(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Remove {removeTarget?.display_name}?</DialogTitle>
            <DialogDescription>Downloaded files are deleted.</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setRemoveTarget(null)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={confirmRemove}>
              Remove
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
