import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from "react";
import {
  buildSpace,
  launchModel,
  launchSpace,
  modelFiles,
  MODEL_PIPELINES,
  searchModels,
  searchSpaces,
  SPACE_CATEGORIES,
  type GgufFile,
  type ModelSummary,
  type SpaceSummary,
} from "@/lib/api";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { CompatBadge } from "@/components/CompatBadge";
import { useGpuGate } from "@/components/GpuGate";
import { formatBytes, formatCount } from "@/lib/format";
import { cn } from "@/lib/utils";

type LoadState = "loading" | "idle" | "error";

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

function CardGridSkeleton({ count = 6 }: { count?: number }) {
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {Array.from({ length: count }).map((_, i) => (
        <Skeleton key={i} className="h-40 w-full" />
      ))}
    </div>
  );
}

type Page<T> = { items: T[]; next_cursor: string | null };

type Feed<T> = {
  items: T[];
  cursor: string | null;
  /** `loading` = first page (grid empty), `more` = appending, `end` = cursor exhausted */
  state: "loading" | "more" | "idle" | "end" | "error";
  error: string | null;
};

function emptyFeed<T>(): Feed<T> {
  return { items: [], cursor: null, state: "loading", error: null };
}

/**
 * A search result that grows page by page. `key` names the search (query +
 * filter); when it changes while the tab is active the feed restarts from
 * page one. `loadMore` appends the next page; the sentinel below calls it
 * whenever the bottom of the grid scrolls into view.
 */
function usePagedFeed<T>(active: boolean, key: string, fetchPage: (cursor: string | null) => Promise<Page<T>>) {
  const [feed, setFeed] = useState<Feed<T>>(emptyFeed);
  const feedRef = useRef<Feed<T>>(emptyFeed());
  const loadedKey = useRef<string | null>(null);
  const requestId = useRef(0);
  const [retryTick, setRetryTick] = useState(0);

  function commit(next: Feed<T>) {
    feedRef.current = next;
    setFeed(next);
  }

  useEffect(() => {
    if (!active || loadedKey.current === key) return;
    loadedKey.current = key;
    const id = ++requestId.current;
    commit(emptyFeed());
    fetchPage(null)
      .then((page) => {
        if (id !== requestId.current) return;
        commit({
          items: page.items,
          cursor: page.next_cursor,
          state: page.next_cursor ? "idle" : "end",
          error: null,
        });
      })
      .catch((e: unknown) => {
        if (id !== requestId.current) return;
        loadedKey.current = null;
        commit({ ...feedRef.current, state: "error", error: String(e) });
      });
  }, [active, key, fetchPage, retryTick]);

  const loadMore = useCallback(() => {
    const current = feedRef.current;
    if ((current.state !== "idle" && current.state !== "error") || !current.cursor) return;
    const id = ++requestId.current;
    commit({ ...current, state: "more", error: null });
    fetchPage(current.cursor)
      .then((page) => {
        if (id !== requestId.current) return;
        commit({
          items: [...feedRef.current.items, ...page.items],
          cursor: page.next_cursor,
          state: page.next_cursor ? "idle" : "end",
          error: null,
        });
      })
      .catch((e: unknown) => {
        if (id !== requestId.current) return;
        commit({ ...feedRef.current, state: "error", error: String(e) });
      });
  }, [fetchPage]);

  const retry = useCallback(() => {
    if (feedRef.current.items.length === 0) setRetryTick((t) => t + 1);
    else loadMore();
  }, [loadMore]);

  return { feed, loadMore, retry };
}

/**
 * Sits under the grid; when it scrolls into view (or is already in view
 * because the page is short) it asks for the next page. Re-armed on every
 * change of `count` so a page that leaves the sentinel visible still fires.
 */
function LoadMoreSentinel({
  enabled,
  count,
  onVisible,
}: {
  enabled: boolean;
  count: number;
  onVisible: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!enabled || !el) return;
    // Observe within the nearest scrolling ancestor so the margin prefetches
    // a screen ahead inside the pane, not relative to the window.
    let root: HTMLElement | null = el.parentElement;
    while (root && !/(auto|scroll)/.test(getComputedStyle(root).overflowY)) root = root.parentElement;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) onVisible();
      },
      { root, rootMargin: "600px 0px" },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [enabled, count, onVisible]);
  return <div ref={ref} className="h-px" aria-hidden />;
}

function FilterChips({
  options,
  value,
  onChange,
}: {
  options: { slug: string; label: string }[];
  value: string | null;
  onChange: (slug: string | null) => void;
}) {
  const chip = (slug: string | null, label: string) => (
    <button
      key={slug ?? "all"}
      type="button"
      onClick={() => onChange(slug)}
      className={cn(
        "rounded-full border px-2.5 py-1 text-xs transition-colors",
        value === slug
          ? "border-foreground bg-foreground text-background"
          : "border-border text-muted-foreground hover:bg-muted hover:text-foreground",
      )}
    >
      {label}
    </button>
  );
  return (
    <div className="flex flex-wrap gap-1.5">
      {chip(null, "All")}
      {options.map((o) => chip(o.slug, o.label))}
    </div>
  );
}

function FeedFooter<T>({
  feed,
  noun,
  onRetry,
}: {
  feed: Feed<T>;
  noun: string;
  onRetry: () => void;
}) {
  if (feed.state === "more") return <CardGridSkeleton count={3} />;
  if (feed.state === "error" && feed.items.length > 0) {
    return (
      <Alert variant="destructive">
        <AlertTitle>Could not load more {noun}</AlertTitle>
        <AlertDescription className="flex items-center gap-3">
          <span className="min-w-0 flex-1 truncate">{feed.error}</span>
          <Button size="sm" variant="outline" onClick={onRetry}>
            Retry
          </Button>
        </AlertDescription>
      </Alert>
    );
  }
  if (feed.state === "end" && feed.items.length > 0) {
    return <p className="py-2 text-center text-xs text-muted-foreground">End of results</p>;
  }
  return null;
}

function FitBadge({ fits }: { fits: GgufFile["fits"] }) {
  if (fits === "gpu") return <Badge>fits GPU</Badge>;
  if (fits === "partial") return <Badge variant="secondary">CPU offload</Badge>;
  return <Badge variant="destructive">too large</Badge>;
}

function pickDefaultFile(files: GgufFile[]): GgufFile | null {
  const gpuFiles = files.filter((f) => f.fits === "gpu");
  const q4 = gpuFiles.find((f) => f.quant.toUpperCase().startsWith("Q4"));
  if (q4) return q4;
  if (gpuFiles[0]) return gpuFiles[0];
  const partial = files.find((f) => f.fits === "partial");
  if (partial) return partial;
  return null;
}

export function Store({ onLaunched }: { onLaunched: () => void }) {
  const [query, setQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [tab, setTab] = useState<"apps" | "models">("apps");
  const [category, setCategory] = useState<string | null>(null);
  const [pipeline, setPipeline] = useState<string | null>(null);

  // Each tab is a feed keyed by its own search; the other tab's search stays
  // lazy until the user selects it (#67).
  const fetchSpaces = useCallback(
    (cursor: string | null) => searchSpaces(debouncedQuery, { category, cursor }),
    [debouncedQuery, category],
  );
  const fetchModels = useCallback(
    (cursor: string | null) => searchModels(debouncedQuery, { pipeline, cursor }),
    [debouncedQuery, pipeline],
  );
  const spacesFeed = usePagedFeed<SpaceSummary>(
    tab === "apps",
    `${debouncedQuery}|${category ?? ""}`,
    fetchSpaces,
  );
  const modelsFeed = usePagedFeed<ModelSummary>(
    tab === "models",
    `${debouncedQuery}|${pipeline ?? ""}`,
    fetchModels,
  );
  const spaces = spacesFeed.feed.items;
  const models = modelsFeed.feed.items;

  const [launchingId, setLaunchingId] = useState<string | null>(null);
  const [buildingId, setBuildingId] = useState<string | null>(null);
  const [launchError, setLaunchError] = useState<string | null>(null);

  const [dialogModel, setDialogModel] = useState<ModelSummary | null>(null);
  const [files, setFiles] = useState<GgufFile[]>([]);
  const [filesState, setFilesState] = useState<LoadState>("loading");
  const [selectedFile, setSelectedFile] = useState<GgufFile | null>(null);
  const [launching, setLaunching] = useState(false);
  const { gate, dialog: gpuDialog } = useGpuGate();

  const [buildConsentSpace, setBuildConsentSpace] = useState<SpaceSummary | null>(null);
  const [consentUseRepoDockerfile, setConsentUseRepoDockerfile] = useState(false);
  const [consentDontAskAgain, setConsentDontAskAgain] = useState(false);
  const buildConsentResolve = useRef<((choice: BuildConsent) => void) | null>(null);

  useEffect(() => {
    const t = setTimeout(() => setDebouncedQuery(query), 400);
    return () => clearTimeout(t);
  }, [query]);

  useEffect(() => {
    if (!dialogModel) return;
    let cancelled = false;
    setFilesState("loading");
    setSelectedFile(null);
    modelFiles(dialogModel.id)
      .then((res) => {
        if (cancelled) return;
        setFiles(res);
        setSelectedFile(pickDefaultFile(res));
        setFilesState("idle");
      })
      .catch(() => {
        if (!cancelled) setFilesState("error");
      });
    return () => {
      cancelled = true;
    };
  }, [dialogModel]);

  function handleSearchKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") setDebouncedQuery(query);
  }

  /** Spaces on a GPU tier hold VRAM of their own; CPU tiers launch without asking. */
  function spaceWantsGpu(space: SpaceSummary): boolean {
    return !!space.hardware && !space.hardware.startsWith("cpu");
  }

  async function runSpace(space: SpaceSummary, env?: Record<string, string>) {
    setLaunchError(null);
    setLaunchingId(space.id);
    try {
      let leaseId: string | null = null;
      if (spaceWantsGpu(space)) {
        const result = await gate({ kind: "space", id: space.id }, space.title ?? space.name);
        if (!result.proceed) return;
        leaseId = result.leaseId;
      }
      await launchSpace(space.id, env, leaseId);
      onLaunched();
    } catch (e) {
      setLaunchError(String(e));
    } finally {
      setLaunchingId(null);
    }
  }

  /**
   * Build locally runs code from the Space's own repository (cloned Dockerfile
   * or generated one, either way `RUN` steps execute here with network access).
   * Ask once; after "Don't ask again" reuse the stored choice silently.
   */
  function confirmBuildLocally(space: SpaceSummary): Promise<BuildConsent> {
    if (loadBoolPreference(BUILD_DONT_ASK_KEY)) {
      return Promise.resolve({
        proceed: true,
        useRepoDockerfile: loadBoolPreference(BUILD_USE_REPO_DOCKERFILE_KEY),
      });
    }
    return new Promise<BuildConsent>((resolve) => {
      buildConsentResolve.current = resolve;
      setConsentUseRepoDockerfile(false);
      setConsentDontAskAgain(false);
      setBuildConsentSpace(space);
    });
  }

  function finishBuildConsent(proceed: boolean) {
    const resolve = buildConsentResolve.current;
    buildConsentResolve.current = null;
    setBuildConsentSpace(null);
    if (proceed && consentDontAskAgain) {
      saveBoolPreference(BUILD_DONT_ASK_KEY, true);
      saveBoolPreference(BUILD_USE_REPO_DOCKERFILE_KEY, consentUseRepoDockerfile);
    }
    resolve?.({ proceed, useRepoDockerfile: consentUseRepoDockerfile });
  }

  async function runBuildSpace(space: SpaceSummary, env?: Record<string, string>) {
    setLaunchError(null);
    setBuildingId(space.id);
    try {
      let leaseId: string | null = null;
      if (spaceWantsGpu(space)) {
        const result = await gate({ kind: "space", id: space.id }, space.title ?? space.name);
        if (!result.proceed) return;
        leaseId = result.leaseId;
      }
      const { proceed, useRepoDockerfile } = await confirmBuildLocally(space);
      if (!proceed) return;
      await buildSpace(space.id, useRepoDockerfile, env, leaseId);
      onLaunched();
    } catch (e) {
      setLaunchError(String(e));
    } finally {
      setBuildingId(null);
    }
  }

  // A Space that needs secrets (#57) is never launched directly: the button
  // opens this dialog first so the user can supply values.
  const [envDialogSpace, setEnvDialogSpace] = useState<SpaceSummary | null>(null);
  const [envDialogKind, setEnvDialogKind] = useState<"run" | "build" | null>(null);
  const [envValues, setEnvValues] = useState<Record<string, string>>({});

  function startLaunch(space: SpaceSummary, kind: "run" | "build") {
    if (space.secrets.length > 0) {
      setEnvDialogSpace(space);
      setEnvDialogKind(kind);
      setEnvValues(Object.fromEntries(space.secrets.map((name) => [name, ""])));
      return;
    }
    if (kind === "run") runSpace(space);
    else runBuildSpace(space);
  }

  function closeEnvDialog() {
    setEnvDialogSpace(null);
    setEnvDialogKind(null);
  }

  async function confirmEnvLaunch() {
    if (!envDialogSpace || !envDialogKind) return;
    const space = envDialogSpace;
    const kind = envDialogKind;
    closeEnvDialog();
    if (kind === "run") await runSpace(space, envValues);
    else await runBuildSpace(space, envValues);
  }

  async function confirmLaunchModel() {
    if (!dialogModel || !selectedFile) return;
    setLaunching(true);
    setLaunchError(null);
    try {
      const tag = `hf.co/${dialogModel.id}:${selectedFile.quant}`;
      const result = await gate(
        { kind: "model", tag, size_bytes: selectedFile.size_bytes },
        dialogModel.name,
      );
      if (!result.proceed) return;
      await launchModel(dialogModel.id, selectedFile.quant, result.leaseId);
      setDialogModel(null);
      onLaunched();
    } catch (e) {
      setLaunchError(String(e));
    } finally {
      setLaunching(false);
    }
  }

  return (
    <div className="flex flex-col gap-4 p-6">
      {gpuDialog}
      <div>
        <h1 className="font-heading text-lg font-medium">Store</h1>
        <p className="text-sm text-muted-foreground">
          Apps (Hugging Face Spaces) and chat models (GGUF) that run on this machine.
        </p>
      </div>

      <Input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={handleSearchKeyDown}
        placeholder="Search Hugging Face apps and models..."
        className="max-w-sm"
      />

      {launchError && (
        <Alert variant="destructive">
          <AlertTitle>Could not launch</AlertTitle>
          <AlertDescription>{launchError}</AlertDescription>
        </Alert>
      )}

      <Tabs value={tab} onValueChange={(v) => setTab(v as "apps" | "models")}>
        <TabsList>
          <TabsTrigger value="apps">Apps</TabsTrigger>
          <TabsTrigger value="models">Models</TabsTrigger>
        </TabsList>

        <TabsContent value="apps" className="mt-4 flex flex-col gap-4">
          <FilterChips options={SPACE_CATEGORIES} value={category} onChange={setCategory} />
          {spacesFeed.feed.state === "loading" && <CardGridSkeleton />}
          {spacesFeed.feed.state === "error" && spaces.length === 0 && (
            <Alert variant="destructive">
              <AlertTitle>Could not load apps</AlertTitle>
              <AlertDescription className="flex items-center gap-3">
                <span className="min-w-0 flex-1 truncate">
                  {spacesFeed.feed.error ?? "Search Hugging Face Spaces failed."}
                </span>
                <Button size="sm" variant="outline" onClick={spacesFeed.retry}>
                  Retry
                </Button>
              </AlertDescription>
            </Alert>
          )}
          {spacesFeed.feed.state === "end" && spaces.length === 0 && (
            <p className="text-sm text-muted-foreground">No apps match this search.</p>
          )}
          {spaces.length > 0 && (
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {spaces.map((space) => (
                <Card key={space.id}>
                  <CardHeader>
                    <CardTitle className="flex min-w-0 items-center gap-2">
                      <span className="text-lg leading-none">{space.emoji ?? "🤗"}</span>
                      <span className="min-w-0 truncate" title={space.title ?? space.name}>
                        {space.title ?? space.name}
                      </span>
                    </CardTitle>
                    <CardDescription
                      className="min-w-0 truncate"
                      title={`${space.author}/${space.name}`}
                    >
                      {space.author}/{space.name}
                    </CardDescription>
                    <CardAction className="shrink-0">
                      <CompatBadge compat={space.compat} reason={space.compat_reason} />
                    </CardAction>
                  </CardHeader>
                  <CardContent className="flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
                    {space.description && (
                      <p className="line-clamp-2 w-full text-sm text-foreground/80" title={space.description}>
                        {space.description}
                      </p>
                    )}
                    {space.category && <Badge variant="secondary">{space.category}</Badge>}
                    {space.sdk && <Badge variant="outline">{space.sdk}</Badge>}
                    <span>{formatCount(space.likes)} likes</span>
                    {space.hardware && <span title="Hardware tier the Space asks for on Hugging Face">{space.hardware}</span>}
                    {space.secrets.length > 0 && (
                      <span className="w-full text-amber-600 dark:text-amber-500">
                        Needs secrets: {space.secrets.join(", ")}
                      </span>
                    )}
                  </CardContent>
                  <CardFooter className="mt-auto flex gap-2">
                    <Button
                      className="flex-1"
                      disabled={space.compat === "incompatible" || launchingId === space.id}
                      onClick={() => startLaunch(space, "run")}
                    >
                      {launchingId === space.id ? "Launching..." : "Run"}
                    </Button>
                    {space.sdk !== "static" && (
                      <Tooltip>
                        <TooltipTrigger
                          render={
                            <Button
                              variant="outline"
                              disabled={
                                buildingId === space.id ||
                                (space.compat === "incompatible" &&
                                  (space.compat_reason?.toLowerCase().includes("static") ?? false))
                              }
                              onClick={() => startLaunch(space, "build")}
                            />
                          }
                        >
                          {buildingId === space.id ? "Building..." : "Build locally"}
                        </TooltipTrigger>
                        <TooltipContent>
                          Clone the Space and build its image on this machine (CPU on non-NVIDIA
                          GPUs)
                        </TooltipContent>
                      </Tooltip>
                    )}
                  </CardFooter>
                </Card>
              ))}
            </div>
          )}
          <FeedFooter feed={spacesFeed.feed} noun="apps" onRetry={spacesFeed.retry} />
          <LoadMoreSentinel
            enabled={tab === "apps" && spacesFeed.feed.state === "idle"}
            count={spaces.length}
            onVisible={spacesFeed.loadMore}
          />
        </TabsContent>

        <TabsContent value="models" className="mt-4 flex flex-col gap-4">
          <FilterChips options={MODEL_PIPELINES} value={pipeline} onChange={setPipeline} />
          {modelsFeed.feed.state === "loading" && <CardGridSkeleton />}
          {modelsFeed.feed.state === "error" && models.length === 0 && (
            <Alert variant="destructive">
              <AlertTitle>Could not load models</AlertTitle>
              <AlertDescription className="flex items-center gap-3">
                <span className="min-w-0 flex-1 truncate">
                  {modelsFeed.feed.error ?? "Search Hugging Face models failed."}
                </span>
                <Button size="sm" variant="outline" onClick={modelsFeed.retry}>
                  Retry
                </Button>
              </AlertDescription>
            </Alert>
          )}
          {modelsFeed.feed.state === "end" && models.length === 0 && (
            <p className="text-sm text-muted-foreground">No models match this search.</p>
          )}
          {models.length > 0 && (
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {models.map((model) => (
                <Card key={model.id}>
                  <CardHeader>
                    <CardTitle className="min-w-0 truncate" title={model.name}>
                      {model.name}
                    </CardTitle>
                    <CardDescription className="min-w-0 truncate" title={model.author}>
                      {model.author}
                    </CardDescription>
                    <CardAction className="shrink-0">
                      <CompatBadge compat={model.compat} reason={model.compat_reason} />
                    </CardAction>
                  </CardHeader>
                  <CardContent className="flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
                    {model.pipeline_tag && <Badge variant="outline">{model.pipeline_tag}</Badge>}
                    <span>{formatCount(model.downloads)} downloads</span>
                  </CardContent>
                  <CardFooter className="mt-auto">
                    <Button
                      className="w-full"
                      disabled={model.compat === "incompatible"}
                      onClick={() => setDialogModel(model)}
                    >
                      Run
                    </Button>
                  </CardFooter>
                </Card>
              ))}
            </div>
          )}
          <FeedFooter feed={modelsFeed.feed} noun="models" onRetry={modelsFeed.retry} />
          <LoadMoreSentinel
            enabled={tab === "models" && modelsFeed.feed.state === "idle"}
            count={models.length}
            onVisible={modelsFeed.loadMore}
          />
        </TabsContent>
      </Tabs>

      <Dialog
        open={dialogModel !== null}
        onOpenChange={(open) => {
          if (!open) setDialogModel(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{dialogModel?.name}</DialogTitle>
            <DialogDescription>{dialogModel?.id}</DialogDescription>
          </DialogHeader>

          {filesState === "loading" && (
            <div className="flex flex-col gap-1.5">
              {Array.from({ length: 3 }).map((_, i) => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          )}

          {filesState === "error" && (
            <Alert variant="destructive">
              <AlertTitle>Could not load files</AlertTitle>
              <AlertDescription>Fetching GGUF files for this model failed.</AlertDescription>
            </Alert>
          )}

          {filesState === "idle" && files.length === 0 && (
            <p className="text-sm text-muted-foreground">No GGUF files found for this model.</p>
          )}

          {filesState === "idle" && files.length > 0 && (
            <div className="flex max-h-64 flex-col gap-1.5 overflow-y-auto">
              {files.map((file) => (
                <button
                  key={file.filename}
                  type="button"
                  onClick={() => setSelectedFile(file)}
                  className={cn(
                    "flex items-center justify-between gap-2 rounded-lg border px-2.5 py-2 text-left text-sm transition-colors",
                    selectedFile?.filename === file.filename
                      ? "border-ring bg-muted"
                      : "border-border hover:bg-muted/50",
                  )}
                >
                  <span className="flex flex-col">
                    <span className="font-medium">{file.quant}</span>
                    <span className="text-xs text-muted-foreground">{formatBytes(file.size_bytes)}</span>
                  </span>
                  <FitBadge fits={file.fits} />
                </button>
              ))}
            </div>
          )}

          <DialogFooter>
            <Button variant="outline" onClick={() => setDialogModel(null)}>
              Cancel
            </Button>
            <Button
              disabled={!selectedFile || selectedFile.fits === "no" || launching}
              onClick={confirmLaunchModel}
            >
              {launching ? "Launching..." : "Run"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={buildConsentSpace !== null}
        onOpenChange={(open) => {
          if (!open) finishBuildConsent(false);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Build locally?</DialogTitle>
            <DialogDescription>
              Build locally runs code from this Space's repository on your PC. Continue?
            </DialogDescription>
          </DialogHeader>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              className="size-4 rounded border-input"
              checked={consentUseRepoDockerfile}
              onChange={(e) => setConsentUseRepoDockerfile(e.target.checked)}
            />
            Use the Space's own Dockerfile (runs its `RUN` steps verbatim)
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
        open={envDialogSpace !== null}
        onOpenChange={(open) => {
          if (!open) closeEnvDialog();
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Secrets for {envDialogSpace?.title ?? envDialogSpace?.name}</DialogTitle>
            <DialogDescription>
              This Space reads these values from its environment; supply them to run it.
            </DialogDescription>
          </DialogHeader>

          <div className="flex flex-col gap-2">
            {envDialogSpace?.secrets.map((name) => (
              <div key={name} className="flex flex-col gap-1">
                <label className="text-xs font-medium text-muted-foreground" htmlFor={`secret-${name}`}>
                  {name}
                </label>
                <Input
                  id={`secret-${name}`}
                  type="password"
                  value={envValues[name] ?? ""}
                  onChange={(e) => setEnvValues((prev) => ({ ...prev, [name]: e.target.value }))}
                />
              </div>
            ))}
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={closeEnvDialog}>
              Cancel
            </Button>
            <Button onClick={confirmEnvLaunch}>
              {envDialogKind === "build" ? "Build locally" : "Run"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
