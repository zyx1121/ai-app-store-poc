import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from "react";
import { Check } from "lucide-react";
import {
  addModel,
  addSpace,
  listLibrary,
  modelFiles,
  MODEL_PIPELINES,
  onLibraryUpdate,
  searchModels,
  searchSpaces,
  SPACE_CATEGORIES,
  type GgufFile,
  type LibraryItem,
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
import { formatBytes, formatCount } from "@/lib/format";
import { cn } from "@/lib/utils";

type LoadState = "loading" | "idle" | "error";

/** A model is added per quant, so an added model's key is repo + quant. */
function modelKey(repo: string, quant: string): string {
  return `${repo}|${quant}`;
}

/** The only action a Store card has: subscribe. Once added it stays disabled
 * with a check, and Running takes over. */
function AddButton({
  added,
  adding,
  disabled,
  onAdd,
}: {
  added: boolean;
  adding: boolean;
  disabled: boolean;
  onAdd: () => void;
}) {
  if (added) {
    return (
      <Button className="w-full" variant="outline" disabled>
        <Check className="size-4" />
        Added
      </Button>
    );
  }
  return (
    <Button className="w-full" disabled={disabled || adding} onClick={onAdd}>
      {adding ? "Adding..." : "Add"}
    </Button>
  );
}

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

export function Store() {
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

  const [addingId, setAddingId] = useState<string | null>(null);
  const [addError, setAddError] = useState<string | null>(null);
  const [library, setLibrary] = useState<LibraryItem[]>([]);

  const [dialogModel, setDialogModel] = useState<ModelSummary | null>(null);
  const [files, setFiles] = useState<GgufFile[]>([]);
  const [filesState, setFilesState] = useState<LoadState>("loading");
  const [selectedFile, setSelectedFile] = useState<GgufFile | null>(null);
  const [adding, setAdding] = useState(false);

  // Which cards say "Added". A Space is one item; a model is one item per
  // quant, so the file list checks the quant too.
  const addedSpaces = new Set(library.filter((i) => i.kind === "space").map((i) => i.repo));
  const addedModelRepos = new Set(library.filter((i) => i.kind === "model").map((i) => i.repo));
  const addedModels = new Set(
    library.filter((i) => i.kind === "model").map((i) => modelKey(i.repo, i.quant ?? "")),
  );

  useEffect(() => {
    const t = setTimeout(() => setDebouncedQuery(query), 400);
    return () => clearTimeout(t);
  }, [query]);

  // Adds from this screen come back on the event too, so the grid and Running
  // never disagree about what is in the library.
  useEffect(() => {
    let cancelled = false;
    listLibrary().then((items) => {
      if (!cancelled) setLibrary(items);
    });
    let unlisten: (() => void) | undefined;
    onLibraryUpdate((items) => setLibrary(items)).then((fn) => {
      if (cancelled) fn();
      else unlisten = fn;
    });
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

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

  /** Add records the Space and nothing else: no pull, no GPU lease, no
   * secrets. Running is where it is run. */
  async function handleAddSpace(space: SpaceSummary) {
    setAddError(null);
    setAddingId(space.id);
    try {
      await addSpace(space.id);
    } catch (e) {
      setAddError(String(e));
    } finally {
      setAddingId(null);
    }
  }

  /** A model is added per quant, so the file list opens first. */
  async function handleAddModel() {
    if (!dialogModel || !selectedFile) return;
    setAdding(true);
    setAddError(null);
    try {
      await addModel(dialogModel.id, selectedFile.quant);
      setDialogModel(null);
    } catch (e) {
      setAddError(String(e));
    } finally {
      setAdding(false);
    }
  }

  return (
    <div className="flex flex-col gap-4 p-6">
      <h1 className="font-heading text-lg font-medium">Store</h1>

      <Input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={handleSearchKeyDown}
        placeholder="Search Hugging Face apps and models..."
        className="max-w-sm"
      />

      {addError && (
        <Alert variant="destructive">
          <AlertTitle>Could not add</AlertTitle>
          <AlertDescription>{addError}</AlertDescription>
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
                  <CardFooter className="mt-auto">
                    <AddButton
                      added={addedSpaces.has(space.id)}
                      adding={addingId === space.id}
                      disabled={space.compat === "incompatible"}
                      onAdd={() => handleAddSpace(space)}
                    />
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
                    <AddButton
                      added={addedModelRepos.has(model.id)}
                      adding={false}
                      disabled={model.compat === "incompatible"}
                      onAdd={() => setDialogModel(model)}
                    />
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
                  <span className="flex items-center gap-1.5">
                    {dialogModel && addedModels.has(modelKey(dialogModel.id, file.quant)) && (
                      <Badge variant="outline">Added</Badge>
                    )}
                    <FitBadge fits={file.fits} />
                  </span>
                </button>
              ))}
            </div>
          )}

          <DialogFooter>
            <Button variant="outline" onClick={() => setDialogModel(null)}>
              Cancel
            </Button>
            <Button
              disabled={
                !selectedFile ||
                selectedFile.fits === "no" ||
                adding ||
                (dialogModel !== null &&
                  addedModels.has(modelKey(dialogModel.id, selectedFile.quant)))
              }
              onClick={handleAddModel}
            >
              {adding ? "Adding..." : "Add"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
