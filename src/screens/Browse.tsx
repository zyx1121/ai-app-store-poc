import { useEffect, useState, type KeyboardEvent } from "react";
import {
  buildSpace,
  launchModel,
  launchSpace,
  modelFiles,
  searchModels,
  searchSpaces,
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

function CardGridSkeleton() {
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {Array.from({ length: 6 }).map((_, i) => (
        <Skeleton key={i} className="h-40 w-full" />
      ))}
    </div>
  );
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

export function Browse({ onLaunched }: { onLaunched: () => void }) {
  const [query, setQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [tab, setTab] = useState<"apps" | "models">("apps");

  const [spaces, setSpaces] = useState<SpaceSummary[]>([]);
  const [spacesState, setSpacesState] = useState<LoadState>("loading");

  const [models, setModels] = useState<ModelSummary[]>([]);
  const [modelsState, setModelsState] = useState<LoadState>("loading");

  const [launchingId, setLaunchingId] = useState<string | null>(null);
  const [buildingId, setBuildingId] = useState<string | null>(null);
  const [launchError, setLaunchError] = useState<string | null>(null);
  const [spacesError, setSpacesError] = useState<string | null>(null);
  const [modelsError, setModelsError] = useState<string | null>(null);

  const [dialogModel, setDialogModel] = useState<ModelSummary | null>(null);
  const [files, setFiles] = useState<GgufFile[]>([]);
  const [filesState, setFilesState] = useState<LoadState>("loading");
  const [selectedFile, setSelectedFile] = useState<GgufFile | null>(null);
  const [launching, setLaunching] = useState(false);
  const { gate, dialog: gpuDialog } = useGpuGate();

  useEffect(() => {
    const t = setTimeout(() => setDebouncedQuery(query), 400);
    return () => clearTimeout(t);
  }, [query]);

  useEffect(() => {
    let cancelled = false;
    setSpacesState("loading");
    searchSpaces(debouncedQuery)
      .then((res) => {
        if (cancelled) return;
        setSpaces(res);
        setSpacesState("idle");
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setSpacesError(String(e));
        setSpacesState("error");
      });
    return () => {
      cancelled = true;
    };
  }, [debouncedQuery]);

  useEffect(() => {
    let cancelled = false;
    setModelsState("loading");
    searchModels(debouncedQuery)
      .then((res) => {
        if (cancelled) return;
        setModels(res);
        setModelsState("idle");
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setModelsError(String(e));
        setModelsState("error");
      });
    return () => {
      cancelled = true;
    };
  }, [debouncedQuery]);

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

  async function runSpace(space: SpaceSummary) {
    setLaunchError(null);
    setLaunchingId(space.id);
    try {
      if (
        spaceWantsGpu(space) &&
        !(await gate({ kind: "space", id: space.id }, space.title ?? space.name))
      )
        return;
      await launchSpace(space.id);
      onLaunched();
    } catch (e) {
      setLaunchError(String(e));
    } finally {
      setLaunchingId(null);
    }
  }

  async function runBuildSpace(space: SpaceSummary) {
    setLaunchError(null);
    setBuildingId(space.id);
    try {
      if (
        spaceWantsGpu(space) &&
        !(await gate({ kind: "space", id: space.id }, space.title ?? space.name))
      )
        return;
      await buildSpace(space.id);
      onLaunched();
    } catch (e) {
      setLaunchError(String(e));
    } finally {
      setBuildingId(null);
    }
  }

  async function confirmLaunchModel() {
    if (!dialogModel || !selectedFile) return;
    setLaunching(true);
    setLaunchError(null);
    try {
      const tag = `hf.co/${dialogModel.id}:${selectedFile.quant}`;
      if (
        !(await gate({ kind: "model", tag, size_bytes: selectedFile.size_bytes }, dialogModel.name))
      )
        return;
      await launchModel(dialogModel.id, selectedFile.quant);
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
      <Input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={handleSearchKeyDown}
        placeholder="Search Hugging Face..."
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

        <TabsContent value="apps" className="mt-4">
          {spacesState === "loading" && <CardGridSkeleton />}
          {spacesState === "error" && (
            <Alert variant="destructive">
              <AlertTitle>Could not load apps</AlertTitle>
              <AlertDescription>{spacesError ?? "Search Hugging Face Spaces failed. Try again."}</AlertDescription>
            </Alert>
          )}
          {spacesState === "idle" && spaces.length === 0 && (
            <p className="text-sm text-muted-foreground">No apps found.</p>
          )}
          {spacesState === "idle" && spaces.length > 0 && (
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
                    {space.sdk && <Badge variant="outline">{space.sdk}</Badge>}
                    <span>{formatCount(space.likes)} likes</span>
                    {space.hardware && <span>{space.hardware}</span>}
                  </CardContent>
                  <CardFooter className="flex gap-2">
                    <Button
                      className="flex-1"
                      disabled={space.compat === "incompatible" || launchingId === space.id}
                      onClick={() => runSpace(space)}
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
                              onClick={() => runBuildSpace(space)}
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
        </TabsContent>

        <TabsContent value="models" className="mt-4">
          {modelsState === "loading" && <CardGridSkeleton />}
          {modelsState === "error" && (
            <Alert variant="destructive">
              <AlertTitle>Could not load models</AlertTitle>
              <AlertDescription>{modelsError ?? "Search Hugging Face models failed. Try again."}</AlertDescription>
            </Alert>
          )}
          {modelsState === "idle" && models.length === 0 && (
            <p className="text-sm text-muted-foreground">No models found.</p>
          )}
          {modelsState === "idle" && models.length > 0 && (
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
                  <CardFooter>
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
    </div>
  );
}
