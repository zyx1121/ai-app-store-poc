import { useEffect, useRef, useState } from "react";
import { generateText } from "ai";
import {
  CircleCheck,
  Download,
  Mic,
  Pause,
  Square,
  TriangleAlert,
  Volume2,
} from "lucide-react";

import {
  SPEACHES_BASE_URL,
  listInstances,
  onInstanceUpdate,
  onServiceUpdate,
  serviceStatus,
  startService,
  stopService,
  type Instance,
  type ServiceState,
  type ServiceStatus,
} from "@/lib/api";
import { ollama } from "@/lib/chat";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Textarea } from "@/components/ui/textarea";
import { Spinner } from "@/components/ui/spinner";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { LogTail } from "@/components/LogTail";
import { MessageResponse } from "@/components/ai-elements/message";
import { cn } from "@/lib/utils";

const SERVICE_ID = "speaches";

const STT_MODELS: { id: string; label: string }[] = [
  { id: "Systran/faster-whisper-small", label: "faster-whisper-small (fast, multilingual)" },
  { id: "deepdml/faster-whisper-large-v3-turbo-ct2", label: "large-v3-turbo (better Chinese)" },
];

const TTS_MODEL = "speaches-ai/Kokoro-82M-v1.0-ONNX";

const TTS_VOICES = ["af_heart", "af_bella", "am_adam", "bm_george", "zf_xiaobei", "zm_yunjian"];

const STT_STORAGE_KEY = "audio.sttModel";
const TTS_STORAGE_KEY = "audio.ttsVoice";

function loadPreference(key: string, fallback: string): string {
  try {
    return localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

function savePreference(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // ignore (private browsing, storage disabled, ...)
  }
}

function isRunningModel(instance: Instance): boolean {
  return instance.kind === "model" && instance.status === "running";
}

const SERVICE_STATE_LABEL: Record<ServiceState, string> = {
  missing: "Not installed",
  pulling: "Pulling image",
  starting: "Starting",
  running: "Running",
  stopped: "Stopped",
  error: "Error",
};

const SERVICE_STATE_VARIANT: Record<
  ServiceState,
  "default" | "secondary" | "destructive" | "outline"
> = {
  missing: "outline",
  pulling: "secondary",
  starting: "secondary",
  running: "default",
  stopped: "outline",
  error: "destructive",
};

function ServiceStateBadge({ state }: { state: ServiceState }) {
  const animated = state === "pulling" || state === "starting";
  return (
    <Badge variant={SERVICE_STATE_VARIANT[state]} className={cn(animated && "animate-pulse")}>
      {SERVICE_STATE_LABEL[state]}
    </Badge>
  );
}

/** Rough Markdown stripping so TTS doesn't read out symbols. */
function stripMarkdown(text: string): string {
  return text
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/`[^`]*`/g, " ")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, " ")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/[*_#>~-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function formatElapsed(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

export function Audio() {
  const [status, setStatus] = useState<ServiceStatus | null>(null);
  const [serviceBusy, setServiceBusy] = useState(false);
  const [serviceError, setServiceError] = useState<string | null>(null);

  const [sttModel, setSttModel] = useState(() =>
    loadPreference(STT_STORAGE_KEY, STT_MODELS[0].id),
  );
  const [ttsVoice, setTtsVoice] = useState(() => loadPreference(TTS_STORAGE_KEY, TTS_VOICES[0]));
  const [installedIds, setInstalledIds] = useState<Set<string>>(new Set());
  const [downloading, setDownloading] = useState<Record<string, boolean>>({});
  const [modelsError, setModelsError] = useState<string | null>(null);

  const [recording, setRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [transcribing, setTranscribing] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [recordError, setRecordError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);

  const [instances, setInstances] = useState<Instance[]>([]);
  const [selectedInstanceId, setSelectedInstanceId] = useState<string | undefined>();
  const [asking, setAsking] = useState(false);
  const [askError, setAskError] = useState<string | null>(null);
  const [reply, setReply] = useState("");
  const [speaking, setSpeaking] = useState(false);
  const [speakError, setSpeakError] = useState<string | null>(null);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioUrlRef = useRef<string | null>(null);

  const running = status?.state === "running";

  useEffect(() => {
    let cancelled = false;
    serviceStatus(SERVICE_ID).then((s) => {
      if (!cancelled) setStatus(s);
    });
    let unlisten: (() => void) | undefined;
    onServiceUpdate((s) => {
      if (s.id !== SERVICE_ID) return;
      setStatus(s);
    }).then((fn) => {
      if (cancelled) fn();
      else unlisten = fn;
    });
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    listInstances().then((res) => {
      if (!cancelled) setInstances(res.filter(isRunningModel));
    });
    let unlisten: (() => void) | undefined;
    onInstanceUpdate((instance) => {
      if (instance.kind !== "model") return;
      setInstances((prev) => {
        const withoutIt = prev.filter((i) => i.id !== instance.id);
        return instance.status === "running" ? [instance, ...withoutIt] : withoutIt;
      });
    }).then((fn) => {
      if (cancelled) fn();
      else unlisten = fn;
    });
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  useEffect(() => {
    if (selectedInstanceId && instances.some((i) => i.id === selectedInstanceId)) return;
    setSelectedInstanceId(instances[0]?.id);
  }, [instances, selectedInstanceId]);

  async function refreshInstalled() {
    try {
      const res = await fetch(`${SPEACHES_BASE_URL}/models`);
      if (!res.ok) throw new Error(`Failed to list models (${res.status})`);
      const body = (await res.json()) as { data: { id: string }[] };
      setInstalledIds(new Set(body.data.map((m) => m.id)));
      setModelsError(null);
    } catch (err) {
      setModelsError(err instanceof Error ? err.message : String(err));
    }
  }

  useEffect(() => {
    if (running) void refreshInstalled();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running]);

  useEffect(() => {
    return () => {
      if (timerRef.current !== null) window.clearInterval(timerRef.current);
      if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
    };
  }, []);

  async function handleStart() {
    setServiceBusy(true);
    setServiceError(null);
    try {
      const s = await startService(SERVICE_ID);
      setStatus(s);
    } catch (err) {
      setServiceError(err instanceof Error ? err.message : String(err));
    } finally {
      setServiceBusy(false);
    }
  }

  async function handleStop() {
    setServiceBusy(true);
    setServiceError(null);
    try {
      await stopService(SERVICE_ID);
    } catch (err) {
      setServiceError(err instanceof Error ? err.message : String(err));
    } finally {
      setServiceBusy(false);
    }
  }

  async function downloadModel(id: string) {
    setDownloading((prev) => ({ ...prev, [id]: true }));
    setModelsError(null);
    try {
      const res = await fetch(`${SPEACHES_BASE_URL}/models/${id}`, { method: "POST" });
      if (!res.ok) throw new Error(`Download failed (${res.status})`);
      await refreshInstalled();
    } catch (err) {
      setModelsError(err instanceof Error ? err.message : String(err));
    } finally {
      setDownloading((prev) => ({ ...prev, [id]: false }));
    }
  }

  function ModelChip({ id }: { id: string }) {
    const installed = installedIds.has(id);
    const busy = downloading[id] ?? false;
    if (installed) {
      return (
        <Badge variant="outline">
          <CircleCheck className="text-foreground" /> Installed
        </Badge>
      );
    }
    return (
      <Button variant="outline" size="sm" disabled={busy} onClick={() => downloadModel(id)}>
        {busy ? <Spinner /> : <Download />}
        {busy ? "Downloading..." : "Download"}
      </Button>
    );
  }

  function selectSttModel(id: string) {
    setSttModel(id);
    savePreference(STT_STORAGE_KEY, id);
  }

  function selectTtsVoice(voice: string) {
    setTtsVoice(voice);
    savePreference(TTS_STORAGE_KEY, voice);
  }

  async function startRecording() {
    setRecordError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      chunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        void handleRecordingStopped();
      };
      mediaRecorderRef.current = recorder;
      recorder.start();
      setRecording(true);
      setRecordingSeconds(0);
      timerRef.current = window.setInterval(() => {
        setRecordingSeconds((s) => s + 1);
      }, 1000);
    } catch (err) {
      setRecordError(err instanceof Error ? err.message : String(err));
    }
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
    setRecording(false);
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }

  async function handleRecordingStopped() {
    const blob = new Blob(chunksRef.current, { type: "audio/webm" });
    chunksRef.current = [];
    if (blob.size === 0) return;
    setTranscribing(true);
    setRecordError(null);
    try {
      const form = new FormData();
      form.append("file", blob, "audio.webm");
      form.append("model", sttModel);
      form.append("response_format", "json");
      const res = await fetch(`${SPEACHES_BASE_URL}/audio/transcriptions`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) throw new Error(`Transcription failed (${res.status})`);
      const body = (await res.json()) as { text: string };
      setTranscript(body.text);
    } catch (err) {
      setRecordError(err instanceof Error ? err.message : String(err));
    } finally {
      setTranscribing(false);
    }
  }

  function toggleRecording() {
    if (recording) stopRecording();
    else void startRecording();
  }

  async function speak(text: string) {
    const input = stripMarkdown(text);
    if (!input) return;
    setSpeaking(true);
    setSpeakError(null);
    try {
      const res = await fetch(`${SPEACHES_BASE_URL}/audio/speech`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          input,
          model: TTS_MODEL,
          voice: ttsVoice,
          response_format: "mp3",
        }),
      });
      if (!res.ok) throw new Error(`Speech synthesis failed (${res.status})`);
      const bytes = await res.blob();
      if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
      const url = URL.createObjectURL(bytes);
      audioUrlRef.current = url;
      if (audioRef.current) {
        audioRef.current.src = url;
        await audioRef.current.play();
      }
    } catch (err) {
      setSpeakError(err instanceof Error ? err.message : String(err));
    } finally {
      setSpeaking(false);
    }
  }

  function stopPlayback() {
    audioRef.current?.pause();
    if (audioRef.current) audioRef.current.currentTime = 0;
  }

  const selectedInstance = instances.find((i) => i.id === selectedInstanceId);

  async function askModel() {
    const prompt = transcript.trim();
    if (!prompt || !selectedInstance) return;
    setAsking(true);
    setAskError(null);
    setReply("");
    try {
      const result = await generateText({
        model: ollama(selectedInstance.model_tag ?? ""),
        prompt,
      });
      setReply(result.text);
      await speak(result.text);
    } catch (err) {
      setAskError(err instanceof Error ? err.message : String(err));
    } finally {
      setAsking(false);
    }
  }

  return (
    <div className="flex h-full flex-col gap-3 p-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <span>Speaches</span>
            {status && <ServiceStateBadge state={status.state} />}
          </CardTitle>
          <CardDescription>{status?.url ?? SPEACHES_BASE_URL}</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-2">
          {status?.error && <p className="text-xs text-destructive">{status.error}</p>}
          {serviceError && (
            <Alert variant="destructive">
              <AlertDescription>{serviceError}</AlertDescription>
            </Alert>
          )}
          {status && <LogTail lines={status.log_tail} />}
        </CardContent>
        <CardFooter className="flex gap-2">
          <Button
            onClick={handleStart}
            disabled={
              serviceBusy || status?.state === "pulling" || status?.state === "starting" || running
            }
          >
            {status?.state === "pulling"
              ? "Pulling..."
              : status?.state === "starting"
                ? "Starting..."
                : "Start"}
          </Button>
          <Button
            variant="outline"
            onClick={handleStop}
            disabled={serviceBusy || !status || status.state === "stopped" || status.state === "missing"}
          >
            Stop
          </Button>
        </CardFooter>
      </Card>

      <fieldset disabled={!running} className="flex flex-col gap-3 disabled:opacity-50">
        <Card>
          <CardHeader>
            <CardTitle>Models</CardTitle>
            <CardDescription>Speech-to-text and text-to-speech models used below.</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            {modelsError && (
              <Alert variant="destructive">
                <AlertDescription>{modelsError}</AlertDescription>
              </Alert>
            )}
            <div className="flex items-center gap-2">
              <Select
                value={sttModel}
                onValueChange={(v) => v && selectSttModel(v)}
                items={Object.fromEntries(STT_MODELS.map((m) => [m.id, m.label]))}
              >
                <SelectTrigger className="w-80">
                  <SelectValue placeholder="STT model" />
                </SelectTrigger>
                <SelectContent>
                  {STT_MODELS.map((m) => (
                    <SelectItem key={m.id} value={m.id}>
                      {m.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <ModelChip id={sttModel} />
            </div>
            <div className="flex items-center gap-2">
              <Select
                value={ttsVoice}
                onValueChange={(v) => v && selectTtsVoice(v)}
                items={Object.fromEntries(TTS_VOICES.map((v) => [v, v]))}
              >
                <SelectTrigger className="w-80">
                  <SelectValue placeholder="TTS voice" />
                </SelectTrigger>
                <SelectContent>
                  {TTS_VOICES.map((v) => (
                    <SelectItem key={v} value={v}>
                      {v}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <ModelChip id={TTS_MODEL} />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Record</CardTitle>
            <CardDescription>Record a clip, then transcribe it.</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            <div className="flex items-center gap-3">
              <Button
                size="icon-lg"
                variant={recording ? "destructive" : "default"}
                onClick={toggleRecording}
              >
                {recording ? <Square /> : <Mic />}
              </Button>
              {recording && (
                <Badge variant="destructive" className="animate-pulse">
                  Recording {formatElapsed(recordingSeconds)}
                </Badge>
              )}
              {transcribing && (
                <span className="flex items-center gap-1.5 text-sm text-muted-foreground">
                  <Spinner /> Transcribing...
                </span>
              )}
            </div>
            {recordError && (
              <Alert variant="destructive">
                <TriangleAlert />
                <AlertDescription>{recordError}</AlertDescription>
              </Alert>
            )}
            <Textarea
              value={transcript}
              onChange={(e) => setTranscript(e.target.value)}
              placeholder="Transcript will appear here. You can also type or edit it directly."
              className="min-h-24"
            />
            <div className="flex gap-2">
              <Button variant="outline" onClick={() => void speak(transcript)} disabled={!transcript.trim() || speaking}>
                {speaking ? <Spinner /> : <Volume2 />} Speak
              </Button>
              <Button variant="outline" onClick={stopPlayback}>
                <Pause /> Stop
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Ask a model</CardTitle>
            <CardDescription>Send the transcript to a running model and hear the reply.</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            <div className="flex items-center gap-2">
              <Select
                value={selectedInstanceId ?? null}
                onValueChange={(v) => setSelectedInstanceId(v ?? undefined)}
                items={Object.fromEntries(instances.map((i) => [i.id, i.display_name]))}
              >
                <SelectTrigger className="w-72">
                  <SelectValue placeholder="Select a running model" />
                </SelectTrigger>
                <SelectContent>
                  {instances.map((instance) => (
                    <SelectItem key={instance.id} value={instance.id}>
                      {instance.display_name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {instances.length === 0 && (
                <p className="text-sm text-muted-foreground">No running models. Launch one from Browse.</p>
              )}
              <span className="flex-1" />
              <Button
                onClick={() => void askModel()}
                disabled={!transcript.trim() || !selectedInstance || asking}
              >
                {asking ? <Spinner /> : null} Ask model
              </Button>
            </div>
            {askError && (
              <Alert variant="destructive">
                <AlertDescription>{askError}</AlertDescription>
              </Alert>
            )}
            {speakError && (
              <Alert variant="destructive">
                <AlertDescription>{speakError}</AlertDescription>
              </Alert>
            )}
            {reply && (
              <div className="rounded-xl border border-border p-3">
                <MessageResponse>{reply}</MessageResponse>
              </div>
            )}
            <audio ref={audioRef} className="hidden" />
          </CardContent>
        </Card>
      </fieldset>
    </div>
  );
}
