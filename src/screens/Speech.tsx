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
  WHISPER_BASE_URL,
  installServiceModel,
  listInstances,
  onInstanceUpdate,
  onServiceUpdate,
  serviceModels,
  serviceStatus,
  serviceTouch,
  startService,
  stopService,
  type Instance,
  type ServiceModel,
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
import { ServiceStateBadge } from "@/components/ServiceStateBadge";
import { MessageResponse } from "@/components/ai-elements/message";

const SERVICE_ID = "speaches";
/** GPU speech to text on AMD / Intel; `unavailable` elsewhere. */
const WHISPER_ID = "whisper";

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

/**
 * Decode any browser recording and re-encode it as 16 kHz mono PCM16 WAV, the
 * one format whisper.cpp's server accepts without ffmpeg. Speaches takes it too.
 */
async function toWav16k(blob: Blob): Promise<Blob> {
  const decodeCtx = new AudioContext();
  const decoded = await decodeCtx.decodeAudioData(await blob.arrayBuffer());
  await decodeCtx.close();
  const rate = 16000;
  const length = Math.ceil(decoded.duration * rate);
  const offline = new OfflineAudioContext(1, length, rate);
  const src = offline.createBufferSource();
  src.buffer = decoded;
  src.connect(offline.destination);
  src.start();
  const mono = (await offline.startRendering()).getChannelData(0);
  const buf = new ArrayBuffer(44 + mono.length * 2);
  const v = new DataView(buf);
  const str = (o: number, s: string) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF");
  v.setUint32(4, 36 + mono.length * 2, true);
  str(8, "WAVE");
  str(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, rate, true);
  v.setUint32(28, rate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  str(36, "data");
  v.setUint32(40, mono.length * 2, true);
  for (let i = 0; i < mono.length; i++) {
    const s = Math.max(-1, Math.min(1, mono[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

function formatElapsed(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

export function Speech() {
  const [status, setStatus] = useState<ServiceStatus | null>(null);
  const [serviceBusy, setServiceBusy] = useState(false);
  const [serviceError, setServiceError] = useState<string | null>(null);

  const [whisper, setWhisper] = useState<ServiceStatus | null>(null);
  const [whisperModels, setWhisperModels] = useState<ServiceModel[]>([]);
  const [whisperBusy, setWhisperBusy] = useState(false);
  const [whisperError, setWhisperError] = useState<string | null>(null);
  const [whisperInstalling, setWhisperInstalling] = useState(false);

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
  const whisperOffered = whisper !== null && whisper.state !== "unavailable";
  const whisperRunning = whisper?.state === "running";
  /** where transcriptions go: the GPU whisper.cpp server when it is up, else Speaches */
  const sttBaseUrl = whisperRunning ? WHISPER_BASE_URL : SPEACHES_BASE_URL;
  const sttServiceId = whisperRunning ? WHISPER_ID : SERVICE_ID;

  useEffect(() => {
    let cancelled = false;
    serviceStatus(SERVICE_ID).then((s) => {
      if (!cancelled) setStatus(s);
    });
    serviceStatus(WHISPER_ID).then((s) => {
      if (cancelled) return;
      setWhisper(s);
      if (s.state !== "unavailable") void refreshWhisperModels();
    });
    let unlisten: (() => void) | undefined;
    onServiceUpdate((s) => {
      if (s.id === SERVICE_ID) setStatus(s);
      if (s.id === WHISPER_ID) setWhisper(s);
    }).then((fn) => {
      if (cancelled) fn();
      else unlisten = fn;
    });
    return () => {
      cancelled = true;
      unlisten?.();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function refreshWhisperModels() {
    try {
      setWhisperModels(await serviceModels(WHISPER_ID));
    } catch (err) {
      setWhisperError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleWhisperStart() {
    setWhisperBusy(true);
    setWhisperError(null);
    try {
      setWhisper(await startService(WHISPER_ID));
    } catch (err) {
      setWhisperError(err instanceof Error ? err.message : String(err));
    } finally {
      setWhisperBusy(false);
    }
  }

  async function handleWhisperStop() {
    setWhisperBusy(true);
    setWhisperError(null);
    try {
      await stopService(WHISPER_ID);
    } catch (err) {
      setWhisperError(err instanceof Error ? err.message : String(err));
    } finally {
      setWhisperBusy(false);
    }
  }

  async function handleWhisperModel(name: string) {
    setWhisperInstalling(true);
    setWhisperError(null);
    try {
      await installServiceModel(WHISPER_ID, name);
      await refreshWhisperModels();
    } catch (err) {
      setWhisperError(err instanceof Error ? err.message : String(err));
    } finally {
      setWhisperInstalling(false);
    }
  }

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
    // Held outside the try so the catch can stop it if the constructor below
    // throws after getUserMedia already acquired the mic; otherwise the track
    // (and the OS mic-in-use indicator) stays on until a later recording succeeds.
    let stream: MediaStream | undefined;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      chunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        stream?.getTracks().forEach((t) => t.stop());
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
      stream?.getTracks().forEach((t) => t.stop());
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
      const wav = await toWav16k(blob);
      const form = new FormData();
      form.append("file", wav, "audio.wav");
      form.append("model", sttModel);
      form.append("response_format", "json");
      const res = await fetch(`${sttBaseUrl}/audio/transcriptions`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) throw new Error(`Transcription failed (${res.status})`);
      const body = (await res.json()) as { text: string };
      setTranscript(body.text);
      // This call went straight from the webview to the service; tell the
      // store so its idle-stop timer resets (#65).
      void serviceTouch(sttServiceId);
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
      void serviceTouch(SERVICE_ID);
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
    <div className="flex h-full flex-col gap-3 overflow-y-auto p-6">
      <Card className="shrink-0">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <span>Speaches</span>
            {status && <ServiceStateBadge state={status.state} />}
            {status && <Badge variant="outline">{status.backend}</Badge>}
          </CardTitle>
          <CardDescription>{status?.url ?? SPEACHES_BASE_URL}</CardDescription>
          {status?.backend === "cpu" && (
            <p className="text-xs text-muted-foreground">
              Running on CPU: transcription is slower
            </p>
          )}
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

      {whisperOffered && whisper && (
        <Card className="shrink-0">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <span>whisper.cpp</span>
              <ServiceStateBadge state={whisper.state} />
              <Badge variant="outline">{whisper.backend}</Badge>
            </CardTitle>
            <CardDescription>{whisper.url}</CardDescription>
            <p className="text-xs text-muted-foreground">
              While it runs, recordings below go here instead of Speaches; text to speech stays on
              Speaches.
            </p>
          </CardHeader>
          <CardContent className="flex flex-col gap-2">
            {whisperModels.map((m) => (
              <div key={m.name} className="flex items-center gap-2 text-sm">
                <span>{m.display_name}</span>
                <span className="flex-1" />
                {m.installed ? (
                  <Badge variant="outline">
                    <CircleCheck className="text-foreground" /> Installed
                  </Badge>
                ) : (
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={whisperInstalling}
                    onClick={() => void handleWhisperModel(m.name)}
                  >
                    {whisperInstalling ? <Spinner /> : <Download />}
                    {whisperInstalling ? "Downloading..." : "Download (190 MB)"}
                  </Button>
                )}
              </div>
            ))}
            {(whisper.error || whisperError) && (
              <Alert variant="destructive">
                <AlertDescription>{whisper.error ?? whisperError}</AlertDescription>
              </Alert>
            )}
            <LogTail lines={whisper.log_tail} />
          </CardContent>
          <CardFooter className="flex gap-2">
            <Button
              onClick={() => void handleWhisperStart()}
              disabled={
                whisperBusy ||
                whisper.state === "pulling" ||
                whisper.state === "starting" ||
                whisperRunning ||
                !whisperModels.some((m) => m.installed)
              }
            >
              {whisper.state === "pulling"
                ? "Downloading..."
                : whisper.state === "starting"
                  ? "Starting..."
                  : "Start"}
            </Button>
            <Button
              variant="outline"
              onClick={() => void handleWhisperStop()}
              disabled={whisperBusy || whisper.state === "stopped" || whisper.state === "missing"}
            >
              Stop
            </Button>
          </CardFooter>
        </Card>
      )}

      <fieldset disabled={!running} className="flex shrink-0 flex-col gap-3 disabled:opacity-50">
        <Card>
          <CardHeader>
            <CardTitle>Models</CardTitle>
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
                <p className="text-sm text-muted-foreground">No running models</p>
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
