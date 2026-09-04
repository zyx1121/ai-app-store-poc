import { useEffect, useState } from "react";
import { Camera, Image, MessageSquare, Mic, Play, Settings, Store as StoreIcon } from "lucide-react";
import { runtimeStatus, type RuntimeStatus } from "@/lib/api";
import { cn } from "@/lib/utils";
import { TooltipProvider } from "@/components/ui/tooltip";
import { TitleBar } from "@/components/TitleBar";
import { Setup } from "@/screens/Setup";
import { Store } from "@/screens/Store";
import { Running } from "@/screens/Running";
import { Chat } from "@/screens/Chat";
import { Speech } from "@/screens/Speech";
import { Vision } from "@/screens/Vision";
import { Images } from "@/screens/Images";

type Route = "store" | "running" | "chat" | "speech" | "vision" | "images" | "setup";

function App() {
  const [status, setStatus] = useState<RuntimeStatus | null>(null);
  const [route, setRoute] = useState<Route>("store");
  const [chatInstanceId, setChatInstanceId] = useState<string | undefined>();

  useEffect(() => {
    let cancelled = false;
    runtimeStatus().then((s) => {
      if (!cancelled) setStatus(s);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const needsSetup = status === null || !status.ready;
  const activeScreen: Route = needsSetup ? "setup" : route;

  // Screens stay mounted once visited and are hidden, not unmounted, when the
  // user navigates away: a chat transcript, a loaded image with its boxes, a
  // generated picture all survive a round trip (issue #27).
  const [visited, setVisited] = useState<Set<Route>>(() => new Set([activeScreen]));
  useEffect(() => {
    setVisited((prev) => (prev.has(activeScreen) ? prev : new Set(prev).add(activeScreen)));
  }, [activeScreen]);
  const mounted = (key: Route) => visited.has(key) || activeScreen === key;
  const paneClass = (key: Route, scroll = true) =>
    cn("h-full", scroll && "overflow-y-auto", activeScreen !== key && "hidden");

  function openChat(instanceId: string) {
    setChatInstanceId(instanceId);
    setRoute("chat");
  }

  // Each entry names what the screen does in plain words; the hint is the
  // second line, so "Images" reads as text-to-image at a glance, not a gallery.
  const navItems: { key: Route; label: string; hint: string; icon: typeof StoreIcon }[] = [
    { key: "store", label: "Store", hint: "Hugging Face apps and models", icon: StoreIcon },
    { key: "running", label: "Running", hint: "What is launched now", icon: Play },
    { key: "chat", label: "Chat", hint: "Talk to a language model", icon: MessageSquare },
    { key: "speech", label: "Speech", hint: "Speech to text, text to speech", icon: Mic },
    { key: "vision", label: "Vision", hint: "Ask about an image, detect objects", icon: Camera },
    { key: "images", label: "Images", hint: "Text to image, image to image", icon: Image },
    { key: "setup", label: "Setup", hint: "Runtime, hardware, updates", icon: Settings },
  ];

  return (
    <TooltipProvider>
      <div className="flex h-screen w-screen flex-col overflow-hidden bg-background text-foreground">
        <TitleBar />
        <div className="flex min-h-0 flex-1">
          <aside className="flex w-56 shrink-0 flex-col gap-1 border-r border-border p-3">
            {navItems.map(({ key, label, hint, icon: Icon }) => (
              <button
                key={key}
                type="button"
                disabled={needsSetup && key !== "setup"}
                onClick={() => setRoute(key)}
                className={cn(
                  "flex items-start gap-2.5 rounded-lg px-2.5 py-1.5 text-left text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:pointer-events-none disabled:opacity-50",
                  activeScreen === key && "bg-muted text-foreground",
                )}
              >
                <Icon className="mt-0.5 size-4 shrink-0" />
                <span className="flex min-w-0 flex-col">
                  <span className="leading-5">{label}</span>
                  <span className="text-[11px] leading-4 text-muted-foreground/80">{hint}</span>
                </span>
              </button>
            ))}
          </aside>

          <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
            {mounted("setup") && (
              <div className={paneClass("setup")}>
                <Setup status={status} onStatusChange={setStatus} />
              </div>
            )}
            {mounted("store") && (
              <div className={paneClass("store")}>
                <Store onLaunched={() => setRoute("running")} />
              </div>
            )}
            {mounted("running") && (
              <div className={paneClass("running")}>
                <Running onOpenChat={openChat} />
              </div>
            )}
            {mounted("chat") && (
              <div className={paneClass("chat", false)}>
                <Chat initialInstanceId={chatInstanceId} />
              </div>
            )}
            {mounted("speech") && (
              <div className={paneClass("speech")}>
                <Speech />
              </div>
            )}
            {mounted("vision") && (
              <div className={paneClass("vision")}>
                <Vision active={activeScreen === "vision"} />
              </div>
            )}
            {mounted("images") && (
              <div className={paneClass("images")}>
                <Images />
              </div>
            )}
          </main>
        </div>
      </div>
    </TooltipProvider>
  );
}

export default App;
