import { useEffect, useState } from "react";
import { Camera, MessageSquare, Mic, Palette, Play, Settings, Store } from "lucide-react";
import { runtimeStatus, type RuntimeStatus } from "@/lib/api";
import { cn } from "@/lib/utils";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Setup } from "@/screens/Setup";
import { Browse } from "@/screens/Browse";
import { Running } from "@/screens/Running";
import { Chat } from "@/screens/Chat";
import { Audio } from "@/screens/Audio";
import { Vision } from "@/screens/Vision";
import { Canvas } from "@/screens/Canvas";

type Route = "browse" | "running" | "chat" | "audio" | "vision" | "canvas" | "setup";

function App() {
  const [status, setStatus] = useState<RuntimeStatus | null>(null);
  const [route, setRoute] = useState<Route>("browse");
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

  const navItems: { key: Route; label: string; icon: typeof Store }[] = [
    { key: "browse", label: "Browse", icon: Store },
    { key: "running", label: "Running", icon: Play },
    { key: "chat", label: "Chat", icon: MessageSquare },
    { key: "audio", label: "Audio", icon: Mic },
    { key: "vision", label: "Vision", icon: Camera },
    { key: "canvas", label: "Canvas", icon: Palette },
    { key: "setup", label: "Setup", icon: Settings },
  ];

  return (
    <TooltipProvider>
      <div className="flex h-screen w-screen overflow-hidden bg-background text-foreground">
        <aside className="flex w-48 shrink-0 flex-col gap-1 border-r border-border p-3">
          <div className="px-2 py-2 font-heading text-sm font-medium">AI App Store</div>
          {navItems.map(({ key, label, icon: Icon }) => (
            <button
              key={key}
              type="button"
              disabled={needsSetup && key !== "setup"}
              onClick={() => setRoute(key)}
              className={cn(
                "flex items-center gap-2 rounded-lg px-2.5 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:pointer-events-none disabled:opacity-50",
                activeScreen === key && "bg-muted text-foreground",
              )}
            >
              <Icon className="size-4" />
              {label}
            </button>
          ))}
        </aside>

        <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
          {mounted("setup") && (
            <div className={paneClass("setup")}>
              <Setup status={status} onStatusChange={setStatus} />
            </div>
          )}
          {mounted("browse") && (
            <div className={paneClass("browse")}>
              <Browse onLaunched={() => setRoute("running")} />
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
          {mounted("audio") && (
            <div className={paneClass("audio")}>
              <Audio />
            </div>
          )}
          {mounted("vision") && (
            <div className={paneClass("vision")}>
              <Vision active={activeScreen === "vision"} />
            </div>
          )}
          {mounted("canvas") && (
            <div className={paneClass("canvas")}>
              <Canvas />
            </div>
          )}
        </main>
      </div>
    </TooltipProvider>
  );
}

export default App;
