import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { ServiceState } from "@/lib/api";

const LABEL: Record<ServiceState, string> = {
  missing: "Not installed",
  pulling: "Downloading",
  starting: "Starting",
  running: "Running",
  stopped: "Stopped",
  error: "Error",
  unavailable: "Not on this GPU",
};

const VARIANT: Record<ServiceState, "default" | "secondary" | "destructive" | "outline"> = {
  missing: "outline",
  pulling: "secondary",
  starting: "secondary",
  running: "default",
  stopped: "outline",
  error: "destructive",
  unavailable: "outline",
};

export function ServiceStateBadge({ state }: { state: ServiceState }) {
  const animated = state === "pulling" || state === "starting";
  return (
    <Badge variant={VARIANT[state]} className={cn(animated && "animate-pulse")}>
      {LABEL[state]}
    </Badge>
  );
}
