import type { InstanceStatus } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

const LABEL: Record<InstanceStatus, string> = {
  pulling: "Pulling",
  building: "Building",
  starting: "Starting",
  running: "Running",
  error: "Error",
  stopped: "Stopped",
};

const VARIANT: Record<
  InstanceStatus,
  "default" | "secondary" | "destructive" | "outline"
> = {
  pulling: "secondary",
  building: "secondary",
  starting: "secondary",
  running: "default",
  error: "destructive",
  stopped: "outline",
};

export function StatusBadge({ status }: { status: InstanceStatus }) {
  const animated = status === "pulling" || status === "building" || status === "starting";
  return (
    <Badge variant={VARIANT[status]} className={cn(animated && "animate-pulse")}>
      {LABEL[status]}
    </Badge>
  );
}
