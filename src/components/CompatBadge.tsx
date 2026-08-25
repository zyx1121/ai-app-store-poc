import type { Compat } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

const LABEL: Record<Compat, string> = {
  ready: "Ready",
  maybe: "Maybe",
  incompatible: "Incompatible",
};

const VARIANT: Record<Compat, "default" | "secondary" | "destructive"> = {
  ready: "default",
  maybe: "secondary",
  incompatible: "destructive",
};

export function CompatBadge({
  compat,
  reason,
}: {
  compat: Compat;
  reason?: string | null;
}) {
  const badge = <Badge variant={VARIANT[compat]}>{LABEL[compat]}</Badge>;

  if (!reason) return badge;

  return (
    <Tooltip>
      <TooltipTrigger render={<span className="inline-flex" />}>
        {badge}
      </TooltipTrigger>
      <TooltipContent>{reason}</TooltipContent>
    </Tooltip>
  );
}
