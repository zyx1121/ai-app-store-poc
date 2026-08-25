/** Compact number formatting, e.g. `12748404 -> "12.7M"`. */
export function formatCount(n: number): string {
  const abs = Math.abs(n);
  const scaled = (value: number, suffix: string) => {
    const s = value.toFixed(1);
    return `${s.endsWith(".0") ? s.slice(0, -2) : s}${suffix}`;
  };
  if (abs >= 1e9) return scaled(n / 1e9, "B");
  if (abs >= 1e6) return scaled(n / 1e6, "M");
  if (abs >= 1e3) return scaled(n / 1e3, "K");
  return String(n);
}

/** Human-readable file size, e.g. `4800000000 -> "4.5 GB"`. */
export function formatBytes(bytes: number): string {
  const gb = bytes / 1024 ** 3;
  if (gb >= 1) return `${gb.toFixed(1)} GB`;
  const mb = bytes / 1024 ** 2;
  if (mb >= 1) return `${mb.toFixed(0)} MB`;
  const kb = bytes / 1024;
  return `${kb.toFixed(0)} KB`;
}
