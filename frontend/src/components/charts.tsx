import type { ReactNode } from "react";

/** Shared Recharts styling so every chart reads from the same theme tokens (works in light and dark). */
export const axisTick = { fill: "var(--muted)", fontSize: 11 };
export const gridProps = { stroke: "var(--grid)", strokeDasharray: "3 3", vertical: false } as const;
export const axisLine = { stroke: "var(--line)" };

interface TipEntry {
  name?: string | number;
  value?: number | string;
  color?: string;
  stroke?: string;
  fill?: string;
  payload?: any;
}

export function ChartTip({ active, payload, label, fmt, title }: { active?: boolean; payload?: TipEntry[]; label?: ReactNode; fmt?: (v: number, e: TipEntry) => string; title?: (label: any, payload: TipEntry[]) => ReactNode }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-line bg-surface px-3 py-2 text-xs shadow-lg">
      <div className="mb-1 font-medium">{title ? title(label, payload) : label}</div>
      {payload.map((p, i) => (
        <div key={i} className="flex items-center gap-2 py-0.5">
          <span className="h-2 w-2 rounded-full" style={{ background: p.color ?? p.stroke ?? p.fill }} />
          <span className="text-muted">{p.name}</span>
          <span className="ml-auto pl-3 font-mono tabular-nums">{typeof p.value === "number" && fmt ? fmt(p.value, p) : String(p.value)}</span>
        </div>
      ))}
    </div>
  );
}

export function ChartLegend({ items }: { items: { label: string; color: string }[] }) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
      {items.map((i) => (
        <span key={i.label} className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ background: i.color }} />
          {i.label}
        </span>
      ))}
    </div>
  );
}
