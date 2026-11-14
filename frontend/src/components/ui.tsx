import clsx from "clsx";
import { AlertTriangle, CheckCircle2, ChevronDown, Info, XCircle } from "lucide-react";
import { useState, type ReactNode } from "react";
import { prettyStatus, tone, type Tone } from "../lib";

/* ---------------------------------------------------------------- layout */
export function Page({ title, subtitle, actions, children }: { title: string; subtitle?: ReactNode; actions?: ReactNode; children: ReactNode }) {
  return (
    <div className="mx-auto w-full max-w-[1200px] px-4 py-6 sm:px-6 lg:px-8">
      <header className="mb-6 flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
          {subtitle && <p className="mt-1 max-w-3xl text-sm text-muted">{subtitle}</p>}
        </div>
        {actions && <div className="flex items-center gap-2">{actions}</div>}
      </header>
      <div className="space-y-6">{children}</div>
    </div>
  );
}

export function Card({ title, description, actions, children, className, pad = true }: {
  title?: ReactNode; description?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string; pad?: boolean;
}) {
  return (
    <section className={clsx("rounded-xl border border-line bg-surface shadow-[0_1px_2px_rgba(15,23,42,0.04)]", className)}>
      {(title || actions) && (
        <div className="flex flex-wrap items-start justify-between gap-2 border-b border-line px-5 py-3.5">
          <div className="min-w-0">
            {title && <h2 className="text-sm font-semibold">{title}</h2>}
            {description && <p className="mt-0.5 text-xs text-muted">{description}</p>}
          </div>
          {actions}
        </div>
      )}
      <div className={pad ? "p-5" : ""}>{children}</div>
    </section>
  );
}

export function Stat({ label, value, hint, tone: t = "neutral" }: { label: string; value: ReactNode; hint?: ReactNode; tone?: Tone }) {
  const color = { neutral: "text-fg", good: "text-good", warn: "text-warn", bad: "text-bad", accent: "text-accent" }[t];
  return (
    <div className="rounded-xl border border-line bg-surface px-4 py-3.5">
      <div className="text-xs font-medium uppercase tracking-wide text-muted">{label}</div>
      <div className={clsx("mt-1 truncate text-2xl font-semibold tabular-nums", color)}>{value}</div>
      {hint && <div className="mt-0.5 text-xs text-muted">{hint}</div>}
    </div>
  );
}

/* ---------------------------------------------------------------- small elements */
const TONES: Record<Tone, string> = {
  neutral: "bg-surface-2 text-muted",
  good: "bg-good-soft text-good",
  warn: "bg-warn-soft text-warn",
  bad: "bg-bad-soft text-bad",
  accent: "bg-accent-soft text-accent",
};

export function Badge({ children, tone: t = "neutral", className }: { children: ReactNode; tone?: Tone; className?: string }) {
  return <span className={clsx("inline-flex items-center gap-1 whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium", TONES[t], className)}>{children}</span>;
}

/** Badge whose colour is derived from the status word itself. */
export const Status = ({ value }: { value: string | null | undefined }) => <Badge tone={tone(value)}>{prettyStatus(value) || "–"}</Badge>;

export function Dot({ color }: { color: string }) {
  return <span className="inline-block h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: color }} />;
}

export function Button({ variant = "secondary", size = "md", className, ...rest }: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" | "ghost" | "danger"; size?: "sm" | "md" }) {
  const v = {
    primary: "bg-accent text-accent-fg hover:opacity-90",
    secondary: "border border-line bg-surface hover:bg-surface-2",
    ghost: "hover:bg-surface-2",
    danger: "bg-bad text-white hover:opacity-90",
  }[variant];
  return (
    <button
      {...rest}
      className={clsx(
        "inline-flex items-center justify-center gap-2 rounded-lg font-medium transition disabled:cursor-not-allowed disabled:opacity-50",
        size === "sm" ? "px-2.5 py-1.5 text-xs" : "px-3.5 py-2 text-sm",
        v,
        className,
      )}
    />
  );
}

export function ButtonLink({ href, children, variant = "secondary", size = "md" }: { href: string; children: ReactNode; variant?: "primary" | "secondary"; size?: "sm" | "md" }) {
  return (
    <a
      href={href}
      className={clsx(
        "inline-flex items-center justify-center gap-2 rounded-lg font-medium transition",
        size === "sm" ? "px-2.5 py-1.5 text-xs" : "px-3.5 py-2 text-sm",
        variant === "primary" ? "bg-accent text-accent-fg hover:opacity-90" : "border border-line bg-surface hover:bg-surface-2",
      )}
    >
      {children}
    </a>
  );
}

export function Alert({ tone: t = "accent", title, children }: { tone?: Tone; title?: ReactNode; children?: ReactNode }) {
  const Icon = { good: CheckCircle2, warn: AlertTriangle, bad: XCircle, accent: Info, neutral: Info }[t];
  return (
    <div className={clsx("flex gap-3 rounded-xl border px-4 py-3 text-sm", TONES[t], "border-transparent")} role={t === "bad" ? "alert" : undefined}>
      <Icon className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
      <div className="min-w-0 text-fg">
        {title && <div className="font-semibold">{title}</div>}
        {children && <div className={clsx("text-[13px] leading-relaxed", title && "mt-0.5")}>{children}</div>}
      </div>
    </div>
  );
}

export function EmptyState({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="rounded-xl border border-dashed border-line bg-surface px-6 py-12 text-center">
      <h3 className="text-base font-semibold">{title}</h3>
      {children && <p className="mx-auto mt-1.5 max-w-xl text-sm text-muted">{children}</p>}
      {action && <div className="mt-4 flex justify-center">{action}</div>}
    </div>
  );
}

export function ProgressBar({ value, tone: t = "accent" }: { value: number; tone?: Tone }) {
  const bar = { accent: "bg-accent", good: "bg-good", warn: "bg-warn", bad: "bg-bad", neutral: "bg-muted" }[t];
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-surface-2" role="progressbar" aria-valuenow={Math.round(value * 100)} aria-valuemin={0} aria-valuemax={100}>
      <div className={clsx("h-full rounded-full transition-all duration-500", bar)} style={{ width: `${Math.max(2, Math.min(100, value * 100))}%` }} />
    </div>
  );
}

/** A value drawn against its reference (e.g. observed rate against the false-positive floor). */
export function RateBar({ value, floor, color }: { value: number | null; floor?: number | null; color: string }) {
  if (value === null || value === undefined) return <span className="text-muted">–</span>;
  const max = Math.max(value, floor ?? 0, 0.0001) * 1.15;
  return (
    <div className="relative h-2 w-full rounded-full bg-surface-2" title={floor != null ? `observed ${(value * 100).toFixed(1)}% · clean-data floor ${(floor * 100).toFixed(1)}%` : undefined}>
      <div className="h-full rounded-full" style={{ width: `${(value / max) * 100}%`, background: color }} />
      {floor != null && <div className="absolute -top-0.5 h-3 w-0.5 rounded bg-fg/70" style={{ left: `${(floor / max) * 100}%` }} />}
    </div>
  );
}

export function Switch({ checked, onChange, disabled, label }: { checked: boolean; onChange: (v: boolean) => void; disabled?: boolean; label: string }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={clsx("relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition disabled:cursor-not-allowed disabled:opacity-40", checked ? "bg-accent" : "bg-line")}
    >
      <span className={clsx("inline-block h-4.5 w-4.5 rounded-full bg-white shadow transition", checked ? "translate-x-[22px]" : "translate-x-[3px]")} style={{ height: 18, width: 18 }} />
    </button>
  );
}

export function Segmented<T extends string | number>({ value, onChange, options, label }: { value: T; onChange: (v: T) => void; options: { value: T; label: ReactNode }[]; label?: string }) {
  return (
    <div role="group" aria-label={label} className="inline-flex rounded-lg border border-line bg-surface-2 p-0.5">
      {options.map((o) => (
        <button
          key={String(o.value)}
          type="button"
          aria-pressed={o.value === value}
          onClick={() => onChange(o.value)}
          className={clsx("rounded-md px-3 py-1 text-xs font-medium transition", o.value === value ? "bg-surface shadow-sm" : "text-muted hover:text-fg")}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Disclosure({ title, children, defaultOpen = false }: { title: ReactNode; children: ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="rounded-lg border border-line">
      <button type="button" aria-expanded={open} onClick={() => setOpen(!open)} className="flex w-full items-center justify-between gap-3 px-4 py-2.5 text-left text-sm font-medium">
        <span className="min-w-0">{title}</span>
        <ChevronDown className={clsx("h-4 w-4 shrink-0 text-muted transition", open && "rotate-180")} />
      </button>
      {open && <div className="border-t border-line px-4 py-3 text-sm text-muted">{children}</div>}
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={clsx("skeleton", className)} />;
}

/* ---------------------------------------------------------------- table */
export interface Col<T> {
  key: string;
  header: ReactNode;
  cell: (row: T) => ReactNode;
  align?: "left" | "right";
  className?: string;
}

export function DataTable<T>({ columns, rows, rowKey, empty = "Nothing to show.", dense = false }: { columns: Col<T>[]; rows: T[]; rowKey: (r: T, i: number) => string; empty?: ReactNode; dense?: boolean }) {
  return (
    <div className="scroll-thin overflow-x-auto">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="border-b border-line text-left text-xs uppercase tracking-wide text-muted">
            {columns.map((c) => (
              <th key={c.key} scope="col" className={clsx("whitespace-nowrap px-3 py-2 font-medium", c.align === "right" && "text-right")}>
                {c.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 && (
            <tr>
              <td colSpan={columns.length} className="px-3 py-8 text-center text-muted">
                {empty}
              </td>
            </tr>
          )}
          {rows.map((r, i) => (
            <tr key={rowKey(r, i)} className="border-b border-line/70 last:border-0 hover:bg-surface-2/60">
              {columns.map((c) => (
                <td key={c.key} className={clsx("px-3 align-top", dense ? "py-1.5" : "py-2.5", c.align === "right" && "text-right tabular-nums", c.className)}>
                  {c.cell(r)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export const Mono = ({ children }: { children: ReactNode }) => <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[12px]">{children}</code>;
