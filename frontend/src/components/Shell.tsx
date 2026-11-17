import clsx from "clsx";
import { BadgeCheck, BrainCircuit, ClipboardCheck, Columns3, FileDown, FlaskConical, Gauge, HeartPulse, LayoutDashboard, Menu, Moon, SearchCheck, ShieldCheck, Sun, TrendingDown, UploadCloud, Wrench, X } from "lucide-react";
import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { api } from "../api";
import { lastAudit, useTheme } from "../lib";

function Brand() {
  return (
    <div className="flex items-center gap-2.5 px-2">
      <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent text-accent-fg">
        <ShieldCheck className="h-4.5 w-4.5" style={{ height: 18, width: 18 }} />
      </div>
      <div className="leading-tight">
        <div className="text-[15px] font-semibold tracking-tight">TaskClean</div>
        <div className="text-[11px] text-muted">Task-aware data quality</div>
      </div>
    </div>
  );
}

const linkCls = ({ isActive }: { isActive: boolean }) =>
  clsx(
    "flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-[13.5px] font-medium transition",
    isActive ? "bg-accent-soft text-accent" : "text-muted hover:bg-surface-2 hover:text-fg",
  );

function NavGroup({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="px-2.5 pb-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted/80">{title}</div>
      <nav className="space-y-0.5">{children}</nav>
    </div>
  );
}

function SidebarBody({ auditId, onNavigate }: { auditId: string | null; onNavigate: () => void }) {
  const { theme, toggle } = useTheme();
  const base = auditId ? `/audit/${auditId}` : null;
  // same query key as AuditLayout, so this is served from cache once an audit page has loaded
  const audit = useQuery({ queryKey: ["audit", auditId], queryFn: () => api.audit(auditId!), enabled: !!auditId });
  const pending = audit.data?.review.total ?? 0;
  const dropCandidates = audit.data?.features?.filter((f) => f.recommendation.startsWith("drop")).length ?? 0;
  const item = (to: string, label: string, Icon: typeof Gauge, end = false, count = 0) => (
    <NavLink key={to} to={to} end={end} className={linkCls} onClick={onNavigate}>
      <Icon className="h-4 w-4 shrink-0" />
      {label}
      {count > 0 && <span className="ml-auto rounded-full bg-warn-soft px-1.5 py-0.5 text-[11px] font-semibold tabular-nums text-warn">{count >= 10000 ? `${(count / 1000).toFixed(count >= 100000 ? 0 : 1)}k` : count.toLocaleString()}</span>}
    </NavLink>
  );
  return (
    <div className="flex h-full flex-col gap-6 overflow-y-auto px-3 py-5">
      <Brand />
      <NavGroup title="Workspace">{item("/", "New audit", UploadCloud, true)}{item("/proof", "Controlled proof", BadgeCheck)}</NavGroup>
      {base && (
        <NavGroup title="Current audit">
          {item(base, "Overview", LayoutDashboard, true)}
          {item(`${base}/baseline`, "Baseline health", HeartPulse)}
          {item(`${base}/quality`, "Quality audit", SearchCheck)}
          {item(`${base}/features`, "Features", Columns3, false, dropCandidates)}
          {item(`${base}/calibration`, "Calibration", Gauge)}
          {item(`${base}/readiness`, "Readiness", ShieldCheck)}
          {item(`${base}/impact`, "Task impact", TrendingDown)}
          {item(`${base}/panel`, "Model panel", BrainCircuit)}
          {item(`${base}/repair`, "Repair plan", Wrench)}
          {item(`${base}/review`, "Review queue", ClipboardCheck, false, pending)}
          {item(`${base}/outputs`, "Outputs", FileDown)}
        </NavGroup>
      )}
      <NavGroup title="Reference">{item("/research", "Research results", FlaskConical)}</NavGroup>
      <div className="mt-auto flex items-center justify-between rounded-lg border border-line px-3 py-2">
        <span className="text-xs text-muted">{theme === "dark" ? "Dark" : "Light"} theme</span>
        <button onClick={toggle} aria-label="Toggle theme" className="rounded-md p-1.5 text-muted hover:bg-surface-2 hover:text-fg">
          {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </button>
      </div>
    </div>
  );
}

export function Shell() {
  const loc = useLocation();
  const [open, setOpen] = useState(false);
  const auditId = loc.pathname.match(/^\/audit\/([^/]+)/)?.[1] ?? lastAudit();
  useEffect(() => setOpen(false), [loc.pathname]);
  return (
    <div className="flex min-h-full">
      <aside className="sticky top-0 hidden h-screen w-60 shrink-0 border-r border-line bg-surface lg:block">
        <SidebarBody auditId={auditId} onNavigate={() => {}} />
      </aside>

      {open && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-black/40" onClick={() => setOpen(false)} />
          <aside className="absolute inset-y-0 left-0 w-64 bg-surface shadow-xl">
            <button onClick={() => setOpen(false)} aria-label="Close menu" className="absolute right-2 top-3 rounded-md p-1.5 text-muted hover:bg-surface-2">
              <X className="h-4 w-4" />
            </button>
            <SidebarBody auditId={auditId} onNavigate={() => setOpen(false)} />
          </aside>
        </div>
      )}

      <div className="min-w-0 flex-1">
        <div className="sticky top-0 z-30 flex items-center gap-3 border-b border-line bg-surface/90 px-4 py-2.5 backdrop-blur lg:hidden">
          <button onClick={() => setOpen(true)} aria-label="Open menu" className="rounded-md p-1.5 hover:bg-surface-2">
            <Menu className="h-5 w-5" />
          </button>
          <span className="text-sm font-semibold">TaskClean</span>
        </div>
        <main>
          <Outlet />
        </main>
      </div>
    </div>
  );
}
