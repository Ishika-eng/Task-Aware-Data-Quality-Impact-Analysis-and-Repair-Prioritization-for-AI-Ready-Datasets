import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Download, Loader2, RotateCw } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type ModelPanel, type PanelCell } from "../api";
import { useAudit } from "../audit-context";
import { axisLine, axisTick, ChartLegend, ChartTip, gridProps } from "../components/charts";
import { Alert, Badge, Button, ButtonLink, Card, DataTable, EmptyState, Page, ProgressBar, Stat, type Col } from "../components/ui";
import { int, num, pct, seconds, signed } from "../lib";

const MODEL_COLOR: Record<string, string> = {
  logistic_regression: "var(--c-label)",
  ridge: "var(--c-label)",
  random_forest: "var(--c-outliers)",
  gradient_boosting: "var(--c-missing)",
  neural_net: "var(--c-duplicates)",
};

const verdictTone = (v: string) => (v === "improved" ? "good" : v === "worse" ? "bad" : "neutral");

/** One delta with its honest label. A change inside the noise carries no percentage. */
function Delta({ c }: { c: PanelCell }) {
  if (c.verdict === "baseline") return <span className="text-muted">baseline</span>;
  if (c.verdict === "no change") return <Badge>no change</Badge>;
  if (c.verdict === "no effect") return <span title="Cells were repaired, but the trained model came out identical to the raw one (for example, the fill equals the model's own default fill)."><Badge>no effect</Badge></span>;
  return (
    <span className="inline-flex flex-wrap items-center justify-end gap-1.5">
      <b className="tabular-nums">{signed(c.delta_f1)}</b>
      <Badge tone={verdictTone(c.verdict)}>{c.verdict}</Badge>
      {c.delta_pct !== null && <span className="text-xs text-muted tabular-nums">{c.delta_pct >= 0 ? "+" : "−"}{Math.abs(c.delta_pct).toFixed(1)}%</span>}
    </span>
  );
}

function Running({ p }: { p: ModelPanel }) {
  return (
    <Card title="Training the model panel" description="Four models, raw rows against repaired rows, same split and seed. This runs in the background; you can keep using the dashboard.">
      <div className="flex items-center gap-3">
        <Loader2 className="h-5 w-5 animate-spin text-accent" />
        <div className="min-w-0 flex-1">
          <ProgressBar value={p.progress ?? 0} />
          <p className="mt-2 truncate text-sm text-muted">{p.status === "queued" ? "Waiting for the audit to finish" : p.message}</p>
        </div>
        <span className="text-sm tabular-nums text-muted">{pct(p.progress ?? 0, 0)}</span>
      </div>
    </Card>
  );
}

export function ModelPanelPage() {
  const a = useAudit();
  const qc = useQueryClient();
  const [rerunError, setRerunError] = useState<string | null>(null);
  const q = useQuery({
    queryKey: ["panel", a.id],
    queryFn: () => api.panel(a.id),
    refetchInterval: (query) => (["queued", "running"].includes(query.state.data?.status ?? "") ? 2000 : false),
  });
  const p = q.data;
  // when the panel finishes, the audit's file list gains model_panel.csv / .json: refresh it once
  const prev = useRef<string | undefined>(undefined);
  useEffect(() => {
    if (prev.current && prev.current !== "ok" && p?.status === "ok") qc.invalidateQueries({ queryKey: ["audit", a.id] });
    prev.current = p?.status;
  }, [p?.status, a.id, qc]);

  const rerun = async (seeds: number) => {
    setRerunError(null);
    try {
      await api.startPanel(a.id, seeds);
      qc.invalidateQueries({ queryKey: ["panel", a.id] });
    } catch (e) {
      setRerunError(e instanceof Error ? e.message : String(e));
    }
  };

  const subtitle = "The same four models trained on your raw rows and on your repaired rows, scored on the same untouched held-out rows. Differences are read against a measured noise floor.";
  if (q.isLoading || !p) return <Page title="Model panel" subtitle={subtitle}><Card title="Loading"><Loader2 className="h-5 w-5 animate-spin text-accent" /></Card></Page>;

  if (p.status === "queued" || p.status === "running")
    return <Page title="Model panel" subtitle={subtitle}><Running p={p} /></Page>;

  if (p.status === "none" || p.status === "declined" || p.status === "error")
    return (
      <Page title="Model panel" subtitle={subtitle}>
        <EmptyState
          title={p.status === "none" ? "The model panel was not run for this audit" : p.status === "declined" ? "The model panel did not run" : "The model panel failed"}
          action={a.request.target && p.status !== "declined" ? <Button variant="primary" onClick={() => rerun(3)}><RotateCw className="h-4 w-4" /> Run the model panel</Button> : undefined}
        >
          {p.status === "none" ? (a.request.target ? "It trains four models on the raw and on the repaired data. It takes about a minute." : "It needs a target column, and this audit was run without one.") : p.reason}
          {rerunError && <span className="mt-2 block text-bad">{rerunError}</span>}
        </EmptyState>
      </Page>
    );

  const reg = p.task === "regression";
  const models = p.models ?? [];
  const arms = p.arms ?? [];
  const cells = p.results ?? [];
  const cell = (m: string, arm: string) => cells.find((c) => c.model === m && c.arm === arm);
  const repairArms = arms.filter((x) => x.key !== "before");
  const singles = repairArms.filter((x) => !["default", "aggressive"].includes(x.key));

  // Scorecard: one row per model
  const scoreCols: Col<(typeof models)[number]>[] = [
    { key: "m", header: "Model", cell: (m) => <span className="flex items-center gap-2 font-medium"><span className="h-2.5 w-2.5 rounded-full" style={{ background: MODEL_COLOR[m.key] }} />{m.label}<span className="hidden text-xs font-normal text-muted xl:inline">{m.role}</span></span> },
    { key: "b", header: "Before (raw)", align: "right", cell: (m) => { const c = cell(m.key, "before"); return c ? <span><b className="tabular-nums">{num(c.f1_mean, 4)}</b><span className="ml-1 text-xs text-muted">± {num(c.f1_std, 4)}</span></span> : "–"; } },
    ...(["default", "aggressive"] as const).filter((k) => arms.some((x) => x.key === k)).map((k) => ({
      key: k, header: k === "default" ? "Default file" : "Aggressive file", align: "right" as const,
      cell: (m: (typeof models)[number]) => { const c = cell(m.key, k); return c ? <Delta c={c} /> : "–"; },
    })),
    { key: "n", header: "Noise floor", align: "right", cell: (m) => <span className="tabular-nums text-muted">±{num(m.noise_floor, 4)}</span> },
  ];

  // Chart: delta of each repair arm, one bar per model; faded = within noise or no change
  const chartData = repairArms.map((x) => {
    const row: Record<string, number | string | null> = { arm: x.label.replace(" only", "").replace(" file", "") };
    models.forEach((m) => { const c = cell(m.key, x.key); row[m.key] = c ? c.delta_f1 : null; row[m.key + "_v"] = c ? c.verdict : null; });
    return row;
  });

  // A short factual summary per repair arm
  const summary = repairArms.map((x) => {
    const cs = models.map((m) => cell(m.key, x.key)).filter((c): c is PanelCell => !!c);
    const n = (v: string) => cs.filter((c) => c.verdict === v).length;
    return { arm: x, improved: n("improved"), worse: n("worse"), noise: n("within noise"), none: n("no change") + n("no effect"), total: cs.length };
  });

  const detailCols: Col<PanelCell>[] = [
    { key: "m", header: "Model", cell: (c) => models.find((m) => m.key === c.model)?.label ?? c.model },
    { key: "a", header: "Training data", cell: (c) => arms.find((x) => x.key === c.arm)?.label ?? c.arm },
    { key: "f", header: reg ? "R²" : "Macro-F1", align: "right", cell: (c) => <span className="tabular-nums">{num(c.f1_mean, 4)} <span className="text-xs text-muted">± {num(c.f1_std, 4)}</span></span> },
    { key: "d", header: "Change vs raw", align: "right", cell: (c) => <Delta c={c} /> },
    { key: "s", header: "Seeds agree", align: "right", cell: (c) => (c.verdict === "baseline" || c.verdict === "no change" || c.verdict === "no effect" ? "–" : pct(c.seeds_agree, 0)) },
    { key: "u", header: reg ? "RMSE (lower is better)" : "ROC-AUC", align: "right", cell: (c) => <span className="tabular-nums">{num(c.auc_mean, 4)}{c.arm !== "before" && c.delta_auc !== null && c.verdict !== "no change" && c.verdict !== "no effect" && <span className="ml-1 text-xs text-muted">{signed(c.delta_auc)}</span>}</span> },
    { key: "l", header: reg ? "R² without copied test rows" : "F1 without copied test rows", align: "right", cell: (c) => <span className="tabular-nums text-muted">{c.f1_noleak_mean === null ? "–" : num(c.f1_noleak_mean, 4)}</span> },
  ];

  return (
    <Page
      title="Model panel"
      subtitle={subtitle}
      actions={
        <>
          <ButtonLink href={api.downloadUrl(a.id, "model_panel.csv")} size="sm"><Download className="h-3.5 w-3.5" /> CSV</ButtonLink>
          <ButtonLink href={api.downloadUrl(a.id, "model_panel.json")} size="sm"><Download className="h-3.5 w-3.5" /> JSON</ButtonLink>
        </>
      }
    >
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Stat label="Models trained" value={models.length} hint={models.map((m) => m.label.split(" ")[0]).join(", ")} />
        <Stat label="Seeds" value={p.n_seeds} hint="each a different split and model seed" />
        <Stat label="Rows" value={int(p.rows_used)} hint={`${int(p.train_rows)} train · ${int(p.test_rows)} held-out`} />
        <Stat label="Held-out rows copied in training" value={pct((p.leakage?.test_rows_with_exact_copy_in_train_pct ?? 0) / 100, 1)} tone={(p.leakage?.test_rows_with_exact_copy_in_train_pct ?? 0) > 1 ? "warn" : "good"} hint="inflates the raw score" />
      </div>

      <Alert tone="warn" title="How to read this">{p.caveat}</Alert>

      <Card title="Scorecard" description={`${p.metric}. “Default file” is what the evidence-based policy applies without review (${(p.default_issues ?? []).length ? (p.default_issues ?? []).join(", ").replace(/_/g, " ") : "nothing: no repair is approved automatically"}); “Aggressive file” applies every proposed repair except feature-anomaly repairs. A change is called improved or worse only if it is larger than the model's noise floor and most seeds agree on its direction.`}>
        <DataTable columns={scoreCols} rows={models} rowKey={(m) => m.key} />
      </Card>

      <Card title="What each repair does to each model" description={`Change in ${reg ? "R²" : "macro-F1"} against training on the raw rows. Solid bars are beyond the noise floor; faded bars are within noise or unchanged. The same repair often helps one model family and hurts another.`}>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="arm" tick={axisTick} axisLine={axisLine} tickLine={false} />
              <YAxis tick={axisTick} axisLine={false} tickLine={false} tickFormatter={(v) => v.toFixed(2)} />
              <ReferenceLine y={0} stroke="var(--line)" />
              <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<ChartTip fmt={(v) => signed(v)} />} />
              {models.map((m) => (
                <Bar key={m.key} isAnimationActive={false} dataKey={m.key} name={m.label} fill={MODEL_COLOR[m.key]} radius={[3, 3, 0, 0]}>
                  {chartData.map((d, i) => <Cell key={i} fill={MODEL_COLOR[m.key]} fillOpacity={d[m.key + "_v"] === "improved" || d[m.key + "_v"] === "worse" ? 1 : 0.28} />)}
                </Bar>
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-3"><ChartLegend items={models.map((m) => ({ label: m.label, color: MODEL_COLOR[m.key] }))} /></div>
      </Card>

      <Card title="In plain words" description="Counts of models, out of the models trained.">
        <ul className="space-y-1.5 text-sm">
          {summary.map((s) => (
            <li key={s.arm.key} className="flex flex-wrap items-baseline gap-x-2">
              <b className="w-40 shrink-0">{s.arm.label}</b>
              <span>{s.improved} improved · {s.worse} worse · {s.noise} within noise · {s.none} no change or no effect</span>
              <span className="text-xs text-muted">{s.arm.text}{s.arm.rows_removed ? ` (${int(s.arm.rows_removed)} training rows removed)` : ""}{s.arm.cells_changed ? ` (${int(s.arm.cells_changed)} cells changed)` : ""}</span>
            </li>
          ))}
        </ul>
        {!singles.some((x) => x.rows_removed > 0 || x.cells_changed > 0) && <p className="mt-2 text-sm text-muted">No single repair type changed any training rows, so every arm equals the raw data. This file looks clean to the detectors.</p>}
      </Card>

      <Card title="Every arm, every model" description={`Mean ${reg ? "R²" : "macro-F1"} over seeds, with the spread. The last column scores only the held-out rows that have no copy in the training rows, so duplicates across the split cannot flatter the raw score.`}>
        <DataTable columns={detailCols} rows={cells} rowKey={(c) => `${c.model}-${c.arm}`} dense />
      </Card>

      {(p.interaction?.length ?? 0) > 0 && (
        <Card title="All repairs together versus the sum of the single repairs" description="Effects of separate repairs do not add up: they interact. Differences between these two numbers are expected.">
          <DataTable
            columns={[
              { key: "m", header: "Model", cell: (r) => models.find((m) => m.key === r.model)?.label ?? r.model },
              { key: "a", header: "Aggressive file (measured)", align: "right", cell: (r) => <span className="tabular-nums">{signed(r.aggressive)}</span> },
              { key: "s", header: "Sum of single repairs", align: "right", cell: (r) => <span className="tabular-nums">{signed(r.sum_of_singles)}</span> },
            ]}
            rows={p.interaction ?? []} rowKey={(r) => r.model} dense
          />
        </Card>
      )}

      {(p.notes?.length ?? 0) > 0 && <Alert tone="accent" title="Notes">{(p.notes ?? []).join(" · ")}</Alert>}

      <div className="flex flex-wrap items-center gap-3 text-xs text-muted">
        <span>Ran in {seconds(p.seconds ?? 0)} · random forest {p.settings?.rf_trees} trees · gradient boosting {p.settings?.gb_iterations} iterations · neural net {p.settings?.mlp_hidden} hidden units · noise floor from {p.settings?.null_draws_per_seed} random 90% retrains per seed.</span>
        {a.request.target && <Button size="sm" onClick={() => rerun(5)}><RotateCw className="h-3.5 w-3.5" /> Re-run with 5 seeds</Button>}
        {rerunError && <span className="text-bad">{rerunError}</span>}
      </div>
    </Page>
  );
}
