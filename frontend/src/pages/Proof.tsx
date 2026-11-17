import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Loader2, RotateCw } from "lucide-react";
import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type Proof, type ProofCell, type Task } from "../api";
import { axisLine, axisTick, ChartLegend, ChartTip, gridProps } from "../components/charts";
import { Alert, Badge, Button, Card, DataTable, EmptyState, Page, ProgressBar, Stat, type Col } from "../components/ui";
import { num, pct, seconds, signed } from "../lib";

const MODEL_COLOR: Record<string, string> = {
  logistic_regression: "var(--c-label)",
  ridge: "var(--c-label)",
  random_forest: "var(--c-outliers)",
  gradient_boosting: "var(--c-missing)",
  neural_net: "var(--c-duplicates)",
};
const verdictTone = (v: string) => (v === "improved" ? "good" : v === "worse" ? "bad" : "neutral");
const CRITERIA: Record<string, string> = {
  C0_evaluable: "C0 The injected errors damage at least 3 of 4 models beyond noise (otherwise recovery cannot be read)",
  C1_complete: "C1 Every arm for every model is reported",
  C2_tuned_positive: "C2 Tuned cleaning improves at least 2 models and worsens none (leak-free validation)",
  C2_tuned_v0_positive: "C2 as originally specified (raw validation)",
  C3_default_not_worse: "C3 The default file is not worse than dirty for any model",
};

function Delta({ c }: { c: ProofCell }) {
  if (c.verdict === "ceiling") return <span className="text-muted">ceiling</span>;
  if (c.verdict === "baseline") return <span className="text-muted">baseline</span>;
  if (c.verdict === "no change" || c.verdict === "no effect") return <Badge>{c.verdict}</Badge>;
  return (
    <span className="inline-flex flex-wrap items-center justify-end gap-1.5">
      <b className="tabular-nums">{signed(c.delta_f1)}</b>
      <Badge tone={verdictTone(c.verdict)}>{c.verdict}</Badge>
      {c.share_recovered !== null && <span className="text-xs text-muted tabular-nums">{pct(c.share_recovered, 0)} of the damage</span>}
    </span>
  );
}

export function ProofPage() {
  const qc = useQueryClient();
  const [err, setErr] = useState<string | null>(null);
  const [task, setTask] = useState<Task>("classification");
  const q = useQuery({
    queryKey: ["proof", task], queryFn: () => api.proof(task),
    refetchInterval: (query) => (["queued", "running"].includes(query.state.data?.status ?? "") ? 2500 : false),
  });
  const p: Proof | undefined = q.data;
  const rerun = async (seeds: number) => {
    setErr(null);
    try { await api.startProof(seeds, task); qc.invalidateQueries({ queryKey: ["proof", task] }); } catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
  };
  const subtitle = "Known errors injected into clean data, repaired by the real TaskClean path, and every model scored on a clean held-out set. This is the evidence that repaired data trains better (or not).";
  const switcher = (
    <div className="inline-flex rounded-lg border border-line p-0.5 text-sm" role="radiogroup" aria-label="ML task">
      {(["classification", "regression"] as Task[]).map((t) => (
        <button key={t} type="button" role="radio" aria-checked={task === t} onClick={() => setTask(t)}
          className={"rounded-md px-3 py-1.5 capitalize transition " + (task === t ? "bg-accent text-white" : "text-muted hover:text-text")}>{t}</button>
      ))}
    </div>
  );

  if (q.isLoading || !p) return <Page title="Controlled proof" subtitle={subtitle} actions={switcher}><Card title="Loading"><Loader2 className="h-5 w-5 animate-spin text-accent" /></Card></Page>;
  if (p.status === "queued" || p.status === "running")
    return (
      <Page title="Controlled proof" subtitle={subtitle} actions={switcher}>
        <Card title="Running the controlled proof" description="Clean rows → inject errors → repair → train four models → score on clean held-out rows. A few minutes.">
          <div className="flex items-center gap-3"><Loader2 className="h-5 w-5 animate-spin text-accent" /><div className="min-w-0 flex-1"><ProgressBar value={p.progress ?? 0} /><p className="mt-2 truncate text-sm text-muted">{p.message}</p></div><span className="text-sm tabular-nums text-muted">{pct(p.progress ?? 0, 0)}</span></div>
        </Card>
      </Page>
    );
  if (p.status !== "ok")
    return (
      <Page title="Controlled proof" subtitle={subtitle} actions={switcher}>
        <EmptyState title={p.status === "error" ? "The controlled proof failed" : "No controlled proof has been run yet"} action={<Button variant="primary" onClick={() => rerun(3)}><RotateCw className="h-4 w-4" /> Run it (3 seeds)</Button>}>
          {p.reason ?? "Run it, or run src/controlled_proof.py to save the registered 5-seed result."}{err && <span className="mt-2 block text-bad">{err}</span>}
        </EmptyState>
      </Page>
    );

  const models = p.models ?? [];
  const cells = p.results ?? [];
  const cell = (m: string, a: string) => cells.find((c) => c.model === m && c.arm === a);
  const crit = p.criteria ?? {};
  const positive = !!crit.C2_tuned_positive?.pass;
  const tunedV = Object.values((crit.C2_tuned_positive?.verdicts ?? {}) as Record<string, string>);
  const armLabel = (k: string) => p.arms?.find((a) => a.key === k)?.label ?? k;
  const shown = ["aggressive", "tuned", "tuned_v0"] as const;

  const scoreCols: Col<(typeof models)[number]>[] = [
    { key: "m", header: "Model", cell: (m) => <span className="flex items-center gap-2 font-medium"><span className="h-2.5 w-2.5 rounded-full" style={{ background: MODEL_COLOR[m.model] }} />{m.label}</span> },
    { key: "c", header: "Clean ceiling", align: "right", cell: (m) => <span className="tabular-nums">{num(m.clean_f1, 4)}</span> },
    { key: "d", header: "Dirty", align: "right", cell: (m) => <span><b className="tabular-nums">{num(m.dirty_f1, 4)}</b>{!m.damage_beyond_noise && <span className="ml-1.5 text-xs text-muted">barely harmed</span>}</span> },
    ...shown.map((k) => ({ key: k, header: armLabel(k), align: "right" as const, cell: (m: (typeof models)[number]) => { const c = cell(m.model, k); return c ? <Delta c={c} /> : "–"; } })),
    { key: "n", header: "Noise floor", align: "right", cell: (m) => <span className="tabular-nums text-muted">±{num(m.noise_floor, 4)}</span> },
  ];
  const chartData = models.map((m) => ({ model: m.label.split(" ")[0] + (m.label.includes("(") ? "" : ""), key: m.model, clean: m.clean_f1, dirty: m.dirty_f1, aggressive: cell(m.model, "aggressive")?.f1_mean ?? null, tuned: cell(m.model, "tuned")?.f1_mean ?? null }));
  const allCols: Col<ProofCell>[] = [
    { key: "m", header: "Model", cell: (c) => models.find((m) => m.model === c.model)?.label ?? c.model },
    { key: "a", header: "Training data", cell: (c) => armLabel(c.arm) },
    { key: "f", header: task === "regression" ? "R² on clean rows" : "Macro-F1 on clean rows", align: "right", cell: (c) => <span className="tabular-nums">{num(c.f1_mean, 4)} <span className="text-xs text-muted">± {num(c.f1_std, 4)}</span></span> },
    { key: "d", header: "Change vs dirty", align: "right", cell: (c) => <Delta c={c} /> },
    { key: "s", header: "Seeds agree", align: "right", cell: (c) => (c.seeds_agree === null || c.verdict === "no change" || c.verdict === "no effect" ? "–" : pct(c.seeds_agree, 0)) },
  ];
  const critRows = Object.entries(crit).map(([k, v]) => ({ key: k, text: CRITERIA[k] ?? k, pass: v.pass }));

  return (
    <Page title="Controlled proof" subtitle={subtitle} actions={<>{switcher}<Badge tone="neutral">{p.source ?? ""}</Badge></>}>
      <Alert tone={positive ? "good" : "warn"} title={positive ? "Positive result: tuned cleaning improved the models on clean held-out data" : "The pre-registered positive criterion was not met"}>
        {positive
          ? `Tuned cleaning improved at least two model families beyond the noise floor and worsened none. The criteria were fixed before the run (docs/PHASE${task === "regression" ? 21 : 20}_SPEC.md).`
          : `Tuned cleaning improved ${tunedV.filter((v) => v === "improved").length} of ${tunedV.length} models beyond noise and worsened ${tunedV.filter((v) => v === "worse").length}. The criterion (improve at least 2, worsen none) was fixed before the run and is reported as measured.`}
        {" "}{p.n_seeds} seeds · {p.rows?.toLocaleString()} {task === "regression" ? "California Housing" : "Adult"} rows · errors injected at {pct(p.rate_per_error_type ?? 0, 0)} each (missing values, outliers, {task === "regression" ? "wrong target values" : "label errors"}, feature corruption, duplicates) into the training rows only · scored on clean held-out rows.
      </Alert>

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {models.map((m) => {
          const t = cell(m.model, "tuned");
          return <Stat key={m.model} label={m.label} value={t?.delta_f1 === null || t?.delta_f1 === undefined ? "–" : signed(t.delta_f1)} tone={t?.verdict === "improved" ? "good" : t?.verdict === "worse" ? "bad" : "neutral"}
            hint={`tuned cleaning vs dirty · ${t?.verdict ?? ""}${t?.share_recovered != null ? ` · ${pct(t.share_recovered, 0)} of the damage` : ""}`} />;
        })}
      </div>

      <Card title="Scorecard" description={`${p.metric}. “Clean ceiling” is what perfect cleaning of the training rows could reach; “Dirty” is no cleaning. The other columns show the change against Dirty. Improved or worse needs a change larger than the model's noise floor with most seeds agreeing on the direction.`}>
        <DataTable columns={scoreCols} rows={models} rowKey={(m) => m.model} />
      </Card>

      <Card title={`${task === "regression" ? "R²" : "Macro-F1"} on clean held-out rows`} description="Clean ceiling, dirty, aggressive file and tuned cleaning, per model.">
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="model" tick={axisTick} axisLine={axisLine} tickLine={false} />
              <YAxis domain={["dataMin - 0.03", "dataMax + 0.01"]} tick={axisTick} axisLine={false} tickLine={false} tickFormatter={(v) => v.toFixed(2)} />
              <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<ChartTip fmt={(v) => v.toFixed(4)} />} />
              <Bar isAnimationActive={false} dataKey="clean" name="Clean ceiling" fill="var(--muted)" fillOpacity={0.45} radius={[3, 3, 0, 0]} />
              <Bar isAnimationActive={false} dataKey="dirty" name="Dirty" fill="var(--bad)" fillOpacity={0.8} radius={[3, 3, 0, 0]} />
              <Bar isAnimationActive={false} dataKey="aggressive" name="Aggressive file" fill="var(--c-outliers)" radius={[3, 3, 0, 0]} />
              <Bar isAnimationActive={false} dataKey="tuned" name="Tuned cleaning" fill="var(--good)" radius={[3, 3, 0, 0]}>{chartData.map((d) => <Cell key={d.key} fill="var(--good)" />)}</Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-3"><ChartLegend items={[{ label: "Clean ceiling", color: "var(--muted)" }, { label: "Dirty", color: "var(--bad)" }, { label: "Aggressive file", color: "var(--c-outliers)" }, { label: "Tuned cleaning", color: "var(--good)" }]} /></div>
      </Card>

      <Card title="Pre-registered criteria" description={`Fixed before the run in docs/PHASE${task === "regression" ? 21 : 20}_SPEC.md (with its visible amendments). Pass or fail, reported as measured.`}>
        <DataTable columns={[
          { key: "t", header: "Criterion", cell: (r: { text: string }) => <span className="text-sm">{r.text}</span> },
          { key: "p", header: "Result", align: "right", cell: (r: { pass: boolean }) => <Badge tone={r.pass ? "good" : "bad"}>{r.pass ? "pass" : "fail"}</Badge> },
        ]} rows={critRows} rowKey={(r) => r.key} dense />
      </Card>

      <Card title="Every arm, every model" description="Nothing is left out. “No change” means the arm produced the same training rows as the dirty data (nothing was repaired); “no effect” means cells were repaired but the trained model came out identical.">
        <DataTable columns={allCols} rows={cells} rowKey={(c) => `${c.model}-${c.arm}`} dense />
      </Card>

      {Object.keys(p.tuned_choices ?? {}).length > 0 && (
        <Card title="What tuned cleaning chose" description={`For each model and seed it picked the repair that did best on a validation slice of the dirty rows, but only if it beat raw by more than ${p.select_margin}. Counts over seeds.`}>
          <ul className="space-y-1 text-sm">
            {Object.entries(p.tuned_choices ?? {}).filter(([k]) => k.startsWith("tuned/")).map(([k, v]) => (
              <li key={k}><b>{models.find((m) => m.model === k.split("/")[1])?.label}</b>: {Object.entries(v).map(([a, n]) => `${armLabel(a)} × ${n}`).join(", ")}</li>
            ))}
          </ul>
        </Card>
      )}

      <Alert tone="accent" title="What this does and does not show">{p.caveat}</Alert>
      <div className="flex flex-wrap items-center gap-3 text-xs text-muted">
        <span>Ran in {seconds(p.seconds ?? 0)}. Saved results: results/phase20_proof.json.</span>
        <Button size="sm" onClick={() => rerun(3)}><RotateCw className="h-3.5 w-3.5" /> Re-run (3 seeds)</Button>
        {err && <span className="text-bad">{err}</span>}
      </div>
    </Page>
  );
}
