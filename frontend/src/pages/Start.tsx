import { useMutation, useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { ArrowRight, FileSpreadsheet, Loader2, Play, Sparkles, UploadCloud } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, type DatasetInfo } from "../api";
import { HealthReport } from "../components/HealthReport";
import { Alert, Badge, Button, Card, Page, ProgressBar, Segmented, Switch } from "../components/ui";
import { cellText, int, seconds } from "../lib";

const NO_TARGET = "__none__";

function Dropzone({ onFile, busy }: { onFile: (f: File) => void; busy: boolean }) {
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files?.[0]; if (f) onFile(f); }}
      className={clsx("flex flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-12 text-center transition", over ? "border-accent bg-accent-soft" : "border-line bg-surface")}
    >
      {busy ? <Loader2 className="h-8 w-8 animate-spin text-accent" /> : <UploadCloud className="h-8 w-8 text-accent" />}
      <p className="mt-3 text-base font-semibold">{busy ? "Reading your file…" : "Drop a CSV file here"}</p>
      <p className="mt-1 text-sm text-muted">Tabular data with a classification target. Encoding, delimiter and decimal comma are detected for you.</p>
      <input ref={input} type="file" accept=".csv,text/csv" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); e.target.value = ""; }} />
      <div className="mt-4 flex flex-wrap justify-center gap-2">
        <Button variant="primary" onClick={() => input.current?.click()} disabled={busy}>
          <FileSpreadsheet className="h-4 w-4" /> Choose CSV file
        </Button>
      </div>
    </div>
  );
}

function PreviewTable({ ds }: { ds: DatasetInfo }) {
  const cols = ds.columns.map((c) => c.name);
  return (
    <div className="scroll-thin max-h-[22rem] overflow-auto rounded-lg border border-line">
      <table className="w-full border-collapse text-xs">
        <thead className="sticky top-0 bg-surface-2">
          <tr>
            {cols.map((c) => (
              <th key={c} className="whitespace-nowrap px-3 py-2 text-left font-medium text-muted">{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {ds.preview.map((r, i) => (
            <tr key={i} className="border-t border-line/70">
              {cols.map((c) => (
                <td key={c} className="whitespace-nowrap px-3 py-1.5 tabular-nums">{cellText(r[c])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const STEPS = [
  { key: "audit", label: "Detect problems", text: "Six detectors scan every row; model-based ones run on a sample above the row limit." },
  { key: "calibration", label: "Calibrate on your data", text: "Inject known errors into a clean-ish reference and measure damage and repair safety." },
  { key: "finalising", label: "Apply the policy", text: "Only repairs the evidence shows to be safe are applied; the rest are flagged." },
];

function RunProgress({ jobId, calibrate, onBack }: { jobId: string; calibrate: boolean; onBack: () => void }) {
  const nav = useNavigate();
  const job = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.job(jobId),
    refetchInterval: (q) => (q.state.data?.status === "done" || q.state.data?.status === "error" ? false : 600),
    staleTime: 0,
  });
  const j = job.data;
  useEffect(() => {
    if (j?.status === "done" && j.audit_id) nav(`/audit/${j.audit_id}`);
  }, [j?.status, j?.audit_id, nav]);

  const steps = STEPS.filter((s) => calibrate || s.key !== "calibration");
  const activeIdx = Math.max(0, steps.findIndex((s) => s.key === j?.stage));
  if (j?.status === "error")
    return (
      <Card title="The audit could not run">
        <Alert tone="bad" title="Stopped">{j.error}</Alert>
        <div className="mt-4"><Button onClick={onBack}>Back to configuration</Button></div>
      </Card>
    );
  return (
    <Card title="Running the audit" description={j ? `${seconds(j.elapsed)} elapsed` : undefined}>
      <ProgressBar value={j?.progress ?? 0} />
      <p className="mt-2 flex items-center gap-2 text-sm"><Loader2 className="h-4 w-4 animate-spin text-accent" /> {j?.message ?? "Starting"}</p>
      <ol className="mt-5 space-y-3">
        {steps.map((s, i) => (
          <li key={s.key} className={clsx("flex gap-3", i > activeIdx && "opacity-50")}>
            <span className={clsx("mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold", i < activeIdx ? "bg-good text-white" : i === activeIdx ? "bg-accent text-accent-fg" : "bg-surface-2 text-muted")}>{i + 1}</span>
            <div><div className="text-sm font-medium">{s.label}</div><div className="text-xs text-muted">{s.text}</div></div>
          </li>
        ))}
      </ol>
    </Card>
  );
}

export function Start() {
  const config = useQuery({ queryKey: ["config"], queryFn: api.config, staleTime: Infinity });
  const [ds, setDs] = useState<DatasetInfo | null>(null);
  const [target, setTarget] = useState<string>(NO_TARGET);
  const [feat, setFeat] = useState(false);
  const [joint, setJoint] = useState(false);
  const [calibrate, setCalibrate] = useState(false);
  const [seeds, setSeeds] = useState(3);
  const [maxRows, setMaxRows] = useState(30000);
  const [jobId, setJobId] = useState<string | null>(null);

  const load = useCallback((d: DatasetInfo) => {
    setDs(d); setTarget(d.suggested_target ?? NO_TARGET); setCalibrate(false); setFeat(false); setJoint(false); setJobId(null);
  }, []);
  const upload = useMutation({ mutationFn: api.uploadDataset, onSuccess: load });
  const demo = useMutation({ mutationFn: api.demoDataset, onSuccess: load });
  const start = useMutation({ mutationFn: api.startAudit, onSuccess: (r) => setJobId(r.job_id) });

  const maxClasses = config.data?.max_classes ?? 20;
  const col = ds?.columns.find((c) => c.name === target);
  const targetProblem = !col ? null : col.n_unique < 2 ? `'${col.name}' has fewer than 2 distinct values: there is nothing to classify.`
    : col.n_unique > maxClasses ? `'${col.name}' has ${int(col.n_unique)} distinct values. TaskClean supports classification only (up to ${maxClasses} classes); this looks like a regression target or an ID column.` : null;
  const hasTarget = target !== NO_TARGET;
  // the stored baseline snapshot (never recomputed) plus, once a target is chosen, that column's profile
  const health = useQuery({
    queryKey: ["health", ds?.id, hasTarget ? target : null],
    queryFn: () => api.datasetHealth(ds!.id, hasTarget ? target : null),
    enabled: !!ds, placeholderData: (prev) => prev, staleTime: Infinity,
  });
  const error = (upload.error ?? demo.error ?? start.error) as Error | null;

  const run = () => ds && start.mutate({
    dataset_id: ds.id, target: hasTarget ? target : null, include_feature_anomalies: feat, include_joint_outliers: joint,
    calibrate: calibrate && hasTarget, seeds, max_model_rows: maxRows,
  });

  if (!ds)
    return (
      <Page title="Audit a dataset" subtitle="Find out which data problems actually hurt your model, and repair only what the evidence says is safe.">
        {error && <Alert tone="bad" title="Could not load that file">{error.message}</Alert>}
        <Dropzone onFile={(f) => upload.mutate(f)} busy={upload.isPending || demo.isPending} />
        <div className="flex items-center gap-3 text-sm text-muted">
          <div className="h-px flex-1 bg-line" /> or <div className="h-px flex-1 bg-line" />
        </div>
        <Card>
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2 font-semibold"><Sparkles className="h-4 w-4 text-accent" /> Try the demo dataset</div>
              <p className="mt-1 max-w-xl text-sm text-muted">6,300 rows of UCI Adult with all five error types injected at 5%. Runs in seconds and shows every part of the dashboard.</p>
            </div>
            <Button onClick={() => demo.mutate()} disabled={demo.isPending || upload.isPending}>
              {demo.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />} Use demo dataset
            </Button>
          </div>
        </Card>
        <div className="grid gap-4 sm:grid-cols-3">
          {[
            ["1. Measure", "Detectors flag missing values, duplicates, outliers, label errors and feature anomalies. A flagged rate is not a confirmed error rate."],
            ["2. Estimate impact", "Each problem is mapped onto a measured damage curve, so you see what it costs a model, not just how common it is."],
            ["3. Repair safely", "Only repairs with zero observed harm and high precision are applied automatically. Everything else is logged for review."],
          ].map(([t, d]) => (
            <Card key={t}><div className="font-semibold">{t}</div><p className="mt-1 text-sm text-muted">{d}</p></Card>
          ))}
        </div>
      </Page>
    );

  return (
    <Page
      title={ds.name}
      subtitle={<>{int(ds.rows)} rows · {ds.n_columns} columns · {ds.input_format.encoding}, “{ds.input_format.delimiter}” delimiter</>}
      actions={<Button variant="ghost" onClick={() => { setDs(null); setJobId(null); }} disabled={start.isPending}>Choose another file</Button>}
    >
      {error && <Alert tone="bad" title="Something went wrong">{error.message}</Alert>}
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_22rem]">
        <div className="min-w-0 space-y-6">
          <Card title="Preview" description="First 10 rows exactly as read from the file.">
            <PreviewTable ds={ds} />
          </Card>
          <Card title="Baseline health report" description="A snapshot of your raw file, captured now, before any cleaning. It is saved with the dataset and never changes.">
            {health.isLoading && <div className="flex items-center gap-2 text-sm text-muted"><Loader2 className="h-4 w-4 animate-spin" /> Profiling the file…</div>}
            {health.isError && <Alert tone="warn" title="The baseline report could not be built">{(health.error as Error).message}. You can still run the audit.</Alert>}
            {health.data && <HealthReport snapshot={health.data.snapshot} target={health.data.target} />}
          </Card>
          <Card title="Columns" description={`${ds.columns.length} columns. Click a candidate to use it as the target.`}>
            <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
              {ds.columns.map((c) => (
                <button key={c.name} type="button" onClick={() => c.target_candidate && setTarget(c.name)} disabled={!c.target_candidate}
                  className={clsx("rounded-lg border px-3 py-2 text-left transition", target === c.name ? "border-accent bg-accent-soft" : "border-line", c.target_candidate ? "hover:border-accent" : "cursor-default")}>
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate text-sm font-medium">{c.name}</span>
                    <Badge tone={target === c.name ? "accent" : "neutral"}>{target === c.name ? "target" : c.kind}</Badge>
                  </div>
                  <div className="mt-0.5 text-xs text-muted">{int(c.n_unique)} distinct · {c.n_missing ? `${int(c.n_missing)} missing` : "none missing"}</div>
                </button>
              ))}
            </div>
          </Card>
        </div>

        <div className="lg:sticky lg:top-6 lg:self-start">
          {jobId ? (
            <RunProgress jobId={jobId} calibrate={calibrate && hasTarget} onBack={() => setJobId(null)} />
          ) : (
            <Card title="Configure the audit">
              <div className="space-y-5">
                <label className="block">
                  <span className="text-xs font-medium uppercase tracking-wide text-muted">Target column</span>
                  <select value={target} onChange={(e) => setTarget(e.target.value)} className="mt-1.5 w-full rounded-lg border border-line bg-surface px-3 py-2 text-sm">
                    <option value={NO_TARGET}>(no target: unlabeled dataset)</option>
                    {ds.columns.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
                  </select>
                </label>
                {targetProblem && <Alert tone="bad">{targetProblem}</Alert>}
                {!hasTarget && <Alert tone="accent">Without a target, missing values, duplicates, outliers and inconsistent values are checked. The label-error check needs a target.</Alert>}

                <div>
                  <span className="text-xs font-medium uppercase tracking-wide text-muted">ML task</span>
                  <div className="mt-1.5 flex items-center gap-2 text-sm"><Badge tone="accent">Classification</Badge><span className="text-xs text-muted">Regression is not supported.</span></div>
                </div>

                <div className="flex items-start justify-between gap-4">
                  <div>
                    <div className="text-sm font-medium">Calibrate on this dataset</div>
                    <p className="text-xs text-muted">Measure damage and repair safety on your own data instead of the UCI Adult benchmark. Adds about 20 seconds to a few minutes.</p>
                  </div>
                  <Switch label="Calibrate on this dataset" checked={calibrate && hasTarget} onChange={setCalibrate} disabled={!hasTarget} />
                </div>
                {calibrate && hasTarget && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Calibration seeds</span>
                    <Segmented label="Calibration seeds" value={seeds} onChange={setSeeds} options={[{ value: 3, label: "3 (quick)" }, { value: 5, label: "5 (thorough)" }]} />
                  </div>
                )}

                <div className="flex items-start justify-between gap-4">
                  <div>
                    <div className="text-sm font-medium">Feature-anomaly detector</div>
                    <p className="text-xs text-muted">One model per column. Slow, about 28% precision on the benchmark, and its automatic repair was harmful, so findings stay flag-only.</p>
                  </div>
                  <Switch label="Feature-anomaly detector" checked={feat} onChange={setFeat} />
                </div>

                <div className="flex items-start justify-between gap-4">
                  <div>
                    <div className="text-sm font-medium">Unusual combinations of numeric values <Badge tone="warn">experimental</Badge></div>
                    <p className="text-xs text-muted">Flags rows whose numbers are individually normal but do not fit together. Flag-only: nothing is ever changed. Off by default because validation failed on real data: it flagged 7% of clean Diabetes rows and 30% of clean Breast Cancer rows, and found only 10–14% of hidden swaps. Treat flags as a list to look at, not a rate to trust.</p>
                  </div>
                  <Switch label="Unusual combinations detector" checked={joint} onChange={setJoint} />
                </div>

                <details className="rounded-lg border border-line px-3 py-2 text-sm">
                  <summary className="cursor-pointer font-medium">Advanced</summary>
                  <label className="mt-3 block">
                    <span className="text-xs text-muted">Row limit for the slow model-based detectors (stratified sample above this)</span>
                    <input type="number" min={2000} max={500000} step={5000} value={maxRows} onChange={(e) => setMaxRows(Number(e.target.value))} className="mt-1 w-full rounded-lg border border-line bg-surface px-3 py-2 text-sm" />
                  </label>
                </details>

                {ds.rows > 50000 && <Alert tone="warn">Large dataset: the model-based detectors may take several minutes.</Alert>}
                <Button variant="primary" className="w-full" onClick={run} disabled={!!targetProblem || start.isPending}>
                  {start.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />} Run audit <ArrowRight className="h-4 w-4" />
                </Button>
              </div>
            </Card>
          )}
        </div>
      </div>
    </Page>
  );
}
