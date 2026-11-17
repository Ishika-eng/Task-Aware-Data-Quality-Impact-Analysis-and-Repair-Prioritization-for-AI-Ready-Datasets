import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import { api } from "../api";
import { useAudit } from "../audit-context";
import { Badge, Card, Dot, Page, RateBar, Stat, Status, Disclosure } from "../components/ui";
import { ISSUE_COLOR, ISSUE_NAME, ISSUE_ORDER, int, num, pct, seconds, signed, tone } from "../lib";

export function Overview() {
  const a = useAudit();
  const ov = a.overview;
  const sum = a.summary.evidence_based;
  const by = <T extends { issue_key: string }>(rows: T[]) => Object.fromEntries(rows.map((r) => [r.issue_key, r])) as Record<string, T>;
  const quality = by(a.quality), policy = by(a.policy), ready = by(a.readiness), impact = by(a.impact);
  const applied = a.selection.applied.map((k) => ISSUE_NAME[k] ?? k);
  const cal = a.calibration;
  const panelQ = useQuery({
    queryKey: ["panel", a.id], queryFn: () => api.panel(a.id), enabled: !!ov.target,
    refetchInterval: (query) => (["queued", "running"].includes(query.state.data?.status ?? "") ? 3000 : false),
  });
  const panel = panelQ.data;
  const agg = (panel?.results ?? []).filter((c) => c.arm === "aggressive");
  const labelOf = (k: string) => panel?.models?.find((m) => m.key === k)?.label ?? k;

  return (
    <Page
      title={a.dataset_name}
      subtitle={<>Audit of {int(ov.rows_analysed)} rows · target <b>{ov.target ?? "none"}</b> · {cal.status === "ok" ? "evidence measured on this dataset" : ov.task === "regression" ? "no regression evidence yet: calibrate to measure impact" : "evidence from the UCI Adult benchmark"} · {ov.task === "regression" ? "regression" : "classification"}</>}
      actions={<Badge tone={cal.status === "ok" ? "good" : "neutral"}>{cal.status === "ok" ? "Calibrated" : "Not calibrated"}</Badge>}
    >
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-5">
        <Stat label="Rows analysed" value={int(ov.rows_analysed)} hint={ov.rows_uploaded !== ov.rows_analysed ? `${int(ov.rows_uploaded)} uploaded` : undefined} />
        <Stat label="Columns" value={ov.columns} />
        <Stat label="Target" value={ov.target ?? "none"} />
        <Stat label={ov.task === "regression" ? "Task" : "Classes"} value={ov.task === "regression" ? "Regression" : ov.n_classes ?? "–"} />
        <Stat label="Audit time" value={seconds(ov.audit_seconds)} />
      </div>

      <Card title="What TaskClean did" description={`Repairs applied: ${a.selection.label}.`}>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <Stat label="Auto-repaired types" value={a.selection.applied.length} tone={a.selection.applied.length ? "good" : "neutral"} hint={applied.join(", ") || "none"} />
          <Stat label="Rows dropped" value={int(sum.rows_dropped)} hint="exact duplicate copies" />
          <Stat label="Cells modified" value={int(sum.cells_modified)} hint="in the evidence-based file" />
          <Stat label="Flagged for review" value={int(sum.flagged_for_review)} tone="warn" hint="proposals logged, not applied" />
        </div>
        <p className="mt-4 text-sm text-muted">
          {a.selection.applied.length
            ? <>Only <b className="text-fg">{applied.join(" and ")}</b> cleared the safety bar (no observed harm, repair precision of at least 95%). Everything else was left untouched and listed for human review.</>
            : <>No repair type cleared the safety bar, so nothing was changed automatically. Every proposal is in the repair log for review.</>}
        </p>
        <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1">
          <Link to={`/audit/${a.id}/review`} className="inline-flex items-center gap-1 text-sm font-medium text-accent hover:underline">Open the review queue ({int(a.review.total)} flagged) <ArrowRight className="h-3.5 w-3.5" /></Link>
          <Link to={`/audit/${a.id}/repair`} className="inline-flex items-center gap-1 text-sm font-medium text-accent hover:underline">Review the repair plan <ArrowRight className="h-3.5 w-3.5" /></Link>
        </div>
      </Card>

      {ov.target && panel && panel.status !== "none" && (
        <Card title="Model panel: before and after cleaning" description="Four model families trained on the raw rows and on the repaired rows, scored on the same untouched held-out rows.">
          {(panel.status === "queued" || panel.status === "running") && <p className="text-sm text-muted">Training in the background… {pct(panel.progress ?? 0, 0)}. Results appear here when it finishes.</p>}
          {panel.status === "ok" && (
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              {agg.map((c) => (
                <Stat key={c.model} label={labelOf(c.model)} value={signed(c.delta_f1)}
                  tone={c.verdict === "improved" ? "good" : c.verdict === "worse" ? "bad" : "neutral"}
                  hint={`${c.verdict} (noise ±${c.noise_floor.toFixed(3)}) · all repairs vs raw`} />
              ))}
            </div>
          )}
          {(panel.status === "declined" || panel.status === "error") && <p className="text-sm text-muted">{panel.reason}</p>}
          <div className="mt-3"><Link to={`/audit/${a.id}/panel`} className="inline-flex items-center gap-1 text-sm font-medium text-accent hover:underline">Open the model panel <ArrowRight className="h-3.5 w-3.5" /></Link></div>
        </Card>
      )}

      <div>
        <h2 className="mb-3 text-sm font-semibold">Issues at a glance</h2>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {[...ISSUE_ORDER.slice().reverse(), "inconsistency", "joint_outliers"].map((k) => {
            const q = quality[k], p = policy[k], r = ready[k], i = impact[k];
            if (!q) return null;
            if (k === "joint_outliers" && q.detected_rate === null) return null;     // opt-in: show the card only when it ran
            const ran = q.detected_rate !== null;
            return (
              <Card key={k}>
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-center gap-2"><Dot color={ISSUE_COLOR[k]} /><span className="font-semibold">{ISSUE_NAME[k]}</span></div>
                  {p ? <Status value={p.default_action} /> : <Badge>Report only</Badge>}
                </div>
                <div className="mt-3 flex items-baseline gap-2">
                  <span className="text-3xl font-semibold tabular-nums">{ran ? pct(q.detected_rate) : "–"}</span>
                  <span className="text-xs text-muted">{ran ? `flagged · ${int(q.detected_count)} ${q.count_unit}` : "not run"}</span>
                </div>
                <div className="mt-2.5"><RateBar value={q.detected_rate} floor={r?.baseline_false_positive_floor} color={ISSUE_COLOR[k]} /></div>
                <div className="mt-1 text-[11px] text-muted">bar: observed rate · tick: rate flagged on clean data</div>
                <dl className="mt-3 grid grid-cols-2 gap-x-3 gap-y-1.5 text-xs">
                  <dt className="text-muted">Detector</dt><dd className="text-right">{q.detector_type}</dd>
                  <dt className="text-muted">Precision / recall</dt><dd className="text-right tabular-nums">{pct(q.benchmark_precision, 0)} / {pct(q.benchmark_recall, 0)}</dd>
                  <dt className="text-muted">Impact estimate</dt><dd className="text-right tabular-nums">{i && i.benchmark_impact_estimate !== null ? `${signed(i.benchmark_impact_estimate)} ${i.impact_unit.replace(" (positive class)", "")}` : "–"}</dd>
                  <dt className="text-muted">Evidence</dt><dd className="text-right"><Status value={q.evidence_status} /></dd>
                  <dt className="text-muted">Status</dt><dd className="text-right">{r ? <Badge tone={tone(r.status)}>{r.status}</Badge> : "–"}</dd>
                </dl>
              </Card>
            );
          })}
        </div>
      </div>

      {(ov.notes.length > 0 || Object.keys(ov.columns_not_analysed).length > 0 || Object.keys(ov.columns_skipped_by_model_based_detectors).length > 0) && (
        <Card title="Data notes">
          <ul className="space-y-1.5 text-sm">
            {ov.notes.map((n) => <li key={n} className="text-muted">• {n}</li>)}
            {Object.entries(ov.columns_not_analysed).map(([c, why]) => <li key={c} className="text-muted">• Column <b className="text-fg">{c}</b> not analysed: {why}.</li>)}
            {Object.entries(ov.columns_skipped_by_model_based_detectors).map(([c, why]) => <li key={c} className="text-muted">• Column <b className="text-fg">{c}</b> skipped by the model-based detectors: {why}.</li>)}
          </ul>
        </Card>
      )}

      <Disclosure title="How to read this dashboard" defaultOpen>
        <ul className="list-disc space-y-1.5 pl-5">
          <li><b className="text-fg">A flagged rate is not a confirmed error rate.</b> Statistical detectors have known false-positive and false-negative behaviour, measured against injected errors.</li>
          <li><b className="text-fg">Detected does not mean safe to repair.</b> Only repair types that never showed harm and had high repair precision are applied automatically.</li>
          <li><b className="text-fg">Impact numbers are estimates</b> from a damage curve measured with a Random Forest, not predictions for your own model.</li>
          <li><b className="text-fg">There is no single readiness score.</b> The evidence did not justify collapsing the dimensions into one number.</li>
        </ul>
        <p className="mt-3 text-xs">Benchmark model: {ov.model_used_for_benchmark_validation}. Minimum-effect floor and other thresholds are listed on the Calibration and Readiness pages. Generated {ov.generated_at}. Row-level model checks ran on {int(ov.model_based_detectors_ran_on_rows)} rows{ov.model_based_detectors_sampled ? " (a stratified sample)" : ""}. {cal.status === "ok" ? `Noise floor ${num(cal.noise_floor)}.` : ""}</p>
      </Disclosure>
    </Page>
  );
}
